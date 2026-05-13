# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn
from .common import (
    point_sample,
    nested_tensor_from_tensor_list,
    get_uncertain_point_coords_with_randomness,
    is_dist_avail_and_initialized,
    get_world_size,
)


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid().flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(
    inputs: torch.Tensor, labels: torch.Tensor, num_masks: int
) -> torch.Tensor:
    r"""
    Args:
        inputs (`torch.Tensor`):
            A float tensor of arbitrary shape.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss.
    """
    cross_entropy_loss = F.binary_cross_entropy_with_logits(
        inputs, labels, reduction="none"
    )
    loss = cross_entropy_loss.mean(1).sum() / num_masks
    return loss

def dice_loss_with_mask(inputs: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor, num_masks: float):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # print(inputs.shape, targets.shape, masks.shape)
    # inputs = inputs.sigmoid() * masks
    inputs = inputs.sigmoid() * masks[0][None, :] if masks.size(0) > 0 else inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets * masks
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def dice_loss_with_mask_v4(inputs: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor, num_masks: float):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid() * masks
    targets = targets * masks
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss_with_mask(
    inputs: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor, num_masks: int
) -> torch.Tensor:
    r"""
    Args:
        inputs (`torch.Tensor`):
            A float tensor of arbitrary shape.
        labels (`torch.Tensor`):
            A tensor with the same shape as inputs. Stores the binary classification labels for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        loss (`torch.Tensor`): The computed loss.
    """
    cross_entropy_loss = F.binary_cross_entropy_with_logits(
        inputs, labels, reduction="none"
    )
    # print(cross_entropy_loss.shape, inputs.shape, labels.shape, masks.shape, num_masks)  #  [9, 12544]
    # loss = (cross_entropy_loss * masks).sum() / (masks.sum() + 1e-6) 
    loss = ((cross_entropy_loss * masks).sum(1) / (masks.sum(1) + 1e-6)).sum() / num_masks 
    return loss

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks,
    alpha: float = 0.25,
    gamma: float = 2,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict=dict(loss_dice=5.0, loss_mask=5.0, loss_ce=2.0),
        cls_weights=None,
        eos_coef=0.1,
        losses=["labels", "masks"],
        num_points=112 * 112,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        ignore_index=255,
        align_corners=False,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)        

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.ignore_index = ignore_index
        self.align_corners = align_corners

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        # target_masks = torch.nested.as_nested_tensor(masks).to_padded_tensor(0)
        target_masks, _ = nested_tensor_from_tensor_list(masks)
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
                self.align_corners,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                # mode='nearest',
                align_corners=self.align_corners,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=self.align_corners,
        ).squeeze(1)

        # point_logits = src_masks.flatten(1)
        # point_labels = target_masks.flatten(1) 
        
        losses = {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, gt_masks):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             gt_masks: [bs, h_net_output, w_net_output]
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        targets = self._get_targets(gt_masks)
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def _get_binary_mask(self, target):
        y, x = target.size()
        target_onehot = torch.zeros(self.num_classes + 1, y, x, device=target.device)
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot
    
    def _get_targets(self, gt_masks):
        gt_masks[gt_masks==self.ignore_index] = self.num_classes
        targets = []
        for mask in gt_masks:
            binary_masks = self._get_binary_mask(mask)
            cls_label = torch.unique(mask)
            labels = cls_label[cls_label!=self.num_classes]
            if labels.size(0) == 0:
                binary_masks = torch.zeros((0, mask.shape[-2], mask.shape[-1]), device=mask.device)
            else:
                binary_masks = binary_masks[labels]
            targets.append({'masks': binary_masks, 'labels': labels})
        return targets
    
    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)



class SSLSetCriterion(nn.Module):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict=dict(loss_dice=5.0, loss_mask=5.0, loss_ce=2.0),
        cls_weights=None,
        eos_coef=0.1,
        losses=["labels", "masks"],
        num_points=112 * 112,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        ignore_index=255,
        align_corners=False,
    ):
        # """Create the criterion.
        # Parameters:
        #     num_classes: number of object categories, omitting the special no-object category
        #     matcher: module able to compute a matching between targets and proposals
        #     weight_dict: dict containing as key the names of the losses and as values their relative weight.
        #     eos_coef: relative classification weight applied to the no-object category
        #     losses: list of all the losses to be applied. See get_loss for list of available losses.
        # """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)        

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.ignore_index = ignore_index
        self.align_corners = align_corners

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()  

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        # print(target_classes_o.shape, target_classes.shape, src_logits.shape, idx)   # torch.Size([11]) torch.Size([8, 100])  torch.Size([8, 100, 20])
        target_classes[idx] = target_classes_o

        loss_mask = (target_classes!=self.num_classes).float().sum(-1) > 0
        
        if loss_mask.float().sum() > 0:
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2)[loss_mask], target_classes[loss_mask], self.empty_weight, # (8, 20, 100), (8, 100)
            )
        else:
            loss_ce = 0
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"] # ([8, 100, 228, 228])
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        # target_masks = torch.nested.as_nested_tensor(masks).to_padded_tensor(0)
        target_masks, _ = nested_tensor_from_tensor_list(masks)
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        
        high_conf_masks = [t["high_conf_mask"] for t in targets]
        high_conf_masks = torch.cat(high_conf_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        high_conf_masks = high_conf_masks[:, None]
        
        
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
                self.align_corners,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=self.align_corners,
            ).squeeze(1)

            # print(target_masks, target_masks.shape, target_masks.dtype)
            # print(point_labels, point_labels.shape, point_labels.dtype)

            point_masks = point_sample(
                high_conf_masks.float(),
                point_coords,
                mode='nearest',
                align_corners=self.align_corners,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=self.align_corners,
        ).squeeze(1)

        losses = {
            # "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            # "loss_dice": dice_loss(point_logits, point_labels, num_masks),
            "loss_mask": sigmoid_ce_loss_with_mask(point_logits, point_labels, point_masks, num_masks),
            "loss_dice": dice_loss_with_mask(point_logits, point_labels, point_masks, num_masks),
        }

        del src_masks
        del target_masks
        del high_conf_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, gt_masks):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        targets = self._get_targets(gt_masks)
        # print(outputs_without_aux, targets)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_masks
                    )
                    l_dict = {f"{k}_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, num_masks

    def _get_binary_mask(self, target):
        y, x = target.size()
        target_onehot = torch.zeros(self.num_classes + 1, y, x, device=target.device)
        target[target==self.ignore_index] = self.num_classes # does not matter due to high conf mask
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot
    
    def _get_targets(self, gt_masks):
        targets = []
        for mask in gt_masks:
            cls_label = torch.unique(mask)
            labels = cls_label[cls_label!=self.ignore_index]  # ignore 255
            binary_masks = self._get_binary_mask(mask)
            if labels.size(0) == 0:
                binary_masks = torch.zeros(0, *mask.shape[-2:], device=mask.device)
                high_conf_mask = torch.zeros(0, *mask.shape[-2:], device=mask.device)
                # labels = torch.zeros((0), device=mask.device)
            else:
                binary_masks = binary_masks[labels]
                high_conf_mask = (mask!=self.ignore_index).unsqueeze(0).repeat(labels.size(0), 1, 1)
            
            targets.append({'masks': binary_masks, 'labels': labels, 'high_conf_mask': high_conf_mask})
        return targets

    
    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
        
    

class SSLSetCriterionV2(nn.Module):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict=dict(loss_dice=5.0, loss_mask=5.0, loss_ce=2.0),
        cls_weights=None,
        eos_coef=0.1,
        losses=["labels", "masks"],
        num_points=112 * 112,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        ignore_index=255,
        align_corners=False,
    ):
        # """Create the criterion.
        # Parameters:
        #     num_classes: number of object categories, omitting the special no-object category
        #     matcher: module able to compute a matching between targets and proposals
        #     weight_dict: dict containing as key the names of the losses and as values their relative weight.
        #     eos_coef: relative classification weight applied to the no-object category
        #     losses: list of all the losses to be applied. See get_loss for list of available losses.
        # """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)        

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.ignore_index = ignore_index
        self.align_corners = align_corners

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()  

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        # print(target_classes_o.shape, target_classes.shape, src_logits.shape, idx)   # torch.Size([11]) torch.Size([8, 100])  torch.Size([8, 100, 20])
        target_classes[idx] = target_classes_o
        
        if target_classes_o.shape[0] > 0:
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight, # (8, 20, 100), (8, 100)
            )
        else:
            loss_ce = 0
        losses = {"loss_ce": loss_ce}
        
        # target_classes = torch.full(
        #     src_logits.shape[:2],
        #     self.num_classes,
        #     dtype=torch.int64,
        #     device=src_logits.device,
        # )
        # target_classes[idx] = target_classes_o
        
        # loss_ce = F.cross_entropy(
        #     src_logits.transpose(1, 2), target_classes, reduction='none' # (8, 20, 100), (8, 100)
        # )
        # fg_mask = target_classes != self.num_classes
        # bg_mask = target_classes == self.num_classes
        # loss_ce_fg = loss_ce[fg_mask].sum() / (fg_mask.sum() + 1e-6) 
        # loss_ce_bg =  loss_ce[bg_mask].sum() / (bg_mask.sum() + 1e-6) 
        # loss_ce_final = (loss_ce_fg + loss_ce_bg) / 2
        # losses = {"loss_ce": loss_ce_final}
        
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"] # ([8, 100, 228, 228])
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        # target_masks = torch.nested.as_nested_tensor(masks).to_padded_tensor(0)
        target_masks, _ = nested_tensor_from_tensor_list(masks)
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        
        high_conf_masks = [t["high_conf_mask"] for t in targets]
        high_conf_masks = torch.cat(high_conf_masks)

        # print(src_masks.shape, target_masks.shape, high_conf_masks.shape)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        high_conf_masks = high_conf_masks[:, None]
        
        # print(src_idx, tgt_idx, src_masks.shape, target_masks.shape, high_conf_masks.shape)  # torch.Size([17, 1, 228, 228]) torch.Size([17, 1, 228, 228]) torch.Size([8, 1, 228, 228])

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
                self.align_corners,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=self.align_corners,
            ).squeeze(1)

            # print(target_masks, target_masks.shape, target_masks.dtype)
            # print(point_labels, point_labels.shape, point_labels.dtype)

            point_masks = point_sample(
                high_conf_masks.float(),
                point_coords,
                mode='nearest',
                align_corners=self.align_corners,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=self.align_corners,
        ).squeeze(1)

        # point_logits = src_masks.flatten(1)
        # point_labels = target_masks.flatten(1) 
        # point_masks = high_conf_masks.flatten(1) 

        # print(high_conf_masks, high_conf_masks.shape)
        # print(point_coords.shape, point_logits.shape, point_labels.shape, point_masks.shape)  # torch.Size([12, 12544, 2]) torch.Size([12, 12544]) torch.Size([12, 12544]) torch.Size([12, 12544])
        # print(point_masks, point_masks.shape, point_masks.dtype)

        losses = {
            # "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            # "loss_dice": dice_loss(point_logits, point_labels, num_masks),
            "loss_mask": sigmoid_ce_loss_with_mask(point_logits, point_labels, point_masks, num_masks),
            "loss_dice": dice_loss_with_mask(point_logits, point_labels, point_masks, num_masks),
        }

        del src_masks
        del target_masks
        del high_conf_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, gt_masks):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        targets = self._get_targets(gt_masks)
        # print(outputs_without_aux, targets)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_masks
                    )
                    l_dict = {f"{k}_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, num_masks

    def _get_binary_mask(self, target):
        y, x = target.size()
        target_onehot = torch.zeros(self.num_classes + 1, y, x, device=target.device)
        target[target==self.ignore_index] = self.num_classes # does not matter due to high conf mask
        target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
        return target_onehot
    
    def _get_targets(self, gt_masks):
        targets = []
        for mask in gt_masks:
            cls_label = torch.unique(mask)
            labels = cls_label[cls_label!=self.ignore_index]  # ignore 255
            binary_masks = self._get_binary_mask(mask)
            if labels.size(0) == 0:
                binary_masks = torch.zeros(0, *mask.shape[-2:], device=mask.device)
                high_conf_mask = torch.zeros(0, *mask.shape[-2:], device=mask.device)
                # labels = torch.zeros((0), device=mask.device)
            else:
                binary_masks = binary_masks[labels]
                high_conf_mask = (mask!=self.ignore_index).unsqueeze(0).repeat(labels.size(0), 1, 1)
            
            targets.append({'masks': binary_masks, 'labels': labels, 'high_conf_mask': high_conf_mask})
        return targets


    # def _get_binary_mask(self, target):
    #     y, x = target.size()
    #     target_onehot = torch.zeros(self.num_classes + 1, y, x, device=target.device)
    #     target_onehot = target_onehot.scatter(dim=0, index=target.unsqueeze(0), value=1)
    #     return target_onehot
    
    # def _get_targets(self, gt_masks):
    #     gt_masks[gt_masks==self.ignore_index] = self.num_classes
    #     targets = []
    #     for mask in gt_masks:
    #         cls_label = torch.unique(mask)
    #         labels = cls_label[cls_label!=self.num_classes]  # ignore 255
    #         # print(labels)
    #         binary_masks = self._get_binary_mask(mask)
    #         if labels.size(0) == 0:
    #             binary_masks = torch.zeros(0, *mask.shape[-2:], device=mask.device)
    #             high_conf_mask = torch.zeros(0, *mask.shape[-2:], device=mask.device)
    #             # labels = torch.zeros(0, dtype=torch.long, device=mask.device)
    #         else:
    #             binary_masks = binary_masks[labels]
    #             high_conf_mask = (mask!=self.num_classes).unsqueeze(0).repeat(labels.size(0), 1, 1)
            
    #         targets.append({'masks': binary_masks, 'labels': labels, 'high_conf_mask': high_conf_mask})
    #     return targets
    
    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
    


class SSLSetCriterionV3(nn.Module):
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict=dict(loss_dice=5.0, loss_mask=5.0, loss_ce=2.0),
        cls_weights=None,
        eos_coef=0.1,
        losses=["labels", "masks"],
        num_points=112 * 112,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        ignore_index=255,
        align_corners=False,
    ):
        # """Create the criterion.
        # Parameters:
        #     num_classes: number of object categories, omitting the special no-object category
        #     matcher: module able to compute a matching between targets and proposals
        #     weight_dict: dict containing as key the names of the losses and as values their relative weight.
        #     eos_coef: relative classification weight applied to the no-object category
        #     losses: list of all the losses to be applied. See get_loss for list of available losses.
        # """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)        

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.ignore_index = ignore_index
        self.align_corners = align_corners

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()  

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        # print(target_classes_o.shape, target_classes.shape, src_logits.shape, idx)   # torch.Size([11]) torch.Size([8, 100])  torch.Size([8, 100, 20])
        target_classes[idx] = target_classes_o
        
        if target_classes_o.shape[0] > 0:
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight, # (8, 20, 100), (8, 100)
            )
        else:
            loss_ce = 0
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"] # ([8, 100, 228, 228])
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        # target_masks = torch.nested.as_nested_tensor(masks).to_padded_tensor(0)
        target_masks, _ = nested_tensor_from_tensor_list(masks)
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        
        keep_masks = [t["keep_masks"] for t in targets]
        keep_masks = torch.cat(keep_masks)

        # print(src_masks.shape, target_masks.shape, keep_masks.shape)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        keep_masks = keep_masks[:, None]
        
        # print(src_idx, tgt_idx, src_masks.shape, target_masks.shape, keep_masks.shape)  # torch.Size([17, 1, 228, 228]) torch.Size([17, 1, 228, 228]) torch.Size([8, 1, 228, 228])

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
                self.align_corners,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=self.align_corners,
            ).squeeze(1)

            # print(target_masks, target_masks.shape, target_masks.dtype)
            # print(point_labels, point_labels.shape, point_labels.dtype)

            point_masks = point_sample(
                keep_masks.float(),
                point_coords,
                mode='nearest',
                align_corners=self.align_corners,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=self.align_corners,
        ).squeeze(1)

        # point_logits = src_masks.flatten(1)
        # point_labels = target_masks.flatten(1) 
        # point_masks = keep_masks.flatten(1) 

        # print(keep_masks, keep_masks.shape)
        # print(point_coords.shape, point_logits.shape, point_labels.shape, point_masks.shape)  # torch.Size([12, 12544, 2]) torch.Size([12, 12544]) torch.Size([12, 12544]) torch.Size([12, 12544])
        # print(point_masks, point_masks.shape, point_masks.dtype)

        losses = {
            # "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            # "loss_dice": dice_loss(point_logits, point_labels, num_masks),
            "loss_mask": sigmoid_ce_loss_with_mask(point_logits, point_labels, point_masks, num_masks),
            "loss_dice": dice_loss_with_mask(point_logits, point_labels, point_masks, num_masks),
        }

        del src_masks
        del target_masks
        del keep_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        # targets = self._get_targets(gt_masks)
        # print(outputs_without_aux, targets)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_masks
                    )
                    l_dict = {f"{k}_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, num_masks
    
    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
    


class SSLSetCriterionV4(nn.Module):
    def __init__(
        self,
        num_classes,
        weight_dict=dict(loss_dice=5.0, loss_mask=5.0, loss_ce=2.0),
        eos_coef=0.1,
        losses=["labels", "masks"],
        num_points=112 * 112,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        ignore_index=255,
        align_corners=False,
    ):
        # """Create the criterion.
        # Parameters:
        #     num_classes: number of object categories, omitting the special no-object category
        #     weight_dict: dict containing as key the names of the losses and as values their relative weight.
        #     eos_coef: relative classification weight applied to the no-object category
        #     losses: list of all the losses to be applied. See get_loss for list of available losses.
        # """
        super().__init__()
        self.num_classes = num_classes
        # self.matcher = matcher 
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)        

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.ignore_index = ignore_index
        self.align_corners = align_corners

    def loss_labels(self, outputs, targets, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()  
        # src_logits = src_logits.view(-1, src_logits.shape[-1])

        target_classes = torch.stack([t["labels"] for t in targets])  # [b, q]
        keep = torch.stack([t["keep"] for t in targets])   # [b, q]
        assignments = torch.stack([t["assignments"] for t in targets])  # [b, q, c+1]

        if keep.sum() > 0:
            loss_ce = F.cross_entropy(
                    src_logits[keep], target_classes[keep], self.empty_weight, 
                )
            # loss_ce = torch.sum(-assignments[keep] * F.log_softmax(src_logits[keep], dim=-1), dim=-1)  # without weights
            # loss_ce = torch.sum(-self.empty_weight.unsqueeze(0) * assignments[keep] * F.log_softmax(src_logits[keep], dim=-1), dim=-1)  # with weights
            # loss_ce = loss_ce.mean()
        else:
            loss_ce = 0
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_masks = outputs["pred_masks"] # ([8, 100, 228, 228])
        
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        # target_masks = torch.nested.as_nested_tensor(masks).to_padded_tensor(0)
        target_masks, _ = nested_tensor_from_tensor_list(masks)
        target_masks = target_masks.to(src_masks)

        keep = torch.stack([t["keep"] for t in targets])
        keep_masks = torch.stack([t["keep_masks"] for t in targets])

        src_masks = src_masks[keep]
        target_masks = target_masks[keep]
        keep_masks = keep_masks[keep]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        keep_masks = keep_masks[:, None]

        # print(src_masks.shape, target_masks.shape, keep.shape, keep_masks.shape)

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
                self.align_corners,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=self.align_corners,
            ).squeeze(1)

            point_masks = point_sample(
                keep_masks.float(),
                point_coords,
                mode='nearest',
                align_corners=self.align_corners,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=self.align_corners,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_with_mask(point_logits, point_labels, point_masks, num_masks),
            "loss_dice": dice_loss_with_mask_v4(point_logits, point_labels, point_masks, num_masks),
        }

        del src_masks
        del target_masks
        del keep
        del keep_masks
        return losses

    def get_loss(self, loss, outputs, targets, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        # targets = self._get_targets(gt_masks)
        # print(outputs_without_aux, targets)

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = torch.stack([t["keep"] for t in targets]).sum() #sum(len(t["labels"]) for t in targets)
        # num_masks = torch.as_tensor(
        #     [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        # )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, num_masks
                    )
                    l_dict = {f"{k}_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses, num_masks
    
    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            # "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)