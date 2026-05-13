import torch
import torch.nn.functional as F
import torch.distributed as dist

def build_mask2former_instance_pseudo_targets(pred_logits, pred_masks,
                                              conf_thres=0.9, min_area=64):
    """
    将 Teacher 输出转为 Mask2Former 格式的实例分割伪标签 target
    - 输入:
        pred_logits: [Q, C+1]  (最后一类是 no-object)
        pred_masks:  [Q, H, W]
    - 输出:
        {"labels": Tensor[N], "masks": Tensor[N,H,W]}
        N = 保留下来的实例数
    """
    # [Q, C+1] → [Q, C]
    probs = F.softmax(pred_logits, dim=-1)
    scores, classes = probs[..., :-1].max(dim=-1)   # [Q]

    # mask 概率
    masks_prob = pred_masks.sigmoid()               # [Q,H,W]

    labels, mask_list = [], []

    for i in range(len(classes)):
        if scores[i] < conf_thres:
            continue

        mask_bin = (masks_prob[i] > 0.5).float()    # 二值化

        if mask_bin.sum() < min_area:
            continue

        labels.append(classes[i])
        mask_list.append(mask_bin)

    if len(labels) == 0:
        return {
            "labels": torch.zeros(0, dtype=torch.long),
            "masks": torch.zeros(0, *pred_masks.shape[-2:])
        }

    return {
        "labels": torch.stack(labels),      # [N]
        "masks": torch.stack(mask_list)     # [N,H,W]
    }


def build_mask2former_semantic_pseudo_targets(pred_logits, pred_masks,
                                            conf_thres=0.9, min_area=64):
    """
    将 Teacher 输出转为 Mask2Former 格式的语义分割伪标签 target
    - 输入:
        pred_logits: [Q, C+1]
        pred_masks:  [Q, H, W]
    - 输出:
        {"labels": Tensor[K], "masks": Tensor[K,H,W]}
        K = 有效类别数 (每个类别一个mask)
    """
    probs = F.softmax(pred_logits, dim=-1)   # [Q,C+1]
    scores, classes = probs[..., :-1].max(dim=-1)  # 去掉 no-object 类
    masks_prob = pred_masks.sigmoid()        # [Q,H,W]

    # Step1: 过滤 query
    keep = (scores > conf_thres) & (masks_prob.view(masks_prob.size(0), -1).sum(dim=-1) > min_area)
    scores, classes, masks_prob = scores[keep], classes[keep], masks_prob[keep]

    if len(classes) == 0:
        return {
            "labels": torch.zeros(0, dtype=torch.long),
            "masks": torch.zeros(0, *pred_masks.shape[-2:])
        }

    # Step2: 按类别聚合
    unique_classes = classes.unique()
    labels, mask_list = [], []

    for c in unique_classes:
        idx = (classes == c).nonzero(as_tuple=False).squeeze(1)
        if len(idx) == 0:
            continue

        # 用类别分数加权平均聚合 query 的 mask 概率
        weights = scores[idx].view(-1, 1, 1)
        masks_c = masks_prob[idx]               # [N,H,W]
        mask_agg = (masks_c * weights).sum(dim=0) / (weights.sum() + 1e-6)

        # 二值化
        mask_bin = (mask_agg > 0.5).float()

        if mask_bin.sum() < min_area:
            continue

        labels.append(c)
        mask_list.append(mask_bin)

    if len(labels) == 0:
        return {
            "labels": torch.zeros(0, dtype=torch.long),
            "masks": torch.zeros(0, *pred_masks.shape[-2:])
        }

    return {
        "labels": torch.stack(labels),         # [K]
        "masks": torch.stack(mask_list)        # [K,H,W]
    }

def build_mask2former_panoptic_pseudo_targets(
    pred_logits, pred_masks, thing_classes,
    conf_thres=0.9, min_area=64
):
    """
    将 Teacher 输出转为 Mask2Former 格式的全景分割伪标签 target
    - 输入:
        pred_logits: [Q, C+1]
        pred_masks:  [Q, H, W]
        thing_classes: list[int], 哪些类别是 thing，其余视为 stuff
    - 输出:
        {"labels": Tensor[K], "masks": Tensor[K,H,W], "isthing": Tensor[K]}
        K = 有效实例 + stuff 区域数
    """
    probs = F.softmax(pred_logits, dim=-1)      # [Q, C+1]
    scores, classes = probs[..., :-1].max(dim=-1)  # [Q]
    masks_prob = pred_masks.sigmoid()           # [Q,H,W]

    labels, masks, isthing_flags = [], [], []

    # ---- 处理 thing 类（多个实例）----
    for i in range(len(classes)):
        c, s = int(classes[i]), float(scores[i])
        if s < conf_thres:
            continue
        if c not in thing_classes:
            continue

        mask_bin = (masks_prob[i] > 0.5).float()
        if mask_bin.sum() < min_area:
            continue

        labels.append(c)
        masks.append(mask_bin)
        isthing_flags.append(1)

    # ---- 处理 stuff 类（每类一个区域，需聚合）----
    stuff_classes = set(classes.tolist()) - set(thing_classes)
    for c in stuff_classes:
        idx = (classes == c).nonzero(as_tuple=False).squeeze(1)
        if len(idx) == 0:
            continue

        # 分数加权平均聚合多个 query
        weights = scores[idx].view(-1, 1, 1)
        masks_c = masks_prob[idx]
        mask_agg = (masks_c * weights).sum(dim=0) / (weights.sum() + 1e-6)

        mask_bin = (mask_agg > 0.5).float()
        if mask_bin.sum() < min_area:
            continue

        labels.append(int(c))
        masks.append(mask_bin)
        isthing_flags.append(0)

    if len(labels) == 0:
        H, W = pred_masks.shape[-2:]
        return {
            "labels": torch.zeros(0, dtype=torch.long),
            "masks": torch.zeros(0, H, W),
            "isthing": torch.zeros(0, dtype=torch.bool)
        }

    return {
        "labels": torch.tensor(labels, dtype=torch.long),
        "masks": torch.stack(masks),              # [K,H,W]
        "isthing": torch.tensor(isthing_flags, dtype=torch.bool)  # [K]
    }

def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2) # [c, self.num_points, 1, 2]
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs) # [c, 1, self.num_points, 1]
    if add_dim:
        output = output.squeeze(3)
    return output # [c, 1, self.num_points]



def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio, align_corners
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=align_corners)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    """
    Find `num_points` most uncertain points from `uncertainty_map` grid.

    Args:
        uncertainty_map (Tensor): A tensor of shape (N, 1, H, W) that contains uncertainty
            values for a set of points on a regular H x W grid.
        num_points (int): The number of points P to select.

    Returns:
        point_indices (Tensor): A tensor of shape (N, P) that contains indices from
            [0, H x W) of the most uncertain points.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1] normalized
            coordinates of the most uncertain points from the H x W grid.
    """
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=torch.float, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(torch.float) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(torch.float) * h_step
    return point_indices, point_coords

def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(tensor_list):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return tensor, mask


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True