import argparse
from copy import deepcopy
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.sampler import FixedSizeDistributedSampler
from dataset.semi import SemiDataset, GenSemiDataset, Mask2FormerSemiDataset
from model.semseg.mask2former import Mask2Former
from model.util.matcher import HungarianMatcher, SSLHungarianMatcher
from model.util.criterion import SetCriterion, SSLSetCriterion
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed

from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import numpy as np
from collections import Counter

import torch.utils.checkpoint as cp

parser = argparse.ArgumentParser(description='Reproduced FixMatch with an EMA Teacher for Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--tau', type=float, default=0.05)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

import torch.distributed as dist

@torch.no_grad()
def distributed_sinkhorn(out, tau=0.05, sinkhorn_iterations=3):
    Q = torch.exp(out / tau).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * dist.get_world_size() # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        # deal with the class with 0 pixels
        sum_of_rows[sum_of_rows==0] = 1e-5 # does not matter due to 0 in nominator
        Q /= sum_of_rows
        Q /= K

        # # normalize each column: total weight per sample must be 1/B
        # Q /= torch.sum(Q, dim=0, keepdim=True)
        # Q /= B

        # normalize each column: total weight per sample must be 1/B
        sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
        sum_of_cols[sum_of_cols==0] = 1e-5 # does not matter due to 0 in nominator
        # print(sum_of_cols, sum_of_cols.shape)
        Q /= sum_of_cols # torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


def evaluate(model, loader, mode, cfg, multiplier=None):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask in loader:
            
            img = img.cuda()
                
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, cfg['nclass'], h, w).cuda()
                count = torch.zeros(b, cfg['nclass'], h, w).cuda()
                
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            outputs = model(img[:, :, row: row + grid, col: col + grid])
                            masks_classes = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
                            masks_probs = outputs["pred_masks"].sigmoid()  # [batch_size, num_queries, height, width]
                            pred = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
                            pred = F.interpolate(pred, (grid, grid), mode='bilinear', align_corners=cfg['align_corners'])

                            final[:, :, row: row + grid, col: col + grid] += pred
                            count[:, :, row: row + grid, col: col + grid] += 1
                        if col == w - grid:
                            break
                        col = min(col + int(grid * 2 / 3), w - grid)
                    if row == h - grid:
                        break
                    row = min(row + int(grid * 2 / 3), h - grid)
                    
                pred = final / count
            
            else:
                assert mode == 'original'
                
                if multiplier is not None:
                    ori_h, ori_w = img.shape[-2:]
                    if multiplier == 512:
                        new_h, new_w = 512, 512
                    else:
                        new_h, new_w = int(ori_h / multiplier + 0.5) * multiplier, int(ori_w / multiplier + 0.5) * multiplier
                    img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=cfg['align_corners'])
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(img)
                    mask_classes = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
                    mask_probs = outputs["pred_masks"].sigmoid()  # [batch_size, num_queries, height, width]
                    pred = torch.einsum("bqc, bqhw -> bchw", mask_classes, mask_probs)
            
                if multiplier is not None:
                    pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=cfg['align_corners'])
            
            pred = pred.argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    # rank, world_size = setup_distributed(port=args.port)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'dinov3_large': {'encoder_size': 'dinov3_large', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'num_queries': 100},
    }
    model = Mask2Former(**{**model_configs[cfg['backbone']], 'nclass': cfg['nclass'], 'align_corners': cfg['align_corners']})
    
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)
    
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name and 'query_feat' not in name and 'query_embed' not in name and 'level_embed' not in name], 'lr': cfg['lr'] * cfg['lr_multi']},
            {'params': [param for name, param in model.named_parameters() if 'query_feat' in name or 'query_embed' in name or 'level_embed' in name], 'lr': cfg['lr'] * cfg['lr_multi'], 'weight_decay': 0}, 
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.05
    )
    
    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    
    criterion_l = SetCriterion(
        cfg['nclass'],
        HungarianMatcher(num_points=112 * 112, align_corners=cfg['align_corners']),
        weight_dict=dict(loss_dice=5.0, loss_mask=5.0, loss_ce=2.0),
        num_points=112 * 112,
        align_corners=cfg['align_corners'],
    ).cuda(local_rank)

    criterion_u = SSLSetCriterion(
        cfg['nclass'],
        SSLHungarianMatcher(cost_dice=0, num_points=112 * 112, align_corners=cfg['align_corners']),
        # # for ADE20K
        # weight_dict=dict(loss_dice=0, loss_mask=5.0, loss_ce=1.0),
        # eos_coef=0.01,
        weight_dict=dict(loss_dice=0, loss_mask=5.0, loss_ce=2.0),
        eos_coef=0.02,
        num_points=112 * 112,
        align_corners=cfg['align_corners'],
    ).cuda(local_rank)

    trainset_u = Mask2FormerSemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    trainset_l = Mask2FormerSemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids) # nsample=n_upsampled[cfg['dataset']] # nsample=len(trainset_u.ids)  # nsample=n_upsampled[cfg['dataset']]
    )
    valset = Mask2FormerSemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )
    
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l
    )
    
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    # trainsampler_u = FixedSizeDistributedSampler(trainset_u, n_upsampled[cfg['dataset']] // torch.distributed.get_world_size())
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u
    )
    
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    )

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best_ema = 0.0
    best_epoch_ema = 0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch_ema = checkpoint['best_epoch_ema']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    
    scaler = GradScaler()
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, Previous best EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best_ema, best_epoch_ema))
        
        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_x_dice = AverageMeter()
        total_loss_x_mask = AverageMeter()
        total_loss_x_ce = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_s_dice = AverageMeter()
        total_loss_s_mask = AverageMeter()
        total_loss_s_ce = AverageMeter()
        total_mask_ratio = AverageMeter()
        total_num_masks = AverageMeter()
        # total_pos_masks = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        
        model.train()
        criterion_l.train()
        criterion_u.train()

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s, ignore_mask, cutmix_box)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            
            # mask_x[mask_x==255] = cfg['nclass']
            # mask_x = F.interpolate(mask_x.unsqueeze(1).float(), (cfg['crop_size'] // 4, cfg['crop_size'] // 4), mode="nearest").squeeze(1).long()
            
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
            ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()

            optimizer.zero_grad()
            criterion_l.zero_grad()
            criterion_u.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    # outputs_u_w = model_ema(img_u_w)
                    
                    pred_u_w_v1_logits_list, pred_u_w_v1_results_list = [], []
                    hflip = True
                    # sizes = [cfg['crop_size'], int(cfg['crop_size']*1.25)]
                    sizes = [cfg['crop_size']] if cfg['dataset'] == 'cityscapes' else [cfg['crop_size'], int(cfg['crop_size']*1.25)]
                    for size in sizes: 
                        img_u_w_resize = F.interpolate(img_u_w[:cfg['batch_size']//2], (size, size), mode='bilinear', align_corners=cfg['align_corners'])
                        pred_u_w_resize = model_ema(img_u_w_resize)
                        pred_u_w_v1_logits_list.append(pred_u_w_resize["pred_logits"])
                        pred_u_w_v1_result = torch.einsum("bqc, bqhw -> bchw", pred_u_w_resize["pred_logits"].softmax(dim=-1)[..., :-1], pred_u_w_resize["pred_masks"].sigmoid())
                        pred_u_w_v1_result = F.interpolate(pred_u_w_v1_result, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=cfg['align_corners'])
                        pred_u_w_v1_results_list.append(pred_u_w_v1_result)

                        if hflip:
                            pred_u_w_resize_hflip = model_ema(img_u_w_resize.flip(3))
                            pred_u_w_v1_logits_list.append(pred_u_w_resize_hflip["pred_logits"])
                            pred_u_w_v1_result = torch.einsum("bqc, bqhw -> bchw", pred_u_w_resize_hflip["pred_logits"].softmax(dim=-1)[..., :-1], pred_u_w_resize_hflip["pred_masks"].flip(3).sigmoid())
                            pred_u_w_v1_result = F.interpolate(pred_u_w_v1_result, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=cfg['align_corners'])
                            pred_u_w_v1_results_list.append(pred_u_w_v1_result)
                    
                    pred_u_w_v1_logits = torch.mean(torch.stack(pred_u_w_v1_logits_list, dim=0), dim=0)  
                    pred_u_w_v1_results = torch.mean(torch.stack(pred_u_w_v1_results_list, dim=0), dim=0)  
                    
                    pred_u_w_v2 = model_ema(img_u_w[cfg['batch_size']//2:])
                    pred_u_w_v2_logits = pred_u_w_v2["pred_logits"]
                    pred_u_w_v2_results = torch.einsum("bqc, bqhw -> bchw", pred_u_w_v2["pred_logits"].softmax(dim=-1)[..., :-1], pred_u_w_v2["pred_masks"].sigmoid())
                    pred_u_w_v2_results = F.interpolate(pred_u_w_v2_results, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=cfg['align_corners'])
                    
                    pred_u_w_logits = torch.cat((pred_u_w_v1_logits, pred_u_w_v2_logits), dim=0).detach()
                    pred_u_w_results = torch.cat((pred_u_w_v1_results, pred_u_w_v2_results), dim=0).detach()

                img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = img_u_s.flip(0)[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]
                
                num_lb, num_ulb = img_x.shape[0], img_u_s.shape[0]
                preds = model(torch.cat((img_x, img_u_s)))

                pred_x, pred_u_s = {}, {}
                for k, v in preds.items():
                    if "aux_outputs" in k:
                        pred_x_aux, pred_u_s_aux = [], []
                        for pred_aux in v:
                            aux_x_logits, aux_u_s_logits = pred_aux["pred_logits"].split([num_lb, num_ulb])
                            aux_x_masks, aux_u_s_masks = pred_aux["pred_masks"].split([num_lb, num_ulb])
                            pred_x_aux.append({"pred_logits": aux_x_logits, "pred_masks": aux_x_masks})
                            pred_u_s_aux.append({"pred_logits": aux_u_s_logits, "pred_masks": aux_u_s_masks})
                        pred_x[k] = pred_x_aux
                        pred_u_s[k] = pred_u_s_aux
                    else:
                        pred_x[k], pred_u_s[k] = preds[k].split([num_lb, num_ulb])

                losses_x = criterion_l(pred_x, mask_x)
                
                loss_ce_x = 0.0
                loss_dice_x = 0.0
                loss_mask_x = 0.0

                for k in losses_x.keys():
                    if "dice" in k:
                        loss_dice_x += losses_x[k] * criterion_l.weight_dict["loss_dice"]
                    elif "mask" in k:
                        loss_mask_x += losses_x[k] * criterion_l.weight_dict["loss_mask"]
                    elif "ce" in k:
                        loss_ce_x += losses_x[k] * criterion_l.weight_dict["loss_ce"]

                loss_x = loss_dice_x + loss_mask_x + loss_ce_x
                
                # cutmix_box = F.interpolate(cutmix_box.unsqueeze(1), (cfg['crop_size'] // 4, cfg['crop_size'] // 4), mode="nearest").squeeze(1)
                # ignore_mask = F.interpolate(ignore_mask.unsqueeze(1).float(), (cfg['crop_size'] // 4, cfg['crop_size'] // 4), mode="nearest").long().squeeze(1)

                pred_logits_w, pred_u_w_cutmixed, ignore_mask_cutmixed = pred_u_w_logits.clone(), pred_u_w_results.clone(), ignore_mask.clone()
                pred_u_w_cutmixed[cutmix_box.unsqueeze(1).expand(pred_u_w_cutmixed.shape) == 1] = pred_u_w_cutmixed.flip(0)[cutmix_box.unsqueeze(1).expand(pred_u_w_cutmixed.shape) == 1]
                ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask.flip(0)[cutmix_box == 1]
                
                b, c, h, w = pred_u_w_cutmixed.shape

                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        logscores = F.log_softmax(pred_u_w_cutmixed.float().permute(0, 2, 3, 1).contiguous().view(-1, c), dim=1)
                        pred_assignments = distributed_sinkhorn(logscores, tau=0.05).view(b, h, w, c).permute(0, 3, 1, 2).contiguous().detach()
                        if torch.isnan(pred_assignments).any():
                            print("assignments contain nan, use pseudo label.")
                            pred_assignments = pred_u_w_cutmixed.detach()


                conf_u_w_cutmixed, pseudo_labels = torch.max(pred_assignments, dim=1)

                # filter loss mask 
                scores_with_noobj, _ = pred_logits_w.softmax(-1).max(dim=-1)   # [b, q]
                ignore_mask_cutmixed[scores_with_noobj.mean(-1) < cfg['conf_thresh']] = 255  # ignore the whole low conf image

                pseudo_labels[conf_u_w_cutmixed < cfg['conf_thresh']] = 255   # ignore the low conf region
                pseudo_labels[ignore_mask_cutmixed==255] = 255     # ignore the padding region & low conf image
                
                
                losses_u_s, num_masks = criterion_u(pred_u_s, pseudo_labels)

                loss_ce_u_s = 0.0
                loss_dice_u_s = 0.0
                loss_mask_u_s = 0.0

                for k in losses_x.keys():
                    if "dice" in k:
                        loss_dice_u_s += losses_u_s[k] * criterion_u.weight_dict["loss_dice"]
                    if "mask" in k:
                        loss_mask_u_s += losses_u_s[k] * criterion_u.weight_dict["loss_mask"]
                    elif "ce" in k:
                        loss_ce_u_s += losses_u_s[k] * criterion_u.weight_dict["loss_ce"]

                loss_u_s = loss_dice_u_s + loss_mask_u_s + loss_ce_u_s

                loss = (loss_x + loss_u_s) / 2.0
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            scaler.update()
            
                
            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            total_loss_x_dice.update(loss_dice_x.item())
            total_loss_s_dice.update(loss_dice_u_s.item())
            total_loss_x_mask.update(loss_mask_x.item())
            total_loss_s_mask.update(loss_mask_u_s.item())
            total_loss_x_ce.update(loss_ce_x.item())
            total_loss_s_ce.update(loss_ce_u_s.item())
            mask_ratio = ((conf_u_w_cutmixed >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())
            total_num_masks.update(num_masks)
            # total_pos_masks.update(num_pos_masks)

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            optimizer.param_groups[2]["lr"] = lr * cfg['lr_multi']
            
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/loss_dice_x', loss_dice_x.item(), iters)
                writer.add_scalar('train/loss_dice_u_s', loss_dice_u_s.item(), iters)
                writer.add_scalar('train/loss_mask_x', loss_mask_x.item(), iters)
                writer.add_scalar('train/loss_mask_u_s', loss_mask_u_s.item(), iters)
                writer.add_scalar('train/loss_ce_x', loss_ce_x.item(), iters)
                writer.add_scalar('train/loss_ce_u_s', loss_ce_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
                writer.add_scalar('train/num_masks', num_masks, iters)
                
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss x dice: {:.3f}, Loss x mask: {:.3f}, Loss x ce: {:.3f}, Loss s dice: {:.3f}, Loss s mask: {:.3f}, Loss s ce: {:.3f}, Mask ratio: {:.3f}, Num masks: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg, total_loss_s.avg, total_loss_x_dice.avg, total_loss_x_mask.avg, total_loss_x_ce.avg, 
                                            total_loss_s_dice.avg, total_loss_s_mask.avg, total_loss_s_ce.avg, total_mask_ratio.avg, total_num_masks.avg))
            
        
        # eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        eval_mode = 'original'
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=cfg['patch_size'])
        
        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class_ema):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU EMA: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU EMA: {:.2f}\n'.format(eval_mode, mIoU_ema))
            
            writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
            for i, iou in enumerate(iou_class_ema):
                writer.add_scalar('eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i]), iou, epoch)
        
        is_best = mIoU_ema >= previous_best_ema
        
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best_ema': previous_best_ema,
                'best_epoch_ema': best_epoch_ema
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
