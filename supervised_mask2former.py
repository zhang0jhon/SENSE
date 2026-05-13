import argparse
import logging
import glob
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset, GenSemiDataset, Mask2FormerSemiDataset
from model.semseg.mask2former import Mask2Former
from model.util.matcher import HungarianMatcher
from model.util.criterion import SetCriterion
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed

from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser(description='Fully-Supervised Training in Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--pretrained-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


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
                    # pred = model(img)
                    outputs = model(img)
                    masks_classes = outputs["pred_logits"].softmax(dim=-1)[..., :-1]
                    masks_probs = outputs["pred_masks"].sigmoid()  # [batch_size, num_queries, height, width]
                    pred = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
            
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


def _collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets

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

    cfg['batch_size'] *= 2
    
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
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    criterion = SetCriterion(
        cfg['nclass'],
        HungarianMatcher(num_points=112 * 112, align_corners=cfg['align_corners']), 
        weight_dict=dict(loss_dice=5.0, loss_mask=5.0, loss_ce=2.0),
        num_points=112 * 112,
        align_corners=cfg['align_corners'],
    ).cuda(local_rank)
    
    # n_upsampled = {
    #     'pascal': 3000, 
    #     'cityscapes': 3000, 
    #     'ade20k': 6000, 
    #     'coco': 30000
    # }
    n_upsampled = {
        'pascal': 1464,
        'cityscapes': 2975,
        'ade20k': 20210, 
        'coco': 118287, #30000
    }
    trainset = Mask2FormerSemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=n_upsampled[cfg['dataset']]
    )
    valset = Mask2FormerSemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )
    
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(
        trainset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler, #collate_fn=_collate_fn,
    )
    
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler,
    )
    
    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    scaler = GradScaler()
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0 and epoch != 0:
            logger.info('===========> Epoch: {:}, LR: {:.7f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        criterion.train()
        
        total_loss = AverageMeter()
        total_loss_dice = AverageMeter()
        total_loss_ce = AverageMeter()
        total_loss_mask = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()
            
            
            # mask[mask==255] = cfg['nclass']
            mask = F.interpolate(mask.unsqueeze(1).float(), (cfg['crop_size'] // 4, cfg['crop_size'] // 4), mode="nearest").squeeze(1).long()

            optimizer.zero_grad()
            criterion.zero_grad()

            loss_ce = 0.0
            loss_dice = 0.0
            loss_mask = 0.0
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = model(img)
                losses = criterion(pred, mask)
                # print(losses.keys())
                for k in losses.keys():
                    if "dice" in k:
                        loss_dice += losses[k] * criterion.weight_dict["loss_dice"]

                    elif "mask" in k:
                        loss_mask += losses[k] * criterion.weight_dict["loss_mask"]

                    elif "ce" in k:
                        loss_ce += losses[k] * criterion.weight_dict["loss_ce"]
                
                loss = loss_dice + loss_mask + loss_ce

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            scaler.update()
            
            total_loss.update(loss.item())
            total_loss_dice.update(loss_dice.item())
            total_loss_ce.update(loss_ce.item())
            total_loss_mask.update(loss_mask.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            optimizer.param_groups[2]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_dice', loss_dice.item(), iters)
                writer.add_scalar('train/loss_mask', loss_mask.item(), iters)
                writer.add_scalar('train/loss_ce', loss_ce.item(), iters)
            
            if (i % (len(trainloader) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, loss dice: {:.3f}, loss mask: {:.3f}, loss ce: {:.3f}'.format(i, total_loss.avg, total_loss_dice.avg, total_loss_mask.avg, total_loss_ce.avg))
        
        if epoch % cfg['eval_epoch'] == 0:
            # eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
            eval_mode = 'original'
            mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=cfg['patch_size'])
            
            if rank == 0:
                for (cls_idx, iou) in enumerate(iou_class):
                    logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                                'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
                logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
                
                writer.add_scalar('eval/mIoU', mIoU, epoch)
                for i, iou in enumerate(iou_class):
                    writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
            
            is_best = mIoU > previous_best
            previous_best = max(mIoU, previous_best)
            if rank == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'previous_best': previous_best,
                }
                torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
                if is_best:
                    torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
