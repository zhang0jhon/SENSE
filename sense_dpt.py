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
from dataset.semi import SemiDataset, GenSemiDataset, GenSemiDatasetV2
from model.semseg.dpt import DPT
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
    # x = torch.clamp(x, min=-100)
    # x = x - x.max(dim=1, keepdim=True)[0]
    Q = torch.exp(out / tau).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * dist.get_world_size() # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q
    # print(sum_Q)

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        # deal with the class with 0 pixels
        sum_of_rows[sum_of_rows==0] = 1e-5 # does not matter due to 0 in nominator
        # print(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

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
    assert mode in ['original', 'sliding_window', 'tta']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:
            
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
                            pred = model(img[:, :, row: row + grid, col: col + grid])
                            final[:, :, row: row + grid, col: col + grid] += pred.softmax(dim=1)
                            count[:, :, row: row + grid, col: col + grid] += 1
                        if col == w - grid:
                            break
                        col = min(col + int(grid * 2 / 3), w - grid)
                    if row == h - grid:
                        break
                    row = min(row + int(grid * 2 / 3), h - grid)

                pred = final / count
            
            elif mode == 'tta':
                ori_h, ori_w = img.shape[-2:]
                preds = []
                flip = True
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    for scale in [1.0, 1.25]:
                        scale_h, scale_w = int(ori_h * scale / multiplier + 0.5) * multiplier, int(ori_w * scale / multiplier + 0.5) * multiplier
                        img = F.interpolate(img, (scale_h, scale_w), mode='bilinear', align_corners=True)
                        pred = model(img)
                        pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)
                        preds.append(pred)
                        
                        if flip:
                            pred_hflip = model(img.flip(3))
                            pred = F.interpolate(pred_hflip.flip(3), (ori_h, ori_w), mode='bilinear', align_corners=True)
                            preds.append(pred)
                            
                pred = torch.mean(torch.stack(preds, dim=0), dim=0)
                
            else:
                assert mode == 'original'
                
                if multiplier is not None:
                    ori_h, ori_w = img.shape[-2:]
                    if multiplier == 512:
                        new_h, new_w = 512, 512
                    else:
                        new_h, new_w = int(ori_h / multiplier + 0.5) * multiplier, int(ori_w / multiplier + 0.5) * multiplier
                    img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred = model(img)
            
                if multiplier is not None:
                    pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)
            
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

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    # model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass'], 'encoder': cfg['backbone'].split('_')[0], 'align_corners': cfg['align_corners']})
    
    
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)
    
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
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
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    
    # n_upsampled = {
    #     'pascal': 3000, 
    #     'cityscapes': 3000, 
    #     'ade20k': 6000, 
    #     'coco': 30000
    # }
    
    trainset_u = GenSemiDatasetV2(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids) # nsample=n_upsampled[cfg['dataset']] # nsample=len(trainset_u.ids)  # nsample=n_upsampled[cfg['dataset']]
    )
    valset = SemiDataset(
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
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
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
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        
        model.train()

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s, ignore_mask, cutmix_box)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
            ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()

            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    preds_u_w = []
                    hflip = True
                    sizes = [cfg['crop_size']] if cfg['dataset'] == 'cityscapes' else [cfg['crop_size'], cfg['crop_size']+int(cfg['crop_size']*0.25 / cfg['patch_size'] + 0.5) * cfg['patch_size']]
                    for size in sizes: 
                        img_u_w_resize = F.interpolate(img_u_w[:cfg['batch_size']//2], (size, size), mode='bilinear', align_corners=True)
                        pred_u_w_resize = model_ema(img_u_w_resize).detach()
                        pred_u_w_resize = F.interpolate(pred_u_w_resize, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=True)
                        preds_u_w.append(pred_u_w_resize)
                        
                        if hflip:
                            pred_u_w_resize_hflip = model_ema(img_u_w_resize.flip(3)).detach()
                            pred_u_w_resize = F.interpolate(pred_u_w_resize_hflip.flip(3), (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=True)
                            preds_u_w.append(pred_u_w_resize)
                    
                    # pred_u_w = torch.mean(torch.stack(preds_u_w, dim=0), dim=0)    
                    pred_u_w_v1 = torch.mean(torch.stack(preds_u_w, dim=0), dim=0)   
                    
                    pred_u_w_v2 = model_ema(img_u_w[cfg['batch_size']//2:]).detach()
                    pred_u_w = torch.cat([pred_u_w_v1, pred_u_w_v2], dim=0)
                    
                    # pred_u_w = model_ema(img_u_w).detach()
                    conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                    mask_u_w = pred_u_w.argmax(dim=1)
                    
                
                img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = img_u_s.flip(0)[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]
                
                num_lb, num_ulb = img_x.shape[0], img_u_s.shape[0]
                pred_x, pred_u_s = model(torch.cat((img_x, img_u_s))).split([num_lb, num_ulb])
                
                pred_u_w_cutmixed, mask_u_w_cutmixed, conf_u_w_cutmixed, ignore_mask_cutmixed = pred_u_w.clone(), mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
                
                b, c, h, w = pred_u_w_cutmixed.shape
                # pred_u_w_cutmixed[(cutmix_box == 1).unsqueeze(1).repeat(1,c,1,1)] = pred_u_w.flip(0)[(cutmix_box == 1).unsqueeze(1).repeat(1,c,1,1)]
                pred_u_w_cutmixed[cutmix_box.unsqueeze(1).expand(pred_u_w_cutmixed.shape) == 1] = pred_u_w.flip(0)[cutmix_box.unsqueeze(1).expand(pred_u_w_cutmixed.shape) == 1]
                mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w.flip(0)[cutmix_box == 1]
                conf_u_w_cutmixed[cutmix_box == 1] = conf_u_w.flip(0)[cutmix_box == 1]
                ignore_mask_cutmixed[cutmix_box == 1] = ignore_mask.flip(0)[cutmix_box == 1]
                
                loss_x = criterion_l(pred_x, mask_x)
                
                ot_mask = ((conf_u_w_cutmixed >= cfg['conf_thresh']) & (ignore_mask_cutmixed != 255)).view(-1)

                if ot_mask.sum() > 0:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=False):
                            scores = F.log_softmax(pred_u_w_cutmixed.float().permute(0, 2, 3, 1).contiguous().view(-1, c), dim=1)
                            assignments = distributed_sinkhorn(scores, tau=0.05).detach()[ot_mask] #.view(b, h, w, c).permute(0, 3, 1, 2).contiguous().detach()
                            if torch.isnan(assignments).any():
                                print("assignments contain nan, use pseudo label.")
                                assignments = pred_u_w_cutmixed.float().permute(0, 2, 3, 1).contiguous().view(-1, c).softmax(dim=1).detach()[ot_mask]
                                # assignments = F.one_hot(mask_u_w_cutmixed.view(-1)[ot_mask], num_classes=cfg['nclass'])
                            
                    loss_u_s = torch.sum(-assignments * F.log_softmax(pred_u_s.permute(0, 2, 3, 1).contiguous().view(-1, c)[ot_mask], dim=1), dim=1)
                    loss_u_s = loss_u_s.mean()     
                else:
                    loss_u_s = torch.zeros_like(loss_x)    
                
                loss = (loss_x + loss_u_s) / 2.0
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            scaler.update()
            
                
            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
                # writer.add_scalar('train/ignore_ratio', ignore_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg, 
                                            total_loss_s.avg, total_mask_ratio.avg))
        
        # eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'tta'
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
