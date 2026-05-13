import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.fpn import SimpleFPN, MultiScaleFPN, ViTFPN

from model.backbone.dinov2 import DINOv2
from model.backbone.dinov3 import DINOv3

from model.layers.ms_deform_pixel_decoder import MSDeformAttnPixelDecoder
from model.layers.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder


class Mask2FormerHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024],
        num_queries=100,
        align_corners=False,
    ):
        super(Mask2FormerHead, self).__init__()

        self.neck = MultiScaleFPN(in_channels, out_channels, features)

        self.pixel_decoder = MSDeformAttnPixelDecoder(
            in_channels=[features] * 4, 
            in_strides=[4, 8, 16, 32], 
            transformer_dropout=0.0,
            transformer_nheads=8,  # 16
            transformer_dim_feedforward=1024,  # 4096
            transformer_enc_layers=6,
            conv_dim=features,  # 2048
            mask_dim=features,  # 2048
            transformer_in_features=[1, 2, 3], # [1, 2, 3, 4],
            common_stride=4,
            norm="groupnorm",
            num_groups=32,
        )
        self.transformer_decoder = MultiScaleMaskedTransformerDecoder(
            in_channels = features,   # 2048
            num_classes=nclass, 
            mask_classification=True,
            hidden_dim=features,
            num_queries=num_queries,
            nheads=8,  # 16
            dim_feedforward=2048, # 4096
            mask_dim=features,  # 2048
            dec_layers=9, 
            pre_norm=False,
            enforce_input_project=False,
        )
        self.align_corners=align_corners
        
    def forward(self, out_features, patch_h, patch_w):       
        out = self.neck([x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w) for x in out_features])
        
        mask_feature, multi_scale_memories = self.pixel_decoder(out, align_corners=self.align_corners)
        
        outputs = self.transformer_decoder(multi_scale_memories, mask_feature, align_corners=self.align_corners)

        return outputs



class Mask2Former(nn.Module):
    def __init__(
        self, 
        encoder_size='base', 
        nclass=21,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        num_queries=100, 
        use_bn=False, 
        align_corners=False
    ):
        super(Mask2Former, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            'large': [4, 11, 17, 23], 
            'giant': [9, 19, 29, 39]
        }
        
        self.encoder_size = encoder_size.split('_')[-1]
        # self.backbone = DINOv3(model_name=encoder_size)
        self.backbone = DINOv2(model_name=self.encoder_size) if "dinov2" in encoder_size else DINOv3(model_name=self.encoder_size)
        self.patch_size = self.backbone.patch_size


        self.head = Mask2FormerHead(nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels, num_queries=num_queries, align_corners=align_corners)

        self.align_corners = align_corners
        


    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
            
            
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        
        features = self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size]
        )
        
        # print([feature.shape for feature in features])
        out = self.head(features, patch_h, patch_w)
        
        return out