import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM

import logging 
from saicinpainting.training.modules.ViT import Cross_Atten

LOGGER = logging.getLogger(__name__)


class Bidirection(BaseDecodeHead):
    """
    Revised by Hao:
    polar_mask (bool): If true, polar masking input feats before cross att. default: False
    mask_num (int): polar mask numbers. choices: [2, 4, 8], default: 4
    fuse_direction (str): fuse direction, choices: ['S2P', 'P2S', 'Bi']. default: 'Bi'. only support wo mask now.
    pixel_shuffle (bool): if true, use pixel shuffle upconv for outpainting head.
    """
    def __init__(self, pool_scales=(1, 2, 3, 6),   
                align_corners=False,
                n_downsampling=3,
                in_nc=1440,
                inter_nc=512,
                out_nc=3,
                max_features=1024,
                up_norm_layer=nn.BatchNorm2d, 
                up_activation=nn.ReLU(True),
                ngf=64,
                polar_mask=False,
                mask_num=4,
                fuse_direction='Bi',
                pixel_shuffle=False,
                **kwargs):
        super(Bidirection, self).__init__(
            input_transform='multiple_select', **kwargs)

        self.polar_mask = polar_mask
        self.mask_num = mask_num
        self.dim_scale_factor = self.mask_num / 4  # default: 1
        self.fuse_direction = fuse_direction
        embedding_scale_factor = inter_nc // 512    # scale factor that related to hidden dim in out-painting head
        self.pixel_shuffle = pixel_shuffle

        # semantics head
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in self.in_channels[:-1]:  # skip the top layer

            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        if self.polar_mask:
            # with polar
            self.atten1 = Cross_Atten(image_size=128, patch_size=8, num_classes=2, dim=512, depth=0, heads=8, mlp_dim=0,
                                      pool='cls', channels=768, dim_head=64, dropout=0., emb_dropout=0.,
                                      polar_mask=True, mask_num=mask_num, embedding_scale_factor=embedding_scale_factor)
            self.atten2 = Cross_Atten(image_size=128, patch_size=8, num_classes=2, dim=512, depth=0, heads=8, mlp_dim=0,
                                      pool='cls', channels=768, dim_head=64, dropout=0., emb_dropout=0.,
                                      polar_mask=True, mask_num=mask_num, embedding_scale_factor=embedding_scale_factor)
            if self.fuse_direction is not 'Bi':
                raise NotImplementedError()
        else:
            # without polar
            if self.fuse_direction == 'Bi':
                self.atten1 = Cross_Atten(image_size=128, patch_size=8, num_classes=2, dim=512, depth=0, heads=8, mlp_dim=0,
                                          pool='cls', channels=192, dim_head=64, dropout=0., emb_dropout=0.,
                                          polar_mask=False, embedding_scale_factor=embedding_scale_factor)
                self.atten2 = Cross_Atten(image_size=128, patch_size=8, num_classes=2, dim=512, depth=0, heads=8, mlp_dim=0,
                                          pool='cls', channels=192, dim_head=64, dropout=0., emb_dropout=0.,
                                          polar_mask=False, embedding_scale_factor=embedding_scale_factor)
            elif self.fuse_direction == 'P2S':
                self.atten1 = Cross_Atten(image_size=128, patch_size=8, num_classes=2, dim=512, depth=0, heads=8,
                                          mlp_dim=0,
                                          pool='cls', channels=192, dim_head=64, dropout=0., emb_dropout=0.,
                                          polar_mask=False, embedding_scale_factor=embedding_scale_factor)
            elif self.fuse_direction == 'S2P':
                self.atten2 = Cross_Atten(image_size=128, patch_size=8, num_classes=2, dim=512, depth=0, heads=8,
                                          mlp_dim=0,
                                          pool='cls', channels=192, dim_head=64, dropout=0., emb_dropout=0.,
                                          polar_mask=False, embedding_scale_factor=embedding_scale_factor)
            else:
                raise NotImplementedError()
       
        # out-painting head
        self.conv1 = nn.Sequential(nn.Conv2d(in_nc, inter_nc, 1, stride=1, bias=False),
                 up_norm_layer(inter_nc),
                 up_activation )

        if self.pixel_shuffle:
            i = 0
            mult = 2 ** (n_downsampling - i)
            self.conv2 = nn.Sequential(nn.Conv2d(min(max_features, ngf * mult),
                                                 4 * min(max_features, int(ngf * mult / 2)),
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       up_norm_layer(4 * min(max_features, int(ngf * mult / 2))),
                                       up_activation,
                                       nn.PixelShuffle(2))
            i = 1
            mult = 2 ** (n_downsampling - i)
            self.conv3 = nn.Sequential(nn.Conv2d(min(max_features, ngf * mult),
                                                 4 * min(max_features, int(ngf * mult / 2)),
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       up_norm_layer(4 * min(max_features, int(ngf * mult / 2))),
                                       up_activation,
                                       nn.PixelShuffle(2))
            i = 2
            mult = 2 ** (n_downsampling - i)
            self.conv4 = nn.Sequential(nn.Conv2d(min(max_features, ngf * mult),
                                                 4 * min(max_features, int(ngf * mult / 2)),
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                       up_norm_layer(4 * min(max_features, int(ngf * mult / 2))),
                                       up_activation,
                                       nn.PixelShuffle(2))
        else:
            # default: deconv
            i = 0
            mult = 2 ** (n_downsampling - i)
            self.conv2 = nn.Sequential(nn.ConvTranspose2d(min(max_features, ngf * mult),
                                                          min(max_features, int(ngf * mult / 2)),
                                                          kernel_size=3, stride=2, padding=1, output_padding=1),
                                       up_norm_layer(min(max_features, int(ngf * mult / 2))),
                                       up_activation)
            i = 1
            mult = 2 ** (n_downsampling - i)
            self.conv3 = nn.Sequential(nn.ConvTranspose2d(min(max_features, ngf * mult),
                                                          min(max_features, int(ngf * mult / 2)),
                                                          kernel_size=3, stride=2, padding=1, output_padding=1),
                                       up_norm_layer(min(max_features, int(ngf * mult / 2))),
                                       up_activation)
            i = 2
            mult = 2 ** (n_downsampling - i)
            self.conv4 = nn.Sequential(nn.ConvTranspose2d(min(max_features, ngf * mult),
                                                          min(max_features, int(ngf * mult / 2)),
                                                          kernel_size=3, stride=2, padding=1, output_padding=1),
                                       up_norm_layer(min(max_features, int(ngf * mult / 2))),
                                       up_activation)

        self.conv5 = nn.Sequential(nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf, out_nc, kernel_size=7, padding=0))

        if self.polar_mask:
            self.conv6 = nn.Conv2d(int(self.dim_scale_factor*embedding_scale_factor*768), 192*embedding_scale_factor,
                                   kernel_size=1, padding=0)
            self.conv7 = nn.Conv2d(int(self.dim_scale_factor*embedding_scale_factor*768), 256*embedding_scale_factor,
                                   kernel_size=1, padding=0)

        if self.fuse_direction == 'Bi':
            self.conv8_1 = nn.Conv2d(256*embedding_scale_factor, 192*embedding_scale_factor, kernel_size=1)
            self.conv8_2 = nn.Conv2d(256*embedding_scale_factor, 192*embedding_scale_factor, kernel_size=1)
        elif self.fuse_direction == 'P2S':
            self.conv8_1 = nn.Conv2d(256*embedding_scale_factor, 192*embedding_scale_factor, kernel_size=1)
        elif self.fuse_direction == 'S2P':
            self.conv8_2 = nn.Conv2d(256*embedding_scale_factor, 192*embedding_scale_factor, kernel_size=1)
        else:
            raise NotImplementedError()

        if not self.polar_mask:
            if self.fuse_direction == 'Bi' or self.fuse_direction == 'S2P':
                self.conv9 = nn.Conv2d(192, 256, kernel_size=1)
    
    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]

        psp_outs.extend(self.psp_modules(x))
    
        psp_outs = torch.cat(psp_outs, dim=1)
       
        output = self.bottleneck(psp_outs)

        return output
    
    def input_transform_for_outpainting(self,inputs):
            inputs = [inputs[i] for i in range(0,len(inputs))]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[1].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]

            inputs = torch.cat(upsampled_inputs, dim=1)
            return inputs

    def forward(self, x):
        x_outpaint = self.input_transform_for_outpainting(x)
        x_seg = self._transform_inputs(x)

        x_outpaint = self.conv1(x_outpaint)
        outpaint_feature = self.conv2(x_outpaint)

        # semantics head #
        laterals = [
            lateral_conv(x_seg[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
  
        laterals.append(self.psp_forward(x_seg))
        
        # build top-down path
        used_backbone_levels = len(laterals)
   
        for i in range(used_backbone_levels - 1, 0, -1):    # i = 3,2,1
        
            prev_shape = laterals[i - 1].shape[2:]
            
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
       
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        seg_feature = fpn_outs[0]

        if self.fuse_direction == 'Bi' or self.fuse_direction == 'P2S':
            atten1 = self.atten1(seg_feature, self.conv8_1(outpaint_feature))

            if self.polar_mask:
                # with polar
                fpn_outs[0] = seg_feature + self.conv6(atten1)
            else:
                # wihtout polar
                fpn_outs[0] = fpn_outs[0] + atten1

        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)

        out_feature = self.fpn_bottleneck(fpn_outs)

        out_seg = self.cls_seg(out_feature)

        # out-painting head #
        if self.fuse_direction == 'Bi' or self.fuse_direction == 'S2P':
            atten2 = self.atten2(self.conv8_2(outpaint_feature), out_feature)

            if self.polar_mask:
                # with polar
                feature = outpaint_feature + self.conv7(atten2)
            else:
                # without polar
                feature = outpaint_feature + self.conv9(atten2)
        else:
            feature = outpaint_feature
        
        out_outpaint = self.conv3(feature)
        out_outpaint = self.conv4(out_outpaint)
        out_outpaint = self.conv5(out_outpaint)

        return out_outpaint, out_seg
