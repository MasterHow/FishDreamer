import torch
import torch.nn as nn

from mmseg.ops import resize
from saicinpainting.training.modules.embeding import Embeding
from saicinpainting.training.modules.CBAM import CABlock, SABlock, CBAMBlock

import logging 
LOGGER = logging.getLogger(__name__)


class UpsamplePolar(nn.Module):
    """
    单向引入语义特征，并进行极性mask划分和attention融合
    att_inC (int): in-channels of cross attention module.
    skip_polar (bool): if true, skip polar mask embedding, default: False
    """
    def __init__(self,
                 in_index=[0, 1, 2, 3],
                 align_corners=False,
                 n_downsampling=3,
                 in_nc=1440,       # tiny small
                 # in_nc=1920,         # base
                 # in_nc=2880,         # large
                 inter_nc=512,
                 out_nc=3,
                 max_features=1024,
                 up_norm_layer=nn.BatchNorm2d,
                 up_activation=nn.ReLU(True),
                 ngf=64,
                 att_inC=1792,
                 skip_polar=False):
        
        super(UpsamplePolar, self).__init__()
        self.in_index = in_index
        self.align_corners = align_corners
        self.skip_polar = skip_polar

        # channel attention
        self.cross_att = CABlock(channel=att_inC, reduction=16)

        self.conv1 = nn.Sequential(nn.Conv2d(in_nc, inter_nc, 1, stride=1, bias=False),
                                   up_norm_layer(inter_nc),
                                   up_activation )

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

        self.conv6 = nn.Conv2d(att_inC, 256, kernel_size=1, padding=0)

    def _transform_inputs(self, inputs):
            """Transform inputs for decoder.

            Args:
                inputs (list[Tensor]): List of multi-level img features.

            Returns:
                Tensor: The transformed inputs
            """

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

    def forward(self, inputs, seg_f):
        inputs = self._transform_inputs(inputs)
        inputs = self.conv1(inputs)
        out_feature = self.conv2(inputs)

        if not self.skip_polar:
            feature_masked = Embeding(out_feature).float()
            seg_f_masked = Embeding(seg_f).float()
            attn_input = torch.hstack((feature_masked, seg_f_masked))
        else:
            attn_input = torch.hstack((out_feature, seg_f))

        attn = self.cross_att(attn_input)

        feature = out_feature + self.conv6(attn)
        out = self.conv3(feature)
        out = self.conv4(out)
        out = self.conv5(out)

        return out
