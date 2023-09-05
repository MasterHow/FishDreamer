import torch
import torch.nn as nn

from mmseg.ops import resize

import logging 
LOGGER = logging.getLogger(__name__)


class Upsample(nn.Module):
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
                 ngf=64):
        
        super(Upsample, self).__init__()
        self.in_index = in_index
        self.align_corners = align_corners

        model = []

        model = [nn.Conv2d(in_nc, inter_nc, 1, stride=1, bias=False),
                 up_norm_layer(inter_nc),
                 up_activation   ]
        
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model = model + [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        model = model + [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, out_nc, kernel_size=7, padding=0)]

        self.model = nn.Sequential(*model)

    def _transform_inputs(self, inputs):
            """Transform inputs for decoder.

            Args:
                inputs (list[Tensor]): List of multi-level img features.

            Returns:
                Tensor: The transformed inputs
            """
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[1].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]

            inputs = torch.cat(upsampled_inputs, dim=1)
            return inputs

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        return self.model(inputs)
