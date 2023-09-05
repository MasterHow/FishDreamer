import torch.nn as nn
import torch

from mmseg.ops import resize

from mmcv.runner import auto_fp16, load_checkpoint
from saicinpainting.training.modules.swin_transformer import SwinTransformer
from saicinpainting.training.modules.mit import mit_b0, mit_b2
from saicinpainting.training.modules.conformer import Conformer
from saicinpainting.training.modules.uper_head import UPerHead
from saicinpainting.training.modules.upsampler import Upsample
from saicinpainting.training.modules.upsampler_polar import UpsamplePolar
from saicinpainting.training.modules.bidirec import Bidirection

import logging

LOGGER = logging.getLogger(__name__)


def make_encoder(kind=None, **kwargs):
    if kind is None or kind == 'default':
        return SwinTransformer(**kwargs)
    elif kind == 'mit_b0':
        return mit_b0(**kwargs)
    elif kind == 'mit_b2':
        return mit_b2(**kwargs)
    elif kind == 'conformer_tiny':
        return Conformer(**kwargs)
    elif kind == 'conformer_small':
        return Conformer(**kwargs)
    else:
        print('Unexpected Backbone, Return Default Swin-Trans...')
        return SwinTransformer(**kwargs)


def make_outpainting_decoder(kind=None, **kwargs):
    if kind is None or kind == 'default':
        return Upsample(**kwargs)
    elif kind == 'polar':
        return UpsamplePolar(**kwargs)
    else:
        print('Unexpected Outpainting Head, Return Default Head...')
        return Upsample(**kwargs)


def make_segmentation_decoder(**kwargs):
    return UPerHead(**kwargs)


def make_bidirectional_decoder(**kwargs):
    return Bidirection(**kwargs)


class SwinuperEncoderDecoder(nn.Module):
    """Swin+Upernet Encoder-Decoder used in Global&Local model.
    This implementation follows:
    Globally and locally Consistent Image Completion
    The architecture of the encoder-decoder is:\
        (conv2d x 6) --> (dilated conv2d x 4) --> (conv2d or deconv2d x 7)
    Args:
        encoder (dict): Config dict to encoder.
        decoder (dict): Config dict to build decoder.
        dilation_neck (dict): Config dict to build dilation neck.

    Revised by Hao:
    config (dict): config args from encoder_decoder.yaml.
    out_intermediate (bool): if True, additionally output the feature after context block, default: False.
    outpainting_head_type (str): kind of outpainting head, choice: ['polar', 'default']
    bi_direction_head (bool): if true, replace separate two heads with a bi-directional head (in qkv fashion).
    pixel_shuffle (bool): if true, use pixel shuffle upconv for outpainting head.
    """

    def __init__(self,
                 config=None,
                 pretrained=None,
                 **kwargs):     # **kwargs of encoder_decoder config
        super().__init__()

        self.config = config
        self.out_intermediate = config['encoder_decoder']['segmentation_head'].get('out_intermediate', False)
        self.outpainting_head_type = config['encoder_decoder']['outpainting_head'].get('kind', 'default')
        self.bi_direction_head = config['encoder_decoder'].get('bi_direction_head', False)
        if self.bi_direction_head:
            self.pixel_shuffle = config['encoder_decoder']['bi_head_settings'].get('pixel_shuffle', False)
        else:
            self.pixel_shuffle = config['encoder_decoder']['outpainting_head'].get('pixel_shuffle', False)
            if self.pixel_shuffle:
                raise NotImplementedError()

        self.backbone = make_encoder(**config['encoder_decoder']['encoder'])  # backbone

        self._init_decode_head(config)

        self.init_weights(pretrained=pretrained)

    def _init_decode_head(self, config):
        if self.bi_direction_head:
            # bidirectional head
            self.bi_decoder = make_bidirectional_decoder(**config['encoder_decoder']['bi_head_settings'])
        else:
            # separate head
            self.outpainting_decode_head = make_outpainting_decoder(**config['encoder_decoder']['outpainting_head'])
            self.segmentation_decode_head = make_segmentation_decoder(**config['encoder_decoder']['segmentation_head'])

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.backbone.init_weights(pretrained=pretrained)

        if self.bi_direction_head:
            self.bi_decoder.init_weights()
        else:
            self.segmentation_decode_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        return x

    @auto_fp16()
    def forward(self, img):
        """Forward Function.
        Args:
            img (torch.Tensor): Input tensor with shape of (n, c+1, h, w).
                                Last channel is mask.
        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        x = self.extract_feat(img[:, :-1])

        if self.bi_direction_head:
            # bidirectional head
            out_outpainting, out_segmentation = self.bi_decoder(x)
        else:
            # separate head
            if self.out_intermediate:
                out_segmentation, seg_feature = self.segmentation_decode_head(x)
            else:
                out_segmentation = self.segmentation_decode_head(x)

            if self.outpainting_head_type == 'polar':
                out_outpainting = self.outpainting_decode_head(x, seg_feature)
            else:
                out_outpainting = self.outpainting_decode_head(x)

        out_outpainting = resize(
            input=out_outpainting,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=False)

        out_segmentation = resize(
            input=out_segmentation,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=False)

        return out_outpainting, out_segmentation
