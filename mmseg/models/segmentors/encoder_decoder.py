from mmseg.ops import resize

from .base import BaseSegmentor
from saicinpainting.training.modules.swin_transformer import SwinTransformer
from saicinpainting.training.modules.decode_heads.uper_head import UPerHead


class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 **kwargs):
        super(EncoderDecoder, self).__init__()

        self.backbone = SwinTransformer(**kwargs)
        self.decode_head= UPerHead()
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

        self.init_weights(pretrained=pretrained)

        assert self.with_decode_head

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        return x

    def encode_decode(self, img):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x)
        
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_test(self, x):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x)
        return seg_logits

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit
