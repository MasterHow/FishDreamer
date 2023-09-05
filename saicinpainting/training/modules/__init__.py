import logging

from saicinpainting.training.modules.ffc import FFCResNetGenerator
from saicinpainting.training.modules.pix2pixhd import GlobalGenerator, MultiDilatedGlobalGenerator, \
    NLayerDiscriminator, MultidilatedNLayerDiscriminator
from saicinpainting.training.modules.swin_transformer import SwinTransformer
from saicinpainting.training.modules.uper_head import UPerHead
from saicinpainting.training.modules.swin_uper_encoder_decoder import SwinuperEncoderDecoder

LOGGER = logging.getLogger(__name__)


def make_encoder_decoder(config,kind,**kwargs):
    if kind == 'swin_upernet':      # encoder_decoder config kind
        return SwinuperEncoderDecoder(config, **kwargs)


def make_discriminator(kind, **kwargs):
    logging.info(f'Make discriminator {kind}')

    if kind == 'pix2pixhd_nlayer_multidilated':
        return MultidilatedNLayerDiscriminator(**kwargs)

    if kind == 'pix2pixhd_nlayer':
        return NLayerDiscriminator(**kwargs)

    raise ValueError(f'Unknown discriminator kind {kind}')
