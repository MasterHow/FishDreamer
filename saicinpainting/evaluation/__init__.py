import logging

import torch

from saicinpainting.evaluation.evaluator import InpaintingEvaluatorOnline, ssim_fid100_f1, lpips_fid100_f1, ssim_psnr100_f1,ssim_psnr100_miou_f1
from saicinpainting.evaluation.losses.base_loss import SSIMScore, LPIPSScore, FIDScore,PSNRScore,mIoUScore


def make_evaluator(device="cpu", kind='default', ssim=True, lpips=False, fid=True, psnr=True,miou=True,integral_kind=None, **kwargs):
    """
    Revised by Hao:
        --device (str): the device of FID computation model, for balancing GPU memory.
    """
    logging.info(f'Make evaluator {kind}')
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = device
    metrics = {}
    if ssim:
        metrics['ssim'] = SSIMScore()
    if lpips:
        metrics['lpips'] = LPIPSScore()
    if fid:
        metrics['fid'] = FIDScore().to(device)
    if psnr:
        metrics['psnr'] = PSNRScore()
    if miou:
        metrics['miou'] = mIoUScore(nClasses=20)
    if integral_kind is None:
        integral_func = None
    elif integral_kind == 'ssim_fid100_f1':
        integral_func = ssim_fid100_f1
    elif integral_kind == 'lpips_fid100_f1':
        integral_func = lpips_fid100_f1
    elif integral_kind == 'ssim_psnr100_f1':
        integral_func = ssim_psnr100_f1
    elif integral_kind ==  'ssim_psnr100_miou_f1':
        integral_func = ssim_psnr100_miou_f1
    else:
        raise ValueError(f'Unexpected integral_kind={integral_kind}')

    if kind == 'default':
        return InpaintingEvaluatorOnline(scores=metrics,
                                         integral_func=integral_func,
                                         integral_title=integral_kind,
                                         **kwargs)
