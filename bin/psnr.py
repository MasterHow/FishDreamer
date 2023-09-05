#!/usr/bin/env python
# coding: utf-8


import os 
import cv2
import glob
import numpy as np
import argparse


def caculate_psnr(p0, p1, peak=255.):
    return 10 * np.log10(peak ** 2 / np.mean((1. * p0 - 1. * p1) ** 2))


def main(args):
    gt_path = args.gtdir

    inpainted_path = args.predictdir


    mask_filenames = sorted(list(glob.glob(os.path.join(gt_path, '*mask.png'), recursive=True)))
    gt_files = [os.path.join(gt_path, os.path.basename(fname.rsplit('-', 1)[0].rsplit('_', 1)[0]) + '.png') for fname in mask_filenames]
    inpainted_files = sorted(list(glob.glob(os.path.join(inpainted_path,"**.png"))))
    psnr = 0
    # print(len(gt_files))
    for i in range(0,len(gt_files)) :
        gt = cv2.imread(gt_files[i])
        inpainted = cv2.imread(inpainted_files[i])
        psnr = psnr + caculate_psnr(gt,inpainted,255.)
    mean_psnr = psnr/len(gt_files)
    print("psnr",mean_psnr)
    
    
if __name__ == '__main__':


    aparser = argparse.ArgumentParser()
   
    aparser.add_argument('gtdir', type=str,
                         help='Path to folder with images and masks (output of gen_mask_dataset.py)')
    aparser.add_argument('predictdir', type=str,
                         help='Path to folder with predicts (e.g. predict_hifill_baseline.py)')
   
    main(aparser.parse_args())
