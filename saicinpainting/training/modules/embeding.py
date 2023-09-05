import numpy as np 
import torch
import cv2

import logging

LOGGER = logging.getLogger(__name__)


def Embeding(input):
    """
    Polar-aware embedding of input feature.
    :param input: (tensor) feature map.
    :return: out: (tensor) multi-level polar feature.
    """

    mask = np.zeros((128, 128))

    mask1= (255 - cv2.circle(mask, (64, 64), 64, 255, -1))

    mask = np.zeros((128, 128))
    mask2 = cv2.circle(mask, (64, 64), 48, 255, -1)
    circular1 = (255 - (mask1 + mask2))

    mask = np.zeros((128, 128))
    mask3 = cv2.circle(mask, (64, 64), 32, 255, -1)
    circular2 = (255 - (255-mask2+mask3))

    out1 = input*(torch.from_numpy(mask1[None, ...]/255).cuda())
    out2 = input*(torch.from_numpy(circular1[None, ...]/255).cuda())
    out3 = input*(torch.from_numpy(circular2[None, ...]/255).cuda())
    out4 = input*(torch.from_numpy(mask3[None, ...]/255).cuda())

    out = torch.hstack((out1, out2, out3, out4))

    return out

    
    
