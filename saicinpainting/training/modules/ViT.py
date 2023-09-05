import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import cv2

import logging

LOGGER = logging.getLogger(__name__)

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x,y, **kwargs):
        return self.fn(self.norm(x),self.norm(y), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_kv = nn.Linear(dim,inner_dim*2, bias = False)
        self.to_q = nn.Linear(dim,inner_dim,bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):    # x is out feature y is seg feature

        kv = self.to_kv(y).chunk(2, dim = -1)
        q = self.to_q(x)    #.chunck(1,dim=-1)
    
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)

        # q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), q)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Cross_Atten(nn.Module):
    """
    Revised by Hao:
    polar_mask (bool): If true, polar masking input feats before cross att. default: False
    mask_num (int): polar mask numbers. choices: [2, 4, 8], default: 4
    embedding_scale_factor (int): scale factor of embedding layer, correlate to head hidden dims.
    """
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=768,
                 dim_head=64, dropout=0., emb_dropout=0., polar_mask=False, mask_num=4, embedding_scale_factor=1):
        super().__init__()

        self.polar_mask = polar_mask
        self.mask_num = mask_num
        self.dim_scale_factor = self.mask_num / 4      # default: 1
        self.embedding_scale_factor = embedding_scale_factor

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = int(channels * patch_height * patch_width * self.dim_scale_factor * embedding_scale_factor)

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.attention = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.linear = nn.Linear(dim, patch_dim)

        # se
        self.to_original = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                     p1=patch_height, p2=patch_width,
                                     h=image_height//patch_height, w=image_width//patch_width)

    def mask(self, input):
        mask = np.zeros((128, 128))

        if self.mask_num == 4:
            # default, R1=32, R2=48, R3=64 + RX(Margin)
            mask1 = (255 - cv2.circle(mask, (64, 64), 64, 255, -1))

            mask = np.zeros((128, 128))
            mask2 = cv2.circle(mask, (64, 64), 48, 255, -1)
            circular1 = (255 - (mask1 + mask2))

            mask = np.zeros((128,128))
            mask3 = cv2.circle(mask, (64, 64), 32, 255, -1)
            circular2 = (255 - (255-mask2+mask3))

            out1 = input*(torch.from_numpy(mask1[None, ...]/255).cuda())
            out2 = input*(torch.from_numpy(circular1[None, ...]/255).cuda())
            out3 = input*(torch.from_numpy(circular2[None, ...]/255).cuda())
            out4 = input*(torch.from_numpy(mask3[None, ...]/255).cuda())

            out = torch.hstack((out1, out2, out3, out4))
        elif self.mask_num == 2:
            # R1=32 + RX
            mask1 = (255 - cv2.circle(mask, (64, 64), 32, 255, -1))
            mask = np.zeros((128, 128))
            mask2 = cv2.circle(mask, (64, 64), 32, 255, -1)

            out1 = input * (torch.from_numpy(mask1[None, ...] / 255).cuda())
            out2 = input * (torch.from_numpy(mask2[None, ...] / 255).cuda())

            out = torch.hstack((out1, out2))
        elif self.mask_num == 8:
            # R:[16, 24, 32, 40, 48, 56, 64] + RX
            mask1 = (255 - cv2.circle(mask, (64, 64), 64, 255, -1))     # RX

            mask = np.zeros((128, 128))
            mask2 = cv2.circle(mask, (64, 64), 56, 255, -1)
            circular1 = (255 - (mask1 + mask2))     # 56

            mask = np.zeros((128, 128))
            mask3 = cv2.circle(mask, (64, 64), 48, 255, -1)
            circular2 = (255 - mask3 - (mask1 + circular1))     # 48

            mask = np.zeros((128, 128))
            mask4 = cv2.circle(mask, (64, 64), 40, 255, -1)
            circular3 = (255 - mask4 - (mask1 + circular1 + circular2))     # 40

            mask = np.zeros((128, 128))
            mask5 = cv2.circle(mask, (64, 64), 32, 255, -1)
            circular4 = (255 - mask5 - (mask1 + circular1 + circular2 + circular3))     # 32

            mask = np.zeros((128, 128))
            mask6 = cv2.circle(mask, (64, 64), 24, 255, -1)
            circular5 = (255 - mask6 - (mask1 + circular1 + circular2 + circular3 + circular4))     # 24

            mask = np.zeros((128, 128))
            mask7 = cv2.circle(mask, (64, 64), 16, 255, -1)
            circular6 = (255 - mask7 - (mask1 + circular1 + circular2 + circular3 + circular4 + circular5))     # 16

            out1 = input * (torch.from_numpy(mask1[None, ...] / 255).cuda())
            out2 = input * (torch.from_numpy(circular1[None, ...] / 255).cuda())
            out3 = input * (torch.from_numpy(circular2[None, ...] / 255).cuda())
            out4 = input * (torch.from_numpy(circular3[None, ...] / 255).cuda())
            out5 = input * (torch.from_numpy(circular4[None, ...] / 255).cuda())
            out6 = input * (torch.from_numpy(circular5[None, ...] / 255).cuda())
            out7 = input * (torch.from_numpy(circular6[None, ...] / 255).cuda())
            out8 = input * (torch.from_numpy(mask7[None, ...] / 255).cuda())

            out = torch.hstack((out1, out2, out3, out4, out5, out6, out7, out8))

        return out

    def forward(self, img, img_seg):

        if self.polar_mask:
            # with mask
            x = self.mask(img).float()
            y = self.mask(img_seg).float()
        else:
            # without mask
            x = img
            y = img_seg

        x = self.to_patch_embedding(x)
        y = self.to_patch_embedding(y)
        b, n, _ = x.shape

        pos = self.pos_embedding[:, :n]

        x += pos
        y += pos

        x = self.dropout(x)
        y = self.dropout(y)

        out = self.attention(x, y)

        out = self.linear(out)

        out = self.to_original(out)

        return out
