"""
U-Net ENCODER SimCLR model for self-supervised pretraining.

= the downsampling path of `Optim_U_Net` (Conv1..Conv6 + Maxpool, reusing the
verbatim-copied `conv_block`) + global-average-pool + a 2-layer MLP projection
head. The encoder module names are IDENTICAL to `Optim_U_Net`, so the pretrained
encoder weights load straight back into the segmentation U-Net via
`Optim_U_Net.load_state_dict(state, strict=False)` (decoder stays random;
`proj.*` keys are simply ignored as "unexpected").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from thesis.chapter_5.segmentation.utils.model import conv_block


class UNetEncoderSimCLR(nn.Module):
    def __init__(self, img_ch=1, filter_size=32, out_dim=32):
        super().__init__()
        # --- encoder: byte-for-byte the same modules/names as Optim_U_Net ---
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=filter_size)
        self.Conv2 = conv_block(ch_in=filter_size, ch_out=filter_size * 2)
        self.Conv3 = conv_block(ch_in=filter_size * 2, ch_out=filter_size * 4)
        self.Conv4 = conv_block(ch_in=filter_size * 4, ch_out=filter_size * 8)
        self.Conv5 = conv_block(ch_in=filter_size * 8, ch_out=filter_size * 16)
        self.Conv6 = conv_block(ch_in=filter_size * 16, ch_out=filter_size * 32)
        # --- SimCLR projection head (NOT loaded into the U-Net) ---
        dim = filter_size * 32  # 1024
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, out_dim))

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))
        x6 = self.Conv6(self.Maxpool(x5))          # bottleneck [B, 1024, h, w]
        g = F.adaptive_avg_pool2d(x6, (1, 1)).flatten(1)  # [B, 1024]
        return self.proj(g)                         # [B, out_dim]

    def encoder_state_dict(self):
        """Just the Conv1..Conv6 params (what Optim_U_Net consumes)."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("proj.")}
