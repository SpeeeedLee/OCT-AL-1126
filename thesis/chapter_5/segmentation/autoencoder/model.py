"""
U-Net-encoder AUTOENCODER for self-supervised pretraining (Fu et al., TMI 2024).

Pretext = reconstruct the input image (MSE). Following the paper (Sec. IV-C),
the autoencoder has NO skip connections — otherwise the U-Net would "collapse"
(skip connections copy the encoder feature straight to the decoder, so the
encoder learns nothing). Only the ENCODER is transferred downstream.

  encoder  = Conv1..Conv6 + Maxpool, byte-for-byte the same modules/names as
             `Optim_U_Net`  → loads back via load_state_dict(strict=False).
  decoder  = symmetric up-path, but each up-block takes ONLY the upsampled
             feature (no concat with an encoder skip) → distinct module names
             (`Dec*`), so it never collides with the U-Net's own decoder keys.

Output = sigmoid → [0,1], matching the downstream `o_data` images (cv2/255).
"""
import torch
import torch.nn as nn

from thesis.chapter_5.segmentation.utils.model import conv_block, up_conv


class UNetAutoencoder(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, filter_size=32):
        super().__init__()
        f = filter_size
        # --- encoder: identical module names to Optim_U_Net (Conv1..Conv6) ---
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=f)
        self.Conv2 = conv_block(ch_in=f, ch_out=f * 2)
        self.Conv3 = conv_block(ch_in=f * 2, ch_out=f * 4)
        self.Conv4 = conv_block(ch_in=f * 4, ch_out=f * 8)
        self.Conv5 = conv_block(ch_in=f * 8, ch_out=f * 16)
        self.Conv6 = conv_block(ch_in=f * 16, ch_out=f * 32)

        # --- skip-free symmetric decoder (NOT transferred; own names) ---
        # each DecUp halves channels + upsamples x2; DecConv refines (NO skip cat)
        self.DecUp6 = up_conv(ch_in=f * 32, ch_out=f * 16)
        self.DecConv6 = conv_block(ch_in=f * 16, ch_out=f * 16)
        self.DecUp5 = up_conv(ch_in=f * 16, ch_out=f * 8)
        self.DecConv5 = conv_block(ch_in=f * 8, ch_out=f * 8)
        self.DecUp4 = up_conv(ch_in=f * 8, ch_out=f * 4)
        self.DecConv4 = conv_block(ch_in=f * 4, ch_out=f * 4)
        self.DecUp3 = up_conv(ch_in=f * 4, ch_out=f * 2)
        self.DecConv3 = conv_block(ch_in=f * 2, ch_out=f * 2)
        self.DecUp2 = up_conv(ch_in=f * 2, ch_out=f)
        self.DecConv2 = conv_block(ch_in=f, ch_out=f)
        self.Dec_1x1 = nn.Conv2d(f, output_ch, kernel_size=1, stride=1, padding=0)

    def encode(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))
        x6 = self.Conv6(self.Maxpool(x5))   # bottleneck
        return x6

    def forward(self, x):
        z = self.encode(x)
        d = self.DecConv6(self.DecUp6(z))    # NO torch.cat — skip-free
        d = self.DecConv5(self.DecUp5(d))
        d = self.DecConv4(self.DecUp4(d))
        d = self.DecConv3(self.DecUp3(d))
        d = self.DecConv2(self.DecUp2(d))
        return torch.sigmoid(self.Dec_1x1(d))   # [B,1,384,512] in [0,1]

    def encoder_state_dict(self):
        """Conv1..Conv6 params only (exactly what Optim_U_Net consumes)."""
        return {k: v for k, v in self.state_dict().items()
                if k.split(".")[0] in
                {"Conv1", "Conv2", "Conv3", "Conv4", "Conv5", "Conv6"}}
