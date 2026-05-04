import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ['dumbbell', 'barbell', 'kettlebell', 'resistance_band', 'pull_up_bar']
NUM_CLASSES = len(CLASS_NAMES)
STRIDES     = [8, 16, 32]   # P3 / P4 / P5


class ConvBNSiLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=None):
        super().__init__()
        p = k // 2 if p is None else p
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MiniBlock(nn.Module):
    """Depthwise-separable Conv block with residual shortcut."""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw    = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_c)
        self.pw    = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_c)
        self.act   = nn.SiLU(inplace=True)
        self.short = (
            nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))
            if stride != 1 or in_c != out_c else nn.Identity()
        )

    def forward(self, x):
        out = self.act(self.bn_dw(self.dw(x)))
        out = self.act(self.bn_pw(self.pw(out)))
        return out + self.short(x)


class GymDetectorMini(nn.Module):
    """
    Small one-stage detector trained from scratch on gym equipment.
    Input : (B, 3, 320, 320)
    Output: list of 3 raw tensors [P3=(B,30,40,40), P4=(B,30,20,20), P5=(B,30,10,10)]
            Strides: [8, 16, 32]. No sigmoid/exp inside — handled by loss/decode.
    """
    def __init__(self, nc=NUM_CLASSES, na=3):
        super().__init__()
        self.nc = nc
        self.na = na
        no = na * (5 + nc)   # 3 × 10 = 30

        # ── Backbone ─────────────────────────────────────────────────────────
        # 320 → 160
        self.stem   = ConvBNSiLU(3, 32, 3, 2)
        # 160 → 80
        self.stage1 = nn.Sequential(MiniBlock(32,  64,  stride=2), MiniBlock(64,  64))
        # 80 → 40  P3 = 128 channels
        self.stage2 = nn.Sequential(MiniBlock(64,  128, stride=2), MiniBlock(128, 128))
        # 40 → 20  P4 = 192 channels
        self.stage3 = nn.Sequential(MiniBlock(128, 192, stride=2), MiniBlock(192, 192))
        # 20 → 10  P5 = 256 channels
        self.stage4 = nn.Sequential(MiniBlock(192, 256, stride=2), MiniBlock(256, 256))

        # ── FPN Neck (top-down only) ──────────────────────────────────────────
        self.p5_lat  = ConvBNSiLU(256, 128, 1)            # reduce P5: 256→128
        self.p4_fuse = ConvBNSiLU(128 + 192, 128, 3)      # concat(P5_up, P4): 320→128
        self.p4_lat  = ConvBNSiLU(128, 64, 1)             # reduce P4_fused: 128→64
        self.p3_fuse = ConvBNSiLU(64 + 128, 64, 3)        # concat(P4_up, P3): 192→64

        # ── Detection Heads ───────────────────────────────────────────────────
        self.head_p5 = nn.Sequential(ConvBNSiLU(128, 128, 3), nn.Conv2d(128, no, 1))
        self.head_p4 = nn.Sequential(ConvBNSiLU(128, 128, 3), nn.Conv2d(128, no, 1))
        self.head_p3 = nn.Sequential(ConvBNSiLU(64,   64, 3), nn.Conv2d(64,  no, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x  = self.stem(x)      # (B, 32, 160, 160)
        x  = self.stage1(x)    # (B, 64,  80,  80)
        p3 = self.stage2(x)    # (B, 128, 40,  40)
        p4 = self.stage3(p3)   # (B, 192, 20,  20)
        p5 = self.stage4(p4)   # (B, 256, 10,  10)

        p5r = self.p5_lat(p5)                                                                      # (B, 128, 10, 10)
        p4f = self.p4_fuse(torch.cat([F.interpolate(p5r, scale_factor=2, mode='nearest'), p4], 1)) # (B, 128, 20, 20)
        p4r = self.p4_lat(p4f)                                                                     # (B,  64, 20, 20)
        p3f = self.p3_fuse(torch.cat([F.interpolate(p4r, scale_factor=2, mode='nearest'), p3], 1)) # (B,  64, 40, 40)

        # Return in stride order: small stride first (P3=stride-8, highest resolution)
        return [self.head_p3(p3f), self.head_p4(p4f), self.head_p5(p5r)]


if __name__ == '__main__':
    model = GymDetectorMini()
    n = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {n:,}  ({n / 37_200_000 * 100:.2f}% of YOLOv7 base)')
    assert n < 3_720_000, f'Over 10% budget: {n:,}'
    x = torch.zeros(1, 3, 320, 320)
    outs = model(x)
    for o, s in zip(outs, STRIDES):
        print(f'  stride {s:2d}  →  {tuple(o.shape)}')
