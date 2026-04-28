import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ResNet34UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=2,
            classes=1,
            activation=None  # no sigmoid — raw logits for BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.model(x)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UNetScratch(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=2,
            classes=1,
            activation=None
        )

    def forward(self, x):
        return self.model(x)

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _ConvBlock(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.up(x))


class VanillaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = _ConvBlock(2, 32);   self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _ConvBlock(32, 64);  self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _ConvBlock(64, 128); self.pool3 = nn.MaxPool2d(2)
        self.enc4 = _ConvBlock(128,256); self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(256, 512)
        self.dec4 = _UpBlock(512, 256)
        self.dec3 = _UpBlock(256, 128)
        self.dec2 = _UpBlock(128, 64)
        self.dec1 = _UpBlock(64,  32)
        # raw logits — no sigmoid
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        return self.head(self.dec1(self.dec2(self.dec3(self.dec4(b)))))

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(model_name, device):
    registry = {
        "resnet34_unet": ResNet34UNet,
        "unet_scratch":  UNetScratch,
        "vanilla_cnn":   VanillaCNN
    }
    if model_name not in registry:
        raise ValueError(f"Unknown model '{model_name}'.")
    model = registry[model_name]().to(device)
    print(f"[model] {model_name} — {model.get_param_count():,} trainable parameters")
    return model
