"""
Modular PyTorch implementation of U-Net architectures for biomedical semantic segmentation.

Supports:
1. Modern Pre-trained Encoders via segmentation_models_pytorch (ResNet34, ResNet50, EfficientNet, etc.)
2. Classic Ronneberger U-Net trained from scratch (DoubleConv baseline)
3. Unified factory function `build_segmentation_model`
"""

from typing import List, Optional, Dict, Any, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    smp = None  # type: ignore
    HAS_SMP = False

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Classic Ronneberger U-Net (From Scratch Baseline)
# ==============================================================================

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2 block."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool2D then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling with Bilinear interpolation / ConvTranspose then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Handle potential padding differences for odd dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping features to target class logits."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ClassicUNet(nn.Module):
    """
    Standard U-Net architecture for 2D semantic segmentation (Ronneberger et al.).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Optional[List[int]] = None,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512, 1024]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Initial stem
        self.inc = DoubleConv(in_channels, features[0])

        # Encoder downsampling stages
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[4] // factor)

        # Decoder upsampling stages
        self.up1 = Up(features[4], features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)

        # Output projection
        self.outc = OutConv(features[0], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# Backwards compatibility alias
UNet = ClassicUNet


# ==============================================================================
# 2. Native Torchvision ResNet U-Net (Zero External Dependencies)
# ==============================================================================

class ResNetDecoderBlock(nn.Module):
    """Bilinear upsampling followed by double 3x3 Conv-BN-ReLU."""

    def __init__(self, in_c: int, skip_c: int, out_c: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Handle possible 1-pixel rounding padding difference
            diff_y = skip.size(2) - x.size(2)
            diff_x = skip.size(3) - x.size(3)
            if diff_y != 0 or diff_x != 0:
                x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TorchvisionResNetUNet(nn.Module):
    """
    U-Net with a pre-trained torchvision ResNet backbone.
    Operates with ZERO external dependencies (no segmentation_models_pytorch required).
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        in_channels: int = 3,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        import torchvision.models as tv_models

        self.encoder_name = encoder_name.lower()
        if self.encoder_name == "resnet50":
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            base = tv_models.resnet50(weights=weights)
            enc_channels = [64, 256, 512, 1024, 2048]
        elif self.encoder_name == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            base = tv_models.resnet18(weights=weights)
            enc_channels = [64, 64, 128, 256, 512]
        else:  # Default: resnet34
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            base = tv_models.resnet34(weights=weights)
            enc_channels = [64, 64, 128, 256, 512]

        # Encoder stem
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = base.conv1

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # Encoder residual blocks
        self.layer1 = base.layer1  # 1/4
        self.layer2 = base.layer2  # 1/8
        self.layer3 = base.layer3  # 1/16
        self.layer4 = base.layer4  # 1/32

        # Decoder stages
        self.dec4 = ResNetDecoderBlock(enc_channels[4], enc_channels[3], 256)
        self.dec3 = ResNetDecoderBlock(256, enc_channels[2], 128)
        self.dec2 = ResNetDecoderBlock(128, enc_channels[1], 64)
        self.dec1 = ResNetDecoderBlock(64, enc_channels[0], 32)
        self.dec0 = ResNetDecoderBlock(32, 0, 16)
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass
        x0 = self.relu(self.bn1(self.conv1(x)))  # (B, 64, H/2, W/2)
        x1 = self.maxpool(x0)                    # (B, 64, H/4, W/4)
        x1 = self.layer1(x1)                     # (B, enc[1], H/4, W/4)
        x2 = self.layer2(x1)                     # (B, enc[2], H/8, W/8)
        x3 = self.layer3(x2)                     # (B, enc[3], H/16, W/16)
        x4 = self.layer4(x3)                     # (B, enc[4], H/32, W/32)

        # Decoder forward pass with skip connections
        d4 = self.dec4(x4, x3)                   # (B, 256, H/16, W/16)
        d3 = self.dec3(d4, x2)                   # (B, 128, H/8, W/8)
        d2 = self.dec2(d3, x1)                   # (B, 64, H/4, W/4)
        d1 = self.dec1(d2, x0)                   # (B, 32, H/2, W/2)
        d0 = self.dec0(d1)                       # (B, 16, H, W)
        logits = self.final(d0)                  # (B, out_channels, H, W)
        return logits


# ==============================================================================
# 3. Modern Pre-trained Segmentation Factory
# ==============================================================================

def build_segmentation_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to construct segmentation models with pre-trained backbones.
    """
    arch = config.get("arch", "unet").lower()
    encoder_name = config.get("encoder_name", "resnet34")
    encoder_weights = config.get("encoder_weights", "imagenet")
    in_channels = int(config.get("in_channels", 3))
    out_channels = int(config.get("out_channels", 1))

    # Priority 1: If segmentation_models_pytorch is available and not classic
    if HAS_SMP and smp is not None and encoder_name != "classic" and arch != "classic":
        logger.info(
            f"Building SMP model: arch={arch}, encoder={encoder_name}, "
            f"weights={encoder_weights}, in_channels={in_channels}, classes={out_channels}"
        )
        common_kwargs = {
            "encoder_name": encoder_name,
            "encoder_weights": encoder_weights,
            "in_channels": in_channels,
            "classes": out_channels,
            "activation": None,
        }
        if arch == "unet":
            decoder_channels = config.get("decoder_channels", (256, 128, 64, 32, 16))
            return smp.Unet(decoder_channels=decoder_channels, **common_kwargs)
        elif arch in ("unetplusplus", "unet++"):
            return smp.UnetPlusPlus(**common_kwargs)
        elif arch in ("deeplabv3plus", "deeplabv3+"):
            return smp.DeepLabV3Plus(**common_kwargs)
        elif arch == "fpn":
            return smp.FPN(**common_kwargs)

    # Priority 2: Native Torchvision ResNet U-Net (ResNet18 / ResNet34 / ResNet50)
    # Works seamlessly without SMP!
    if encoder_name.lower() in ("resnet18", "resnet34", "resnet50") and arch in ("unet", "resnet_unet"):
        logger.info(
            f"Building Native TorchvisionResNetUNet: encoder={encoder_name}, "
            f"pretrained={encoder_weights == 'imagenet'}, in_channels={in_channels}, out_channels={out_channels}"
        )
        return TorchvisionResNetUNet(
            encoder_name=encoder_name,
            pretrained=(encoder_weights == "imagenet"),
            in_channels=in_channels,
            out_channels=out_channels,
        )

    # Priority 3: Fallback to Classic Ronneberger U-Net from scratch
    logger.info(f"Building ClassicUNet (from scratch, in_channels={in_channels}, out_channels={out_channels})")
    return ClassicUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        features=config.get("features", [64, 128, 256, 512, 1024]),
        bilinear=config.get("bilinear", True),
    )

