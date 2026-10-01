"""
Modular U-Net architecture for biomedical semantic segmentation.
Leverages pre-trained ImageNet backbones (ResNet34, ResNet50, EfficientNet, etc.)
via segmentation_models_pytorch.
"""

from typing import Dict, Any
import logging
import torch.nn as nn
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


def build_segmentation_model(config: Dict[str, Any]) -> nn.Module:
    """
    Construct a segmentation model with a pre-trained backbone.
    Outputs raw logits (activation=None) for numerical stability with ComboLoss / BCEWithLogitsLoss.
    """
    arch = config.get("arch", "unet").lower()
    encoder_name = config.get("encoder_name", "resnet34")
    encoder_weights = config.get("encoder_weights", "imagenet")
    in_channels = int(config.get("in_channels", 3))
    out_channels = int(config.get("out_channels", 1))

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

    if arch in ("unetplusplus", "unet++"):
        return smp.UnetPlusPlus(**common_kwargs)
    elif arch in ("deeplabv3plus", "deeplabv3+"):
        return smp.DeepLabV3Plus(**common_kwargs)
    elif arch == "fpn":
        return smp.FPN(**common_kwargs)

    decoder_channels = config.get("decoder_channels", (256, 128, 64, 32, 16))
    return smp.Unet(decoder_channels=decoder_channels, **common_kwargs)


# Backwards compatibility alias
UNet = smp.Unet
