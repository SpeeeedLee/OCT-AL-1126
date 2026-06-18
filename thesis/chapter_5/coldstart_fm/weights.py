#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resolve / download external foundation-model weights that are not auto-fetched
by transformers/torchvision (i.e. RadImageNet and friends).

We cache to the HF hub cache (default) and just return the local path.
"""
import os


def resolve_radimagenet_weights():
    """RadImageNet ResNet-50 (PyTorch port).

    Source: HF mirror `Lab-Rasool/RadImageNet` -> `ResNet50.pt`.
    The state_dict is keyed `backbone.{0..7}` (a Sequential of resnet50
    children[:-1]); load it into `_RadResNet50` in extractors.py.

    Override with env RADIMAGENET_RESNET50_PT=/abs/path.pt if you have it locally.
    """
    env = os.environ.get("RADIMAGENET_RESNET50_PT")
    if env and os.path.isfile(env):
        return env
    from huggingface_hub import hf_hub_download
    return hf_hub_download("Lab-Rasool/RadImageNet", "ResNet50.pt")
