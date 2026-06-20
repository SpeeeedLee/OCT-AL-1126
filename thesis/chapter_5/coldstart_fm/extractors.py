#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foundation-model feature extractors for cold-start active learning (Ch.5 §5.2).
================================================================================
A uniform interface so that *any* foundation model can be plugged into the
clustering-based initial-set selection of:

    "From Cold Start to Active Learning: Embedding-Based Scan Selection for
     Medical Image Segmentation" (arXiv:2601.18532, 2026)

Design (per user request):
  * One **base class** `FoundationExtractor` defines the shared contract.
  * One **subclass per model family** (TorchVision ResNet, DINOv2, CLIP,
    RadImageNet, RETFound, BiomedCLIP, MedImageInsight). Different families load
    / forward very differently, so they do NOT share a subclass; models *within*
    a family (different sizes) share their subclass and only differ by an arg.
  * For models that ship a **classification head**, we take the **penultimate
    embedding** (the representation that feeds the head), never the logits.

Every extractor exposes:
    .model_id     str   unique id, used as the cache filename
    .family       str
    .embed_dim    int   (filled after first forward if unknown)
    .transform    callable: PIL.Image(RGB) -> Tensor[3,H,W]
    .embed(batch) Tensor[B, D]   (no_grad, on cpu, float32)

Our OCT images are grayscale; ImageFolder's default loader already converts to
RGB (3 identical channels), and every FM here expects 3-channel input, so we
just feed the RGB-replicated image through each model's own canonical
preprocessing.

The registry `build_extractor(model_id, device)` maps an id -> instance.
`list_models()` returns every known id grouped by family.
"""
import torch
import torch.nn as nn
from torchvision import transforms


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class FoundationExtractor:
    """Contract shared by all families."""

    family = "base"

    def __init__(self, model_id, device):
        self.model_id = model_id
        self.device = device
        self.embed_dim = None          # filled by subclass or after first forward
        self.model = None
        self.transform = None

    @torch.no_grad()
    def embed(self, batch):
        """batch: Tensor[B,3,H,W] on cpu -> Tensor[B,D] float32 on cpu."""
        batch = batch.to(self.device, non_blocking=True)
        feats = self._forward(batch)
        feats = feats.float().reshape(feats.size(0), -1).cpu()
        if self.embed_dim is None:
            self.embed_dim = feats.size(1)
        return feats

    def _forward(self, batch):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# ImageNet normalization helper
# --------------------------------------------------------------------------- #
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _imagenet_transform(size=224):
    # Resize whole image to size x size (no crop) to keep the full OCT field.
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Family 1: TorchVision ResNet (ImageNet-supervised)  -> penultimate (pre-fc)
# --------------------------------------------------------------------------- #
class TorchVisionResNetExtractor(FoundationExtractor):
    family = "resnet_imagenet"
    # model_id suffix -> (torchvision ctor name, weights enum name, dim)
    _SPECS = {
        "resnet18": ("resnet18", "ResNet18_Weights", 512),
        "resnet34": ("resnet34", "ResNet34_Weights", 512),
        "resnet50": ("resnet50", "ResNet50_Weights", 2048),
        "resnet101": ("resnet101", "ResNet101_Weights", 2048),
        "resnet152": ("resnet152", "ResNet152_Weights", 2048),
    }

    def __init__(self, model_id, device):
        super().__init__(model_id, device)
        from torchvision import models
        arch = model_id.split(":")[1]
        ctor_name, weights_name, dim = self._SPECS[arch]
        weights = getattr(models, weights_name).IMAGENET1K_V1
        net = getattr(models, ctor_name)(weights=weights)
        net.fc = nn.Identity()          # drop classification head -> pooled feature
        self.model = net.eval().to(device)
        self.embed_dim = dim
        self.transform = _imagenet_transform(224)

    def _forward(self, batch):
        return self.model(batch)


# --------------------------------------------------------------------------- #
# Family 2: DINOv2 (HF transformers)  -> CLS / pooler_output
# --------------------------------------------------------------------------- #
class DINOv2Extractor(FoundationExtractor):
    family = "dinov2"
    _SPECS = {
        "small": ("facebook/dinov2-small", 384),
        "base": ("facebook/dinov2-base", 768),
        "large": ("facebook/dinov2-large", 1024),
    }

    def __init__(self, model_id, device):
        super().__init__(model_id, device)
        from transformers import AutoModel, AutoImageProcessor
        size = model_id.split(":")[1]
        hf_name, dim = self._SPECS[size]
        self.model = AutoModel.from_pretrained(hf_name).eval().to(device)
        self.embed_dim = dim
        proc = AutoImageProcessor.from_pretrained(hf_name)
        self.transform = _hf_processor_transform(proc)

    def _forward(self, batch):
        out = self.model(pixel_values=batch)
        # DINOv2: pooler_output is the CLS token after layernorm.
        return out.pooler_output


# --------------------------------------------------------------------------- #
# Family 3: CLIP image encoder (HF transformers)  -> projected image embedding
# --------------------------------------------------------------------------- #
class CLIPExtractor(FoundationExtractor):
    family = "clip"
    _SPECS = {
        "base": ("openai/clip-vit-base-patch16", 512),
        "large": ("openai/clip-vit-large-patch14", 768),
    }

    def __init__(self, model_id, device):
        super().__init__(model_id, device)
        from transformers import CLIPModel, CLIPImageProcessor
        size = model_id.split(":")[1]
        hf_name, dim = self._SPECS[size]
        self.model = _from_pretrained_safe(CLIPModel, hf_name).eval().to(device)
        self.embed_dim = dim
        proc = CLIPImageProcessor.from_pretrained(hf_name)
        self.transform = _hf_processor_transform(proc)

    def _forward(self, batch):
        # image part of the dual encoder -> projected joint-space embedding
        return self.model.get_image_features(pixel_values=batch)


# --------------------------------------------------------------------------- #
# Family 4: RadImageNet ResNet-50 (the paper's extractor)  -> penultimate
# --------------------------------------------------------------------------- #
class _RadResNet50(nn.Module):
    """ResNet-50 truncated to the avgpool (matches the published RadImageNet
    state_dict, which is keyed `backbone.{0..7}` = a Sequential of children[:-1])."""

    def __init__(self):
        super().__init__()
        from torchvision import models
        net = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(net.children())[:-1])   # conv1..layer4..avgpool

    def forward(self, x):
        return self.backbone(x).flatten(1)                          # [B, 2048]


class RadImageNetExtractor(FoundationExtractor):
    family = "radimagenet"
    _SPECS = {"resnet50": 2048}

    def __init__(self, model_id, device, weight_path=None):
        super().__init__(model_id, device)
        from .weights import resolve_radimagenet_weights
        net = _RadResNet50()
        wp = weight_path or resolve_radimagenet_weights()
        sd = torch.load(wp, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = net.load_state_dict(sd, strict=False)
        # only avgpool (no params) should be missing; nothing should be unexpected
        param_missing = [m for m in missing if "num_batches_tracked" not in m]
        print(f"  [RadImageNet] loaded {wp}")
        print(f"  [RadImageNet] missing={len(missing)} unexpected={len(unexpected)}")
        if unexpected or param_missing:
            print(f"  [RadImageNet][warn] missing={param_missing[:6]} unexpected={unexpected[:6]}")
        self.model = net.eval().to(device)
        self.embed_dim = 2048
        # RadImageNet port: [0,1] then ImageNet-style normalization (see weights.py note).
        self.transform = _imagenet_transform(224)

    def _forward(self, batch):
        return self.model(batch)


# --------------------------------------------------------------------------- #
# Family 4b: OUR OWN SimCLR (θ²) ResNet-18  -> 512-d backbone penultimate
#   The exact checkpoint the downstream one-shot trainer initialises from
#   (build_simclr_classifier). Uses the codebase's own preprocessing (full-res,
#   [0.5] normalize) — i.e. the features exactly as used elsewhere in the thesis,
#   NOT 224-resize. This is the "in-domain self-trained" comparison point.
# --------------------------------------------------------------------------- #
class SimCLRExtractor(FoundationExtractor):
    family = "simclr"
    _SPECS = {"resnet18": 512, "resnet50": 2048, "resnet101": 2048, "resnet152": 2048}
    # our θ² SimCLR ckpts share this name pattern (best cfg lr2e-4/bs256/ep500)
    _CKPT_TMPL = "SSL/simclr/ckpt/{arch}_simclr_lr0.0002_bs256_ep500.pkl"

    def __init__(self, model_id, device, ckpt=None, **kw):
        super().__init__(model_id, device)
        import os
        from classification.model.simclr.resnet_simclr import ResNetSimCLR
        size = model_id.split(":")[1]                       # resnet50 | resnet152_best | ...
        is_best = size.endswith("_best")                    # use the lowest-val-loss ckpt
        arch = size[:-5] if is_best else size               # strip "_best"
        dim = self._SPECS[arch]
        model = ResNetSimCLR(arch, 32)
        if ckpt:
            path = ckpt
        else:
            fn = f"{arch}_simclr_lr0.0002_bs256_ep500" + ("_wval_best" if is_best else "") + ".pkl"
            path = os.path.join(_REPO_ROOT(), "SSL", "simclr", "ckpt", fn)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"SimCLR ckpt not found: {path}. Pretrain it first "
                f"(SSL/simclr/run.py -a {arch} ...) or pass ckpt=<path>.")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  [SimCLR:{size}] loaded {path}  missing={len(missing)} unexpected={len(unexpected)}")
        # backbone penultimate: everything up to avgpool (drop the MLP projection head)
        self.model = nn.Sequential(*list(model.backbone.children())[:-1]).eval().to(device)
        self.embed_dim = dim
        # codebase convention (get_data val transform): full-res, [0.5] normalize, no resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def _forward(self, batch):
        return self.model(batch).flatten(1)


# --------------------------------------------------------------------------- #
# Family 5: BiomedCLIP image encoder (open_clip)  -> projected image embedding
# --------------------------------------------------------------------------- #
class BiomedCLIPExtractor(FoundationExtractor):
    family = "biomedclip"
    _SPECS = {
        # 15M PubMedCentral image-text pairs; ViT-B/16 image tower -> 512-d
        "base": ("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", 512),
    }

    def __init__(self, model_id, device, **kw):
        super().__init__(model_id, device)
        import open_clip
        size = model_id.split(":")[1]
        hf_name, dim = self._SPECS[size]
        model, preprocess = open_clip.create_model_from_pretrained(hf_name)
        self.model = model.eval().to(device)
        self.embed_dim = dim
        self.transform = preprocess          # open_clip val transform: PIL -> Tensor[3,H,W]

    def _forward(self, batch):
        return self.model.encode_image(batch)


# --------------------------------------------------------------------------- #
# Family 6: RETFound (retinal OCT, MAE ViT-L/16)  -> CLS token
#   NOTE: weights are GATED on HF (YukunZhou/RETFound_mae_natureOCT). Request
#   access on the model page and `huggingface-cli login` (or set HF_TOKEN) first.
#   We load the official .pth into a plain timm ViT-L/16 — NO remote code.
# --------------------------------------------------------------------------- #
class RETFoundExtractor(FoundationExtractor):
    family = "retfound"
    _SPECS = {
        # variant -> (HF repo, weight file, dim).  All MAE ViT-L/16, 1024-d CLS.
        # Nature-2023 modality-specific weights (pretrained on MEH-MIDAS, Moorfields):
        "oct": ("YukunZhou/RETFound_mae_natureOCT", "RETFound_mae_natureOCT.pth", 1024),
        "cfp": ("YukunZhou/RETFound_mae_natureCFP", "RETFound_mae_natureCFP.pth", 1024),
        # 2024 re-pretrains on other cohorts (separately gated). NOTE: these are
        # **CFP (color fundus)**, NOT OCT — only ':oct' is OCT modality.
        "meh": ("YukunZhou/RETFound_mae_meh", "RETFound_mae_meh.pth", 1024),            # CFP, AlzEye (Moorfields, London)
        "shanghai": ("YukunZhou/RETFound_mae_shanghai", "RETFound_mae_shanghai.pth", 1024),  # CFP, SDPP (Shanghai)
    }

    def __init__(self, model_id, device, **kw):
        super().__init__(model_id, device)
        import timm
        from huggingface_hub import hf_hub_download
        size = model_id.split(":")[1]
        repo, fname, dim = self._SPECS[size]
        # MAE ViT-L/16 encoder; global_pool='token' -> CLS embedding (1024-d)
        net = timm.create_model("vit_large_patch16_224", pretrained=False,
                                num_classes=0, global_pool="token", img_size=224)
        try:
            wp = hf_hub_download(repo, fname)
        except Exception as e:
            raise RuntimeError(
                f"RETFound weights are gated. Request access at "
                f"https://huggingface.co/{repo} and run `huggingface-cli login` "
                f"(or export HF_TOKEN=...). Original error: {repr(e)[:160]}")
        ck = torch.load(wp, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck) if isinstance(ck, dict) else ck
        sd = {k: v for k, v in sd.items() if not k.startswith(("mask_token", "decoder"))}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f"  [RETFound:{size}] loaded {wp}")
        print(f"  [RETFound:{size}] missing={len(missing)} unexpected={len(unexpected)}")
        self.model = net.eval().to(device)
        self.embed_dim = dim
        self.transform = _imagenet_transform(224)

    def _forward(self, batch):
        return self.model(batch)


# --------------------------------------------------------------------------- #
# Family 7: MedImageInsight (Microsoft UniCL, local package in feature_extract/)
#   Different interface: consumes base64 image BYTES (does its own preprocessing
#   internally), so input_mode='path' — extract_embeddings feeds file paths, not
#   tensors. We force model.eval() (the bundled load_model() omits it, leaving
#   dropout ON -> non-deterministic features).
# --------------------------------------------------------------------------- #
class MedImageInsightExtractor(FoundationExtractor):
    family = "medimageinsight"
    input_mode = "path"                      # consumes file paths, not tensors
    _SPECS = {"base": 1024}
    _MODEL_DIR = "feature_extract/MedImageInsights/2024.09.27"

    def __init__(self, model_id, device, model_dir=None, **kw):
        super().__init__(model_id, device)
        import os
        from feature_extract.MedImageInsights.medimageinsightmodel import MedImageInsight
        md = model_dir or os.path.join(_REPO_ROOT(), self._MODEL_DIR)
        clf = MedImageInsight(model_dir=md,
                              vision_model_name="medimageinsigt-v1.0.0.pt",
                              language_model_name="language_model.pth")
        clf.load_model()
        clf.model.eval()                     # CRITICAL: deterministic features
        clf.model.to(device)
        clf.device = device                  # encode() moves inputs to clf.device
        self.clf = clf
        self.embed_dim = 1024
        self.transform = None                # unused in path mode

    def embed_paths(self, paths):
        """list[str] file paths -> np.ndarray [B, 1024] (float32)."""
        import base64
        b64 = [base64.encodebytes(open(p, "rb").read()).decode("utf-8") for p in paths]
        return self.clf.encode(images=b64)["image_embeddings"]


def _REPO_ROOT():
    import os
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# --------------------------------------------------------------------------- #
# HF processor -> torchvision-style transform wrapper
# --------------------------------------------------------------------------- #
def _from_pretrained_safe(model_cls, hf_name):
    """transformers blocks torch.load on torch<2.6 for .bin-only repos. Prefer
    safetensors; otherwise build from config and load the .bin via our own
    torch.load(weights_only=True) (which IS safe and allowed on torch 2.5.1)."""
    try:
        return model_cls.from_pretrained(hf_name, use_safetensors=True)
    except Exception:
        pass
    try:
        return model_cls.from_pretrained(hf_name)
    except ValueError as e:
        if "vulnerability" not in str(e) and "weights_only" not in str(e):
            raise
    from huggingface_hub import hf_hub_download
    cfg = model_cls.config_class.from_pretrained(hf_name)
    model = model_cls(cfg)
    bin_path = hf_hub_download(hf_name, "pytorch_model.bin")
    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [m for m in missing if "position_ids" not in m]
    if real_missing or unexpected:
        print(f"  [{hf_name}] load_state_dict missing={real_missing[:5]} "
              f"unexpected={unexpected[:5]}")
    return model


def _hf_processor_transform(processor):
    """Wrap a HF image processor so it returns a single Tensor[3,H,W] from a PIL
    image (so it composes with a standard Dataset/DataLoader)."""
    def _fn(pil_img):
        out = processor(images=pil_img, return_tensors="pt")
        return out["pixel_values"][0]
    return _fn


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_FAMILY_OF = {
    "resnet_imagenet": (TorchVisionResNetExtractor,
                        ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]),
    "dinov2": (DINOv2Extractor, ["small", "base", "large"]),
    "clip": (CLIPExtractor, ["base", "large"]),
    "radimagenet": (RadImageNetExtractor, ["resnet50"]),
    "simclr": (SimCLRExtractor, ["resnet18", "resnet50", "resnet101", "resnet152",
                                 "resnet50_best", "resnet152_best"]),  # +_best = lowest-val ckpt
    "biomedclip": (BiomedCLIPExtractor, ["base"]),
    "retfound": (RETFoundExtractor, ["oct", "cfp", "meh", "shanghai"]),   # GATED weights (HF token)
    "medimageinsight": (MedImageInsightExtractor, ["base"]),  # local feature_extract/ package
}


def list_models():
    """Return {family: [model_id, ...]}."""
    out = {}
    for fam, (_, sizes) in _FAMILY_OF.items():
        out[fam] = [f"{fam}:{s}" for s in sizes]
    return out


def build_extractor(model_id, device, **kw):
    """model_id format '<family>:<size>', e.g. 'dinov2:base'."""
    fam = model_id.split(":")[0]
    if fam not in _FAMILY_OF:
        raise ValueError(f"unknown family '{fam}'. known: {list(_FAMILY_OF)}")
    cls, _ = _FAMILY_OF[fam]
    return cls(model_id, device, **kw)


if __name__ == "__main__":
    import json
    print(json.dumps(list_models(), indent=2))
