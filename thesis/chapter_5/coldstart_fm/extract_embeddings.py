#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract & cache foundation-model embeddings for D_train (Ch.5 §5.2 cold-start).
================================================================================
Runs one foundation model over ALL 2032 training images (in ImageFolder order,
so row i == train index i, identical to the indices used by run_AL / get_data),
and caches the result to:

    thesis/chapter_5/coldstart_fm/embeddings/{model_id}.pt

so that clustering / selection never has to recompute features.

Cache payload (a dict saved via torch.save):
    model_id  : str
    embeddings: FloatTensor [N, D]
    indices   : LongTensor  [N]   (0..N-1 = train positional index)
    labels    : LongTensor  [N]   (ground-truth class, for analysis only)
    classes   : list[str]

Run from repo root:
    python3 thesis/chapter_5/coldstart_fm/extract_embeddings.py \
        --model dinov2:base --device cuda:4
    # all core models:
    python3 thesis/chapter_5/coldstart_fm/extract_embeddings.py --all --device cuda:4
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from thesis.chapter_5.coldstart_fm.extractors import build_extractor, list_models  # noqa: E402

DATA_DIR = os.path.join(REPO, "ds", "classification", "seven_class")
OUT_DIR = os.path.join(REPO, "thesis", "chapter_5", "coldstart_fm", "embeddings")


def cache_path(model_id):
    return os.path.join(OUT_DIR, model_id.replace(":", "__") + ".pt")


def extract_one(model_id, device, batch_size=64, num_workers=4, overwrite=False):
    out_path = cache_path(model_id)
    if os.path.isfile(out_path) and not overwrite:
        d = torch.load(out_path, map_location="cpu", weights_only=False)
        print(f"[skip] {model_id}: cached {tuple(d['embeddings'].shape)} at {out_path}")
        return out_path

    print(f"\n=== extracting {model_id} on {device} ===")
    ex = build_extractor(model_id, device)
    input_mode = getattr(ex, "input_mode", "tensor")

    if input_mode == "path":
        # extractor consumes raw file paths (does its own preprocessing), e.g.
        # MedImageInsight. Iterate ImageFolder.samples in order (row i == train idx i).
        ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"))
        paths = [s[0] for s in ds.samples]
        labels = torch.tensor(ds.targets, dtype=torch.long)
        feats = []
        for i in tqdm(range(0, len(paths), batch_size), desc=model_id):
            batch = paths[i:i + batch_size]
            arr = ex.embed_paths(batch)               # np [B, D]
            feats.append(torch.as_tensor(arr, dtype=torch.float32))
        embeddings = torch.cat(feats).contiguous()
    else:
        # ImageFolder applies ex.transform per image; default loader -> RGB (gray replicated).
        ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=ex.transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        feats, labels = [], []
        for x, y in tqdm(loader, desc=model_id):
            feats.append(ex.embed(x))
            labels.append(y)
        embeddings = torch.cat(feats).contiguous()
        labels = torch.cat(labels).contiguous()
    n = embeddings.size(0)

    payload = {
        "model_id": model_id,
        "embeddings": embeddings.float(),
        "indices": torch.arange(n, dtype=torch.long),
        "labels": labels.long(),
        "classes": list(ds.classes),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(payload, out_path)
    print(f"[saved] {model_id}: embeddings {tuple(embeddings.shape)} "
          f"(dim={ex.embed_dim}) -> {out_path}")
    del ex
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="model id, e.g. dinov2:base")
    ap.add_argument("--all", action="store_true", help="extract every registered model")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.all:
        models = [m for ms in list_models().values() for m in ms]
    elif args.model:
        models = [args.model]
    else:
        ap.error("pass --model <id> or --all")

    print("models to extract:", models)
    for m in models:
        try:
            extract_one(m, args.device, args.batch_size, args.num_workers, args.overwrite)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] {m}: {repr(e)[:300]}")


if __name__ == "__main__":
    main()
