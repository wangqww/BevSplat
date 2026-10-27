#!/usr/bin/env python3
"""Portable path/device orchestration for the pinned UniK3D VIGOR wrapper.

The dependency's camera, resize, train-mode inference and distance filtering stay
unchanged. This avoids its old hardcoded CUDA2 CLI and output-root side effects.
"""
import argparse
import json
from pathlib import Path
import sys

CITIES = {"NewYork", "Seattle", "SanFrancisco", "Chicago"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--unik3d-root", type=Path, required=True)
    p.add_argument("--vigor-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--area", choices=["same", "cross"], required=True)
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--device", default="auto", help="auto, cpu, or a visible cuda:N")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="0=complete split; positive=preview prefix")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Check selected paths without loading weights")
    a = p.parse_args(argv)
    if a.batch_size < 1 or a.limit < 0:
        p.error("batch-size must be positive; limit must be nonnegative")
    return a


def collect_jobs(paths, vigor_root, output_root, area, *, limit=0, overwrite=False):
    paths = paths[:limit] if limit else paths
    jobs = []
    for item in paths:
        image = Path(item)
        relative = image.relative_to(vigor_root)
        if (len(relative.parts) != 3 or relative.parts[0] not in CITIES
                or relative.parts[1] != "pano_mask_sky" or image.suffix != ".jpg"):
            raise ValueError(f"Unexpected VIGOR panorama path: {image}")
        if not image.is_file():
            raise FileNotFoundError(f"Required masked panorama is missing: {image}")
        output = output_root / relative.parts[0] / f"UniK3D_{area}_metric" / (image.stem + "_depth.npy")
        if output.exists() and not overwrite:
            raise FileExistsError(f"Output already exists; choose a new --output-root or --overwrite: {output}")
        jobs.append((str(image), str(output)))
    return jobs


def main(argv=None):
    args = parse_args(argv)
    source = args.unik3d_root.expanduser().resolve()
    for name in ["dataset_vigor.py", "model_vigor.py"]:
        if not (source / name).is_file():
            raise FileNotFoundError(f"--unik3d-root must contain {name}: {source}")
    root = args.vigor_root.expanduser().resolve()
    output = (args.output_root or root).expanduser().resolve()
    sys.path.insert(0, str(source))
    # Constructor only: __getitem__ in the historical dependency writes a hardcoded root.
    from dataset_vigor import VIGORDataset
    dataset = VIGORDataset(str(root), rotation_range=0, split=args.area,
                           train=args.split == "train", amount=1.)
    jobs = collect_jobs(dataset.grd_list, root, output, args.area,
                        limit=args.limit, overwrite=args.overwrite)
    summary = {"area": args.area, "split": args.split, "samples": len(jobs),
               "output_root": str(output), "preview": bool(args.limit),
               "first_output": jobs[0][1] if jobs else None}
    if args.dry_run:
        print(json.dumps({**summary, "dry_run": True}))
        return
    if not jobs:
        raise ValueError("Selected split is empty")
    import numpy as np
    import torch
    from model_vigor import ModelVigor
    np.random.seed(2022)  # Preserve the original preprocessing runner's seed.
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    net = ModelVigor(args, device).to(device)
    net.train()  # Intentional: preserve the existing runner, not an eval-mode rewrite.
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start:start + args.batch_size]
        images, outputs = zip(*batch)
        net.create_data(list(images), list(outputs))
        print(f"saved {start + len(batch)}/{len(jobs)}", flush=True)
    print(json.dumps({**summary, "device": str(device), "mode": "historical train-mode inference"}))


if __name__ == "__main__":
    main()
