"""Create KITTI sky-masked RGB using the positive pixels of masked metric depth.

This is a new, explicit depth-derived RGB recipe, not a recovered historical
RGB generator. Run the pinned KITTI depth preprocessing first. No model/GPU runs.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


DEPTH_SHAPE = (256, 1024)


def rgb_from_depth(image, depth):
    """Bilinearly resize RGB to the depth grid, then blacken zero-depth pixels."""
    if not isinstance(depth, torch.Tensor):
        raise TypeError('Expected a torch.Tensor depth map')
    if depth.dtype != torch.float32 or tuple(depth.shape) != DEPTH_SHAPE:
        raise ValueError('Expected float32 depth with shape (256, 1024)')
    depth = depth.detach().cpu()
    if not torch.isfinite(depth).all().item() or (depth < 0).any().item():
        raise ValueError('Depth must be finite and nonnegative')
    rgb = np.array(image.convert('RGB').resize(
        (DEPTH_SHAPE[1], DEPTH_SHAPE[0]), Image.Resampling.BILINEAR))
    rgb[depth.numpy() == 0] = 0
    return Image.fromarray(rgb)


def process_one(source, depth_path, destination):
    depth = torch.load(depth_path, map_location='cpu', weights_only=True)
    with Image.open(source) as image:
        result = rgb_from_depth(image, depth)
    if destination.exists():
        with Image.open(destination) as existing:
            if (existing.mode == 'RGB' and existing.size == result.size
                    and np.array_equal(np.asarray(existing), np.asarray(result))):
                return 'unchanged'
        raise FileExistsError(
            f'Existing RGB differs; preserve it and use a separate --output-root: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects historical outputs, including a concurrent writer.
    # An interrupted write is reported as invalid on the next run, never reused.
    with destination.open('xb') as output:
        result.save(output, format='PNG')
    return 'processed'


def collect_sources(root, limit=0):
    if limit < 0:
        raise ValueError('--limit must be nonnegative')
    directory = root / 'depth_data'
    if not directory.is_dir():
        raise FileNotFoundError(f'Missing KITTI depth_data directory: {directory}')
    sources = sorted(directory.glob('*/*/image_02/data/*.png'))
    if not sources:
        raise ValueError(f'No original KITTI PNGs found under {directory}')
    return sources[:limit] if limit else sources


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kitti-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path,
                        help='Mirror depth_data/ here; default writes only grd_no_sky/ in KITTI root')
    parser.add_argument('--limit', type=int, default=0,
                        help='Process at most N sorted PNGs; 0 means all')
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
    root = args.kitti_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else root
    try:
        sources = collect_sources(root, args.limit)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    counts = dict(processed=0, unchanged=0, failed=0)
    for source in sources:
        image_directory = source.parent.parent
        depth_path = image_directory / 'grd_depth' / (source.stem + '_grd_depth.pt')
        destination = (output_root / image_directory.relative_to(root)
                       / 'grd_no_sky' / source.name)
        try:
            counts[process_one(source, depth_path, destination)] += 1
        except Exception as error:
            counts['failed'] += 1
            print(f'{source}: {type(error).__name__}: {error}', file=sys.stderr)
    print(json.dumps(dict(counts, output_root=str(output_root),
                          rgb_size=[1024, 256], keep_rule='masked metric depth > 0'),
                     sort_keys=True))
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
