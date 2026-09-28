"""Render per-image relative depth from existing metric arrays, without inference.

Only visualization is normalized: P1..P99 of finite positive radial distances
map to 0..1. The outer percentiles saturate. No filtering or geometry edits.
Color: near blue -> far red. Grayscale: near white -> far black.
"""
from pathlib import Path
import argparse
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, PngImagePlugin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parent.parent


def normalize_relative(depth, lower=1.0, upper=99.0):
    if depth.ndim != 2:
        raise ValueError('Expected a 2D depth array')
    if not 0 <= lower < upper <= 100:
        raise ValueError('Expected 0 <= lower < upper <= 100')
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError('Depth has no finite positive samples')
    low, high = np.percentile(depth[valid], [lower, upper])
    relative = np.zeros(depth.shape, dtype=np.float32)
    if high > low:
        relative[valid] = np.clip((depth[valid] - low) / (high - low), 0, 1)
    else:
        # A flat-depth image contains no relative ordering to stretch.
        relative[valid] = 0.5
    return relative, valid, (float(low), float(high))


def colorize(relative, valid):
    u8 = np.rint(relative * 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb


def grayscale(relative, valid):
    gray = np.rint((1 - relative) * 255).astype(np.uint8)
    gray[~valid] = 0
    return gray


def fingerprint(paths):
    hashes = {}
    for path in paths:
        with path.open('rb') as stream:
            hashes[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'depthmaster-results')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'depthmaster-relative-depth')
    parser.add_argument('--overview', type=Path,
                        default=Path(__file__).resolve().parent / 'preview' / 'relative-depth-comparison.png')
    parser.add_argument('--lower-percentile', type=float, default=1.0)
    parser.add_argument('--upper-percentile', type=float, default=99.0)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    sources = sorted(args.input_dir.glob('*.range_m.npy'))
    if not sources:
        raise FileNotFoundError(f'No *.range_m.npy files in {args.input_dir}')
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError('Use a separate visualization directory')
    names = [source.name.removesuffix('.range_m.npy') for source in sources]
    targets = [args.output_dir / f'{name}.relative.{kind}.png'
               for name in names for kind in ('color', 'gray')] + [args.overview]
    if not args.overwrite and any(path.exists() for path in targets):
        raise FileExistsError('Visualization exists; choose another directory/overview or use --overwrite')
    protected = sorted(path for path in args.input_dir.iterdir() if path.is_file())
    before = fingerprint(protected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.overview.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(sources), 3, squeeze=False,
                                figsize=(18, 3.3 * len(sources) + 1), facecolor='white')
    for row, (source, name) in enumerate(zip(sources, names)):
        depth = np.load(source, allow_pickle=False)
        relative, valid, limits = normalize_relative(
            depth, args.lower_percentile, args.upper_percentile)
        rgb = colorize(relative, valid)
        gray = grayscale(relative, valid)
        metadata = {
            'source': str(source.resolve()), 'units': 'unitless; visualization only',
            'normalization': 'per-image clipped linear metric range',
            'percentiles': [args.lower_percentile, args.upper_percentile],
            'metric_bounds_m': list(limits), 'invalid': 'black',
            'color_order': 'blue near, red far', 'gray_order': 'white near, black far',
        }
        info = PngImagePlugin.PngInfo()
        info.add_text('DepthVisualization', json.dumps(metadata))
        for kind, array in (('color', rgb), ('gray', gray)):
            target = args.output_dir / f'{name}.relative.{kind}.png'
            Image.fromarray(array).save(target, pnginfo=info)
            with Image.open(target) as check:
                check.load()
                assert check.size == (depth.shape[1], depth.shape[0])
                np.testing.assert_array_equal(np.asarray(check), array)
        absolute = np.zeros_like(relative)
        absolute[valid] = np.clip(depth[valid] / 10, 0, 1)
        images = [colorize(absolute, valid), rgb, gray]
        titles = [f'{name} | fixed 0-10 m',
                  f'{name} | per-image P{args.lower_percentile:g}-P{args.upper_percentile:g}',
                  f'{name} | relative grayscale']
        for column, (array, title) in enumerate(zip(images, titles)):
            axes[row, column].imshow(array, cmap='gray' if column == 2 else None,
                                     vmin=0 if column == 2 else None,
                                     vmax=255 if column == 2 else None)
            axes[row, column].set_title(title, loc='left', fontsize=12, pad=7)
            axes[row, column].axis('off')
        print('EXPORTED', json.dumps({
            'name': name, 'size_wh': [depth.shape[1], depth.shape[0]],
            'metric_bounds_m': list(limits), 'valid_fraction': float(valid.mean()),
            'gray_levels': int(len(np.unique(gray[valid]))),
        }), flush=True)
    figure.suptitle('Depth display comparison | same predictions, no geometry changes', fontsize=18, y=0.988)
    figure.subplots_adjust(left=0.025, right=0.985, top=0.956, bottom=0.065, wspace=0.04, hspace=0.15)
    color_axis = figure.add_axes([0.36, 0.031, 0.25, 0.01])
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap='turbo'),
                              cax=color_axis, orientation='horizontal')
    colorbar.set_ticks([0, 0.5, 1], labels=['near (0)', '0.5', 'far (1)'])
    figure.text(0.67, 0.026, 'Grayscale: white = near | black = far', fontsize=11)
    figure.text(0.025, 0.009, 'Relative scales are independent per image. P1/P99 tails saturate. '
                'Use the original .range_m.npy for backprojection.', fontsize=10, color='#444444')
    figure.savefig(args.overview, dpi=150)
    plt.close(figure)
    after = fingerprint(protected)
    if after != before:
        raise RuntimeError('Original input artifacts changed during export')
    print(f'VERIFIED {len(targets) - 1} full-resolution PNGs; {len(protected)} original files unchanged', flush=True)
    print(f'OVERVIEW {args.overview.resolve()}', flush=True)


if __name__ == '__main__':
    main()
