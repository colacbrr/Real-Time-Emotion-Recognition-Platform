#!/usr/bin/env python3
"""Render local contact sheets for image-duplication review, without identity inference."""
import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import map_coordinates
from skimage.metrics import structural_similarity

from evaluation import digest, read_csv, write_json


@lru_cache(maxsize=1)
def alignment_grid():
    y, x = np.mgrid[4:44, 4:44].astype(float)
    parameters = [(s, dy, dx) for s in np.linspace(.85, 1.15, 13)
                  for dy in range(-3, 4) for dx in range(-3, 4)]
    coords = np.array([[(y - 23.5) / s + 23.5 + dy,
                        (x - 23.5) / s + 23.5 + dx]
                       for s, dy, dx in parameters]).transpose(1, 0, 2, 3)
    return parameters, coords


def aligned_evidence(a, b):
    """Supporting similarity, never an automatic duplicate/identity verdict."""
    if a.shape != (48, 48) or b.shape != (48, 48):
        raise ValueError('Alignment review expects 48x48 grayscale images')
    parameters, coords = alignment_grid()
    centered_a = a[4:44, 4:44].astype(float).ravel()
    centered_a -= centered_a.mean()
    aligned_b = map_coordinates(b.astype(float), coords, order=1, mode='nearest').reshape(len(parameters), -1)
    aligned_b -= aligned_b.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(centered_a) * np.linalg.norm(aligned_b, axis=1)
    ncc = np.divide(aligned_b @ centered_a, denominator,
                    out=np.zeros(len(parameters)), where=denominator != 0)
    best = int(ncc.argmax())
    scale, dy, dx = parameters[best]
    return {'aligned_ncc': f'{ncc[best]:.6f}', 'scale': f'{scale:.3f}',
            'shift_y': str(dy), 'shift_x': str(dx)}


def render_review(recovery, source, output):
    if output.exists():
        raise ValueError('Use a fresh output directory')
    audit = json.loads((recovery / 'audit.json').read_text())
    manifest = recovery / 'historical_manifest.csv'
    if digest(manifest) != audit['manifest_sha256'] or str(source.resolve()) != audit['source_root']:
        raise ValueError('Source or manifest does not match recovery audit')
    rows = {r['id']: r for r in read_csv(manifest)}
    candidates = read_csv(recovery / 'similarity_candidates.csv')
    images = {}
    for ident in {p[k] for p in candidates for k in ('id_a', 'id_b')}:
        path = source / rows[ident]['path']
        if digest(path) != rows[ident]['sha256']:
            raise ValueError(f'Image changed: {ident}')
        with Image.open(path) as im:
            images[ident] = ImageOps.exif_transpose(im).convert('L').copy()
    review = []
    for i, pair in enumerate(candidates, 1):
        a, b = (np.asarray(images[pair[k]], dtype=np.float64) for k in ('id_a', 'id_b'))
        if a.shape != b.shape:
            raise ValueError('This review expects equal image dimensions')
        review.append({'pair_index': str(i), **pair,
                       'ssim': f'{structural_similarity(a, b, data_range=255):.6f}',
                       'rmse': f'{np.sqrt(np.mean((a - b) ** 2)):.6f}',
                       **aligned_evidence(a, b),
                       'decision': '', 'reason': ''})
    output.mkdir(parents=True)
    per_page, card_w, card_h = 18, 270, 145
    for first in range(0, len(review), per_page):
        sheet = Image.new('RGB', (3 * card_w, 6 * card_h + 30), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 6), 'Image duplication review: labels hidden; no person identification', fill='black')
        for j, pair in enumerate(review[first:first + per_page]):
            x, y = (j % 3) * card_w, 30 + (j // 3) * card_h
            draw.text((x + 8, y), f"#{pair['pair_index']}  {pair['id_a']} / {pair['id_b']}  d={pair['dhash_distance']}", fill='black')
            for offset, key in ((8, 'id_a'), (116, 'id_b')):
                sheet.paste(images[pair[key]].resize((96, 96), Image.Resampling.NEAREST), (x + offset, y + 16))
            draw.text((x + 8, y + 115), f"SSIM {pair['ssim']} RMSE {float(pair['rmse']):.1f}", fill='black')
        sheet.save(output / f'candidates_{first // per_page + 1:02}.png')
    with (output / 'review_template.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(review[0]) if review else ['pair_index', 'decision', 'reason'])
        writer.writeheader()
        writer.writerows(review)
    write_json(output / 'provenance.json', {
        'manifest_sha256': digest(manifest), 'audit_sha256': digest(recovery / 'audit.json'),
        'candidates_sha256': digest(recovery / 'similarity_candidates.csv'),
        'implementation_sha256': digest(Path(__file__)), 'n_pairs': len(review),
        'rendering': 'original grayscale pixels enlarged 2x nearest-neighbor; class labels hidden',
        'alignment': 'NCC on central 40x40 pixels; 13 scales 0.85..1.15; integer shifts -3..3 px; linear interpolation; best of 637 transforms',
        'meaning': 'review image duplication only; same person is not inferred',
        'limits': 'candidate screening is not exhaustive; SSIM/RMSE are supporting evidence, not automatic verdicts'})
    return review


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = render_review(args.recovery, args.source, args.output)
    print(f'Rendered {len(rows)} pairs; no decisions assigned automatically.')
