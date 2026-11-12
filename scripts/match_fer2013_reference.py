#!/usr/bin/env python3
"""Match recovered image content against a public FER2013 distribution by hashes.

Official ASPICV predict labels and reference class-bearing paths are withheld from exports.
No labels, splits, training artifacts or prior results are modified.
"""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps

from evaluation import digest, pixel_hash, read_csv, write_json

LABELS = {'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'}


def match(manifest, source, reference, output):
    if output.exists():
        raise ValueError('Use a fresh output directory')
    provenance = json.loads((reference / 'download_provenance.json').read_text())
    for name, item in provenance['files'].items():
        if digest(reference / name) != item['sha256']:
            raise ValueError(f'Reference artifact changed: {name}')
    metadata = json.loads((reference / 'metadata.json').read_text())
    local = read_csv(manifest)
    if not local or len({r['id'] for r in local}) != len(local):
        raise ValueError('Empty manifest or duplicate local ID')
    for row in local:
        path = (source / row['path']).resolve()
        path.relative_to(source.resolve())
        if digest(path) != row['sha256'] or pixel_hash(path) != row['pixel_sha256']:
            raise ValueError(f'Local image changed: {row["id"]}')
    byte_index, pixel_index = defaultdict(list), defaultdict(list)
    ref_counts = Counter()
    with zipfile.ZipFile(reference / 'fer2013-msambare-v1.zip') as archive:
        for item in archive.infolist():
            if item.is_dir() or PurePosixPath(item.filename).suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            data = archive.read(item)
            byte_sha = hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as im:
                rgb = ImageOps.exif_transpose(im).convert('RGB')
                pixel_sha = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            parts = PurePosixPath(item.filename).parts
            label = next((x for x in parts if x in LABELS), '')
            split = next((x for x in parts if x in ('train', 'test')), '')
            record = {'path': item.filename, 'label': label, 'split': split}
            byte_index[byte_sha].append(record)
            pixel_index[pixel_sha].append(record)
            ref_counts[f'{split}/{label}'] += 1
    matches, overview = [], []
    for row in local:
        byte_matches = byte_index.get(row['sha256'], [])
        pixels = pixel_index.get(row['pixel_sha256'], [])
        strongest = byte_matches or pixels
        method = 'byte_exact' if byte_matches else 'pixel_exact' if pixels else 'unmatched'
        known = row['split'] != 'predict' and bool(row['label'])
        ref_labels = {r['label'] for r in strongest if r['label']}
        agreement = ('not_evaluated_official_unlabeled' if not known else 'unmatched' if not strongest else
                     'all_candidates_agree' if ref_labels == {row['label']} else
                     'some_candidates_agree' if row['label'] in ref_labels else 'no_candidate_agrees')
        overview.append({'id': row['id'], 'local_split': row['split'], 'local_label': row['label'],
                         'method': method, 'byte_candidates': len(byte_matches), 'pixel_candidates': len(pixels),
                         'label_agreement': agreement})
        for ref in strongest:
            matches.append({'id': row['id'], 'local_split': row['split'], 'local_label': row['label'],
                            'method': method, 'reference_split': ref['split'],
                            'reference_path': ref['path'] if known else '',
                            'reference_path_sha256': hashlib.sha256(ref['path'].encode()).hexdigest(),
                            'reference_class': ref['label'] if known else '',
                            'label_agrees': str(ref['label'] == row['label']).lower() if known else '',
                            'note': '' if known else 'reference class/path withheld; not official ground truth'})
    result = {
        'reference_distribution': 'msambare/fer2013', 'reference_dataset_id': metadata.get('id'),
        'reference_version': metadata.get('currentVersionNumber'),
        'reference_dataset_page': provenance['dataset_page'],
        'reference_license_declaration': metadata.get('licenseName'),
        'reference_n_images': sum(ref_counts.values()), 'reference_class_split_counts': dict(ref_counts),
        'local_n_images': len(local), 'content_match_counts': dict(Counter(r['method'] for r in overview)),
        'labeled_agreement_counts': dict(Counter(r['label_agreement'] for r in overview if r['local_label'])),
        'official_unlabeled_rows': sum(r['local_split'] == 'predict' for r in overview),
        'unmatched_ids': [r['id'] for r in overview if r['method'] == 'unmatched'],
        'local_manifest_sha256': digest(manifest),
        'reference_archive_sha256': digest(reference / 'fer2013-msambare-v1.zip'),
        'reference_metadata_sha256': digest(reference / 'metadata.json'),
        'download_provenance_sha256': digest(reference / 'download_provenance.json'),
        'implementation_sha256': digest(Path(__file__)),
        'limitations': ['Content correspondence does not prove the academic packager downloaded from this uploader',
                        'Uploader license declaration does not independently establish original image rights',
                        'Reference labels are candidates, not authority for academic relabeling or official test truth',
                        'Official predict reference classes and class-bearing paths withheld; no test score computed',
                        'Only exact bytes/pixels are matched; no face identity inference or approximate attribution'],
    }
    output.mkdir(parents=True)
    for name, rows, fields in (
        ('image_overview.csv', overview, list(overview[0])),
        ('content_matches.csv', matches, ['id', 'local_split', 'local_label', 'method', 'reference_split',
                                          'reference_path', 'reference_path_sha256', 'reference_class', 'label_agrees', 'note']),
    ):
        with (output / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    result['output_sha256'] = {name: digest(output / name) for name in ('image_overview.csv', 'content_matches.csv')}
    write_json(output / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'source', 'reference', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    report = match(args.manifest, args.source, args.reference, args.output)
    print(json.dumps({k: v for k, v in report.items() if k not in ('reference_class_split_counts', 'limitations')}, indent=2))
