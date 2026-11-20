"""Portable, deterministic mask preparation with an auditable split manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from backend.skinvision.evaluation.splits import grouped_split, stable_order

LAYOUTS = {
    'isic2016': ('ISBI2016_ISIC_Part1_Training_Data', 'ISBI2016_ISIC_Part1_Training_GroundTruth',
                 'ISBI2016_ISIC_Part1_Test_Data', 'ISBI2016_ISIC_Part1_Test_GroundTruth', '_Segmentation.png', 788/900, 112/900),
    'isic2017': ('train/images', 'train/masks', 'val/images', 'val/masks', '_segmentation.png', .8, .2),
    'isic2018': ('ISIC2018_Task1-2_Training_Input', 'ISIC2018_Task1_Training_GroundTruth', None, None, '_segmentation.png', .72, .18),
    'ph2': ('images', 'masks', None, None, '.jpg', .7, .1),
}


def prepare(dataset, root, output, *, size=256, seed=20261119, groups=None):
    if size < 1:
        raise ValueError('size must be positive')
    root, output = Path(root).resolve(), Path(output).resolve()
    train_images, train_masks, test_images, test_masks, suffix, train_fraction, val_fraction = LAYOUTS[dataset]
    metadata = {}
    if groups:
        with Path(groups).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['image_id'] in metadata:
                    raise ValueError('Duplicate group metadata image_id')
                metadata[row['image_id']] = row
    records = []
    for image_folder, mask_folder, official in [(train_images, train_masks, 'train_pool'), (test_images, test_masks, 'test')]:
        if image_folder is None:
            continue
        images = sorted(p for p in (root/image_folder).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
        if not images:
            raise ValueError(f'No images in {root/image_folder}')
        for image in images:
            mask = root/mask_folder/(image.stem+suffix)
            if not mask.is_file():
                raise FileNotFoundError(mask)
            records.append({'id': image.stem, 'image': str(image), 'mask': str(mask), 'official_pool': official,
                            'patient_id': metadata.get(image.stem, {}).get('patient_id'),
                            'lesion_id': metadata.get(image.stem, {}).get('lesion_id')})
    if len({row['id'] for row in records}) != len(records):
        raise ValueError('Image IDs overlap across source pools')
    if groups:
        # Group across the entire inventory first, so official split conflicts cannot hide.
        # A supplied evaluation pool already owns all test samples. Avoid a
        # negative floating-point remainder for ratios such as 0.8 and 0.2.
        test_fraction = 0.0 if test_images else 1-train_fraction-val_fraction
        assigned = grouped_split(records, seed, (train_fraction, val_fraction, test_fraction))
        pools = {}
        for row in assigned:
            pools.setdefault(row['group_id'], set()).add(row['official_pool'])
        if any(len(values)>1 for values in pools.values()):
            raise ValueError('Patient/lesion crosses official train/test pools; resolve before preparation')
        records = assigned
        for row in records:
            if row['official_pool'] == 'test':
                row['split'] = 'test'
    else:
        pool = sorted((row for row in records if row['official_pool']=='train_pool'), key=lambda row: stable_order(row['id'], seed))
        n_train = int(len(pool)*train_fraction)
        n_val = len(pool)-n_train if test_images else int(len(pool)*val_fraction)
        for index, row in enumerate(pool):
            row['split'] = 'train' if index<n_train else 'val' if index<n_train+n_val else 'test'
        for row in records:
            if row['official_pool']=='test':
                row['split']='test'
    if any(not any(row['split']==split for row in records) for split in ('train','val','test')):
        raise ValueError('Empty split; supply more groups/images rather than inventing samples')
    # Validate all images before creating output; never overwrite a prior data preparation.
    for row in records:
        for key in ('image', 'mask'):
            with Image.open(row[key]) as im:
                im.verify()
            row[key+'_sha256'] = hashlib.sha256(Path(row[key]).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'val', 'test'):
        subset = sorted((row for row in records if row['split']==split), key=lambda row: stable_order(row['id'], seed))
        images = np.lib.format.open_memmap(output/f'data_{split}.npy', mode='w+', dtype='uint8', shape=(len(subset),size,size,3))
        masks = np.lib.format.open_memmap(output/f'mask_{split}.npy', mode='w+', dtype='uint8', shape=(len(subset),size,size))
        for index, row in enumerate(subset):
            with Image.open(row['image']) as im:
                images[index] = np.asarray(im.convert('RGB').resize((size,size), Image.Resampling.BILINEAR))
            with Image.open(row['mask']) as im:
                masks[index] = (np.asarray(im.convert('L').resize((size,size), Image.Resampling.NEAREST))>=128).astype('uint8')*255
            row['array_index'] = index
        images.flush(); masks.flush()
        del images, masks
    manifest = {'version':'segmentation-preparation-v1', 'dataset':dataset, 'seed':seed, 'size':size,
                'grouping':'patient/lesion connected groups' if groups else 'image-level; patient independence unverified',
                'source_pool_note':'isic2017 val directory remains legacy held-out evaluation pool' if dataset=='isic2017' else 'Supplied official pools preserved when available',
                'mask_resize':'nearest then binary >=128', 'dtype':'uint8', 'records':records}
    (output/'split-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


def main(dataset=None):
    parser=argparse.ArgumentParser(description=__doc__)
    if dataset is None:
        parser.add_argument('--dataset', required=True, choices=LAYOUTS)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20261119)
    parser.add_argument('--groups', type=Path, help='CSV: image_id,patient_id,lesion_id. All images need at least one group ID.')
    args=parser.parse_args()
    result=prepare(dataset or args.dataset,args.root,args.output,size=args.size,seed=args.seed,groups=args.groups)
    print(json.dumps({split:sum(row['split']==split for row in result['records']) for split in ('train','val','test')}))


if __name__=='__main__':
    main()
