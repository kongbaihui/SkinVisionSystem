"""Freeze IDs and hashes before observing any experimental predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from backend.skinvision.evaluation.splits import stable_order
from backend.skinvision.evaluation.metrics import LABEL_ORDER


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/research-v1/protocol-v1.json'))
    destination = parser.parse_args().output
    if destination.exists():
        raise FileExistsError('Protocol already frozen')
    segmentation = []
    root = Path('data/research/isic2016')
    for source_split, role, count in [('Training','calibration',8),('Test','test',24)]:
        paths = sorted((root/f'ISBI2016_ISIC_Part1_{source_split}_Data').glob('*.jpg'),key=lambda p:stable_order(p.stem))[:count]
        if len(paths)!=count:
            raise ValueError('Incomplete segmentation data')
        for image in paths:
            mask = root/f'ISBI2016_ISIC_Part1_{source_split}_GroundTruth'/f'{image.stem}_Segmentation.png'
            segmentation.append({'id':image.stem,'image':image.as_posix(),'mask':mask.as_posix(),
                                 'image_sha256':digest(image),'mask_sha256':digest(mask),'split':role,
                                 'source_split':source_split,'group_information':'not provided in original archive'})
    root = Path('data/research/isic2018-task3')
    csv_path = root/'ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv'
    with csv_path.open(encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    groups = {label:[] for label in LABEL_ORDER}
    for row in rows:
        positive = [label for label in LABEL_ORDER if float(row[label])==1]
        if len(positive)!=1:
            raise ValueError('Disease ground truth must be one-hot')
        groups[positive[0]].append(row['image'])
    classification = []
    for label, ids in groups.items():
        for image_id in sorted(ids,key=stable_order)[:10]:
            image = root/'ISIC2018_Task3_Test_Input'/f'{image_id}.jpg'
            classification.append({'id':image_id,'image':image.as_posix(),'image_sha256':digest(image),
                                   'truth':label,'split':'official_test','group_information':'not supplied for official test'})
    if len(classification)!=70:
        raise ValueError('Insufficient cases in a class')
    protocol = {'version':'research-v1','seed':20261119,'selection':'smallest SHA256(seed:image_id); 10 per class for balanced classification subset',
        'segmentation':segmentation,'classification':classification,
        'segmentation_comparison':['rgb_minus1_1','train_minmax_255'],
        'controls':'Same old weights, CLAHE, inpaint, RGB, 256x256, four-head mean sigmoid, threshold0.5; only numerical normalization varies',
        'selection_rule':'Choose normalization by mean calibration Dice only, ties retain original; test both fixed conditions',
        'classification_comparison':['full','roi'],'classification_model':'adeIaide EfficientNet-B0 pinned b5ebe73c3b89baa90ee4f991afa318385414a973',
        'reliability':{'cases':'same 24 segmentation test IDs','low_quality_dice_threshold':.7,'tta':['identity','hflip','vflip','rotate-5','rotate+5'],
                       'occlusion':'first 8 fixed test IDs; 2x2 grid, normalized fill0'},
        'report':{'conditions':['text_only','image_text','constrained'],'cases':'first 6 fixed segmentation test IDs plus 2 explicit fault-state cases',
                  'technical_review':'human-readable claim audit with referenced numeric facts; no clinical accuracy claim'},
        'limitations':['Small fixed subsets; report exact n/support, not full benchmark performance',
                       'Old segmentation training membership unknown; not an independent generalization claim',
                       'Official splits retained; missing patient/lesion IDs cannot establish patient independence',
                       'Community classifier training membership cannot be independently audited']}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'segmentation':len(segmentation),'classification':len(classification),'protocol_sha256':digest(destination)}))


if __name__ == '__main__':
    main()
