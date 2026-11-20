from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from scripts.data_preparation.prepare_segmentation import prepare


def test_portable_preparation_binary_masks_and_fixed_manifest(tmp_path):
    root=tmp_path/'中文 data'
    (root/'images').mkdir(parents=True); (root/'masks').mkdir()
    for index in range(10):
        Image.new('RGB',(9,7),(30+index,90,160)).save(root/'images'/f'im{index}.jpg')
        mask=np.zeros((7,9),dtype='uint8');mask[:,4:]=255
        Image.fromarray(mask).save(root/'masks'/f'im{index}.jpg')
    first=prepare('ph2',root,tmp_path/'one',size=16)
    second=prepare('ph2',root,tmp_path/'two',size=16)
    assert first==second
    assert [len(np.load(tmp_path/'one'/f'data_{split}.npy')) for split in ('train','val','test')]==[7,1,2]
    assert set(np.unique(np.load(tmp_path/'one'/'mask_train.npy')))=={0,255}
    assert len({row['id'] for row in first['records']})==10
    with pytest.raises(FileExistsError):
        prepare('ph2',root,tmp_path/'one',size=16)


def test_preparation_missing_pair_fails_without_output(tmp_path):
    (tmp_path/'images').mkdir();(tmp_path/'masks').mkdir()
    Image.new('RGB',(9,7)).save(tmp_path/'images'/'missing.jpg')
    with pytest.raises(FileNotFoundError):
        prepare('ph2',tmp_path,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_isic2017_grouped_split_preserves_official_pool(tmp_path):
    metadata = ['image_id,patient_id,lesion_id']
    for pool, count in [('train', 20), ('val', 4)]:
        image_dir = tmp_path/pool/'images'
        mask_dir = tmp_path/pool/'masks'
        image_dir.mkdir(parents=True)
        mask_dir.mkdir()
        for index in range(count):
            image_id = f'{pool}_{index}'
            Image.new('RGB', (8, 8), (40, 80, 120)).save(image_dir/f'{image_id}.jpg')
            Image.new('L', (8, 8), 255).save(mask_dir/f'{image_id}_segmentation.png')
            metadata.append(f'{image_id},patient_{image_id},lesion_{image_id}')
    groups = tmp_path/'groups.csv'
    groups.write_text('\n'.join(metadata), encoding='utf-8')
    result = prepare('isic2017', tmp_path, tmp_path/'prepared', size=8, groups=groups)
    assert {row['split'] for row in result['records']} == {'train', 'val', 'test'}
    assert all((row['split'] == 'test') == (row['official_pool'] == 'test')
               for row in result['records'])
    assert len(result['records']) == 24
