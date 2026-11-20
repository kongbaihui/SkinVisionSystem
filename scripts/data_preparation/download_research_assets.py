"""Download official research inputs with resumable streams and SHA256 manifests.

No downloaded code is executed. Archives are kept intact for later audited import.
ISIC 2016: CC0. ISIC 2018 Task3 and classifier: CC BY-NC 4.0.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile
import httpx

REVISION = 'b5ebe73c3b89baa90ee4f991afa318385414a973'
MODEL_ROOT = f'https://huggingface.co/adeIaide/skin-lesion-7class-efficientnet/resolve/{REVISION}/'
DATA_ROOT = 'https://isic-archive.s3.amazonaws.com/challenges/'
CLASSIFIER_SHA256 = {
    'model.safetensors': 'fed0c3cb0a15823f4f6a82d4dd2725a0b8e6068a971664e1504403fba5b54562',
    'config.json': '3b09f5eb750b0d458f208b4c7c46e3babe58a46106f3fb0e9eac8d3a1d88f3f8',
    'README.md': '83a3b04334a3c1ee27310ba7c111ca66feb5fd76f4b5b8a546a9ae68af2a2738',
}
ASSETS = {
    'classifier': [('weights/classification/efficientnet-b0-research/' + name, MODEL_ROOT + name, 'CC-BY-NC-4.0')
                   for name in ['model.safetensors', 'config.json', 'README.md']],
    'segmentation': [('data/downloads/' + name, DATA_ROOT + '2016/' + name, 'CC0') for name in
        ['ISBI2016_ISIC_Part1_Training_Data.zip', 'ISBI2016_ISIC_Part1_Training_GroundTruth.zip',
         'ISBI2016_ISIC_Part1_Test_Data.zip', 'ISBI2016_ISIC_Part1_Test_GroundTruth.zip']],
    'classification': [('data/downloads/' + name, DATA_ROOT + '2018/' + name, 'CC-BY-NC-4.0') for name in
        ['ISIC2018_Task3_Training_LesionGroupings.csv', 'ISIC2018_Task3_Training_GroundTruth.zip',
         'ISIC2018_Task3_Validation_Input.zip', 'ISIC2018_Task3_Validation_GroundTruth.zip',
         'ISIC2018_Task3_Test_Input.zip', 'ISIC2018_Task3_Test_GroundTruth.zip']],
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(path, url):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    if path.exists():
        return
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with httpx.Client(follow_redirects=True, timeout=90) as client:
                with client.stream('GET', url, headers={'Range':f'bytes={offset}-'} if offset else {}) as response:
                    response.raise_for_status()
                    append = offset > 0 and response.status_code == 206
                    if append and not response.headers.get('content-range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected range response')
                    with partial.open('ab' if append else 'wb') as file:
                        for chunk in response.iter_bytes(1024*1024):
                            file.write(chunk)
            partial.replace(path)
            return
        except (httpx.HTTPError, OSError):
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('group', choices=ASSETS)
    args = parser.parse_args()
    manifest = Path('data/downloads') / f'{args.group}-manifest.json'
    records = []
    for filename, url, license_name in ASSETS[args.group]:
        path = Path(filename)
        print('Downloading ' + path.name, flush=True)
        download(path, url)
        if path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
                if bad:
                    raise ValueError('Archive CRC check failed: ' + bad)
        digest = sha256(path)
        if args.group == 'classifier' and digest != CLASSIFIER_SHA256[path.name]:
            raise ValueError(f'{path} does not match the pinned revision {REVISION}')
        records.append({'path':filename, 'url':url, 'bytes':path.stat().st_size,
                        'sha256':digest, 'license':license_name})
        print('Verified ' + path.name + ': ' + str(path.stat().st_size) + ' bytes', flush=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
