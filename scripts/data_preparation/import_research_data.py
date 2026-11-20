"""Extract downloaded official ZIPs with containment and collision validation."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import zlib


def extract(archive_path, destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Unsafe archive entry')
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                checksum = 0
                with target.open('rb') as file:
                    for chunk in iter(lambda:file.read(1024*1024), b''):
                        checksum = zlib.crc32(chunk, checksum)
                if target.stat().st_size != item.file_size or checksum != item.CRC:
                    raise ValueError('Existing file differs; refusing to overwrite: ' + str(target))
                continue
            with archive.open(item) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('group', choices=['segmentation','classification'])
    args = parser.parse_args()
    manifest = json.loads((Path('data/downloads') / (args.group+'-manifest.json')).read_text())
    target = Path('data/research') / ('isic2016' if args.group == 'segmentation' else 'isic2018-task3')
    for record in manifest:
        source = Path(record['path'])
        if source.suffix == '.zip':
            extract(source, target)
            print('Extracted ' + source.name, flush=True)
        elif source.suffix == '.csv':
            target.mkdir(parents=True, exist_ok=True)
            output = target / source.name
            if output.exists() and output.read_bytes() != source.read_bytes():
                raise ValueError('Existing metadata differs')
            if not output.exists():
                shutil.copyfile(source, output)
    print('Imported ' + args.group)


if __name__ == '__main__':
    main()
