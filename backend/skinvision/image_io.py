"""OpenCV file I/O that also supports Unicode paths on Windows."""
from pathlib import Path
import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path,dtype=np.uint8),flags)


def write_image(path, image):
    ok, encoded=cv2.imencode(Path(path).suffix,image)
    if not ok:
        raise ValueError(f'Cannot encode image: {path}')
    encoded.tofile(path)
    return True
