import argparse
import json

import cv2
from backend.skinvision.image_io import read_image

from backend.skinvision.metrics.lesion_metrics import extract_quant_features


def main():
    parser = argparse.ArgumentParser(description="Run lesion quant metrics demo")
    parser.add_argument("--image", required=True, help="Path to lesion image")
    parser.add_argument("--mask", required=True, help="Path to binary/probability mask image")
    args = parser.parse_args()

    image = read_image(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    mask = read_image(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {args.mask}")

    features = extract_quant_features(image, mask)
    print(json.dumps(features, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
