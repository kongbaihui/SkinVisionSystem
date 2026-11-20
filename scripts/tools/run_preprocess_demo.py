import argparse
import os

import cv2
from backend.skinvision.image_io import read_image, write_image

from backend.skinvision.preprocess.vision_preprocess import preprocess_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run CLAHE+blackhat+inpaint preprocess demo")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--outdir", default="outputs/preprocess_demo", help="Output directory")
    args = parser.parse_args()

    image = read_image(args.input)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")

    result = preprocess_pipeline(image)
    os.makedirs(args.outdir, exist_ok=True)
    write_image(os.path.join(args.outdir, "01_enhanced.png"), result["enhanced"])
    write_image(os.path.join(args.outdir, "02_hair_mask.png"), result["hair_mask"])
    write_image(os.path.join(args.outdir, "03_repaired.png"), result["repaired"])
    write_image(os.path.join(args.outdir, "04_resized.png"), result["resized"])
    print(f"Saved preprocess outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
