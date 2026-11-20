import argparse
import json
import os

import cv2
import numpy as np
import torch

from backend.skinvision.metrics.lesion_metrics import extract_quant_features
from backend.skinvision.models.networks import LMDAUNet
from backend.skinvision.checkpoints import load_model_weights
from backend.skinvision.preprocess.vision_preprocess import preprocess_pipeline, normalize_like_training
from backend.skinvision.image_io import read_image, write_image


def load_model(weights_path, device):
    model = LMDAUNet()
    load_model_weights(model, weights_path, device)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_mask(model, chw, device, threshold=0.5):
    x = torch.from_numpy(chw).unsqueeze(0).to(device)
    outputs = model(x, mode="test")
    prob_maps = [torch.sigmoid(output) for output in outputs]
    mean_prob = torch.mean(torch.stack(prob_maps, dim=0), dim=0)
    prob = mean_prob.squeeze().detach().cpu().numpy().astype(np.float32)
    mask = (prob >= threshold).astype(np.uint8) * 255
    return prob, mask


def main():
    parser = argparse.ArgumentParser(description="Single-image inference with quant outputs")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--weights", required=True, help="Path to model checkpoint .pth")
    parser.add_argument("--outdir", default="outputs/infer_single", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument('--normalization',choices=['rgb_minus1_1','train_minmax_255'],default='rgb_minus1_1')
    parser.add_argument('--threads',type=int,default=4)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.set_num_threads(args.threads)
    image = read_image(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    prep = preprocess_pipeline(image)
    model = load_model(args.weights, device)
    chw=prep['chw'] if args.normalization=='rgb_minus1_1' else normalize_like_training(prep['resized'])
    prob, mask = infer_mask(model, chw, device, threshold=args.threshold)
    metrics = extract_quant_features(prep["resized"], mask)

    os.makedirs(args.outdir, exist_ok=True)
    mask_path = os.path.join(args.outdir, "mask.png")
    prob_path = os.path.join(args.outdir, "prob.png")
    json_path = os.path.join(args.outdir, "features.json")
    write_image(mask_path, mask)
    write_image(prob_path, (prob * 255.0).clip(0, 255).astype(np.uint8))
    np.save(os.path.join(args.outdir,'probability.npy'),prob,allow_pickle=False)
    with open(os.path.join(args.outdir,'model-info.json'),'w',encoding='utf-8') as handle:
        json.dump({'normalization':args.normalization,'threshold':args.threshold,'weights':args.weights,'device':str(device)},handle)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"mask_path": mask_path, "prob_path": prob_path, "features": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
