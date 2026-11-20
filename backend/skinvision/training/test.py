import os
import sys
import warnings

from backend.skinvision.training.options import configure, select_device
from backend.skinvision.checkpoints import load_model_weights
import torch
from torch.utils.data import DataLoader

from backend.skinvision.config.config_setting import setting_config
from backend.skinvision.models.networks import LMDAUNet
from backend.skinvision.training.engine import structure_loss, test_one_epoch
from backend.skinvision.training.loader import isic_loader
from backend.skinvision.training.utils import get_logger, get_optimizer, get_scheduler, log_config_info, set_seed

os.environ["CUDA_VISIBLE_DEVICES"] = setting_config.gpu_id
warnings.filterwarnings("ignore")


def resolve_resume_model(config):
    if getattr(config, "test_weights", ""):
        candidate = config.test_weights
        if not os.path.isabs(candidate):
            candidate = os.path.join(os.getcwd(), candidate)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"test_weights not found: {candidate}")

    ckpt_candidates = []
    for root, _, files in os.walk("outputs/training"):
        if os.path.basename(root) != "checkpoints":
            continue
        best_files = [os.path.join(root, filename) for filename in files if filename.startswith("best-") and filename.endswith(".pth")]
        latest_files = [os.path.join(root, filename) for filename in files if filename == "latest.pth"]
        ckpt_candidates.extend(best_files if best_files else latest_files)

    if not ckpt_candidates:
        raise FileNotFoundError("No checkpoint found under outputs/training/**/checkpoints. Supply --test-weights for another location.")

    return max(ckpt_candidates, key=os.path.getmtime)


def main(config=setting_config) -> None:
    device = select_device(config)
    sys.path.append(config.work_dir + "/")
    log_dir = os.path.join(config.work_dir, "log")
    checkpoint_dir = os.path.join(config.work_dir, "checkpoints")
    resume_model = resolve_resume_model(config)
    outputs = os.path.join(config.work_dir, "outputs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    global logger
    logger = get_logger("test", log_dir)
    log_config_info(config, logger)
    set_seed(config.seed)
    torch.cuda.empty_cache()

    model = LMDAUNet(**config.model_config).to(device)
    test_loader = DataLoader(isic_loader(path_Data=config.data_path, train=False, Test=True), batch_size=1, shuffle=False, pin_memory=device.type == "cuda", num_workers=config.num_workers, drop_last=True)
    load_model_weights(model, resume_model, device)
    test_one_epoch(test_loader, model, structure_loss, logger, config)


if __name__ == "__main__":
    main(configure(setting_config))