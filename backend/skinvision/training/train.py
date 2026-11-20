import datetime
import os
import sys
import warnings

from backend.skinvision.training.options import configure, select_device
from backend.skinvision.checkpoints import load_checkpoint, load_model_state, load_model_weights
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from backend.skinvision.config.config_setting import setting_config
from backend.skinvision.models.networks import LMDAUNet
from backend.skinvision.training.engine import structure_loss, test_one_epoch, train_one_epoch, val_one_epoch
from backend.skinvision.training.loader import isic_loader
from backend.skinvision.training.utils import cal_params_flops, get_logger, get_optimizer, get_scheduler, log_config_info, set_seed

os.environ["CUDA_VISIBLE_DEVICES"] = setting_config.gpu_id
warnings.filterwarnings("ignore")


def main(config=setting_config) -> None:
    device = select_device(config)
    sys.path.append(config.work_dir + "/")
    log_dir = os.path.join(config.work_dir, "log")
    checkpoint_dir = os.path.join(config.work_dir, "checkpoints")
    resume_model = os.path.join(checkpoint_dir, "latest.pth")
    outputs = os.path.join(config.work_dir, "outputs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    global logger
    logger = get_logger("train", log_dir)
    log_config_info(config, logger)
    set_seed(config.seed)
    torch.cuda.empty_cache()

    train_loader = DataLoader(isic_loader(path_Data=config.data_path, train=True), batch_size=config.batch_size, shuffle=True, pin_memory=device.type == "cuda", num_workers=config.num_workers)
    val_loader = DataLoader(isic_loader(path_Data=config.data_path, train=False), batch_size=1, shuffle=False, pin_memory=device.type == "cuda", num_workers=config.num_workers, drop_last=True)
    test_loader = DataLoader(isic_loader(path_Data=config.data_path, train=False, Test=True), batch_size=1, shuffle=False, pin_memory=device.type == "cuda", num_workers=config.num_workers, drop_last=True)

    model = LMDAUNet(**config.model_config).to(device)
    if getattr(config, "profile", False):
        cal_params_flops(model, config.input_size_h, logger)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler(enabled=config.amp and device.type == "cuda")
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        checkpoint = load_checkpoint(resume_model, device)
        load_model_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch += checkpoint["epoch"]
        min_loss = checkpoint["min_loss"]
        min_epoch = checkpoint["min_epoch"]
        logger.info(f"resuming from {resume_model}. epoch: {checkpoint['epoch']}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {checkpoint['loss']:.4f}")

    print(datetime.datetime.now())
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        train_one_epoch(train_loader, model, structure_loss, optimizer, scheduler, epoch, logger, config, scaler=scaler)
        loss = val_one_epoch(val_loader, model, structure_loss, epoch, logger, config)
        print(datetime.datetime.now())
        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pth"))
            min_loss = loss
            min_epoch = epoch

        torch.save({
            "epoch": epoch,
            "min_loss": float(min_loss),
            "min_epoch": min_epoch,
            "loss": float(loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, "latest.pth"))

    best_path = os.path.join(checkpoint_dir, "best.pth")
    if os.path.exists(best_path):
        load_model_weights(model, best_path, device)
        test_one_epoch(test_loader, model, structure_loss, logger, config)
        os.rename(best_path, os.path.join(checkpoint_dir, f"best-epoch{min_epoch}-loss{min_loss:.4f}.pth"))


if __name__ == "__main__":
    main(configure(setting_config))
