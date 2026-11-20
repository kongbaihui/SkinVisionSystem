import logging
import copy
import logging.handlers
import math
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from matplotlib import pyplot as plt
from thop import profile


def set_seed(seed):
	os.environ["PYTHONHASHSEED"] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	cudnn.benchmark = False
	cudnn.deterministic = True


def get_logger(name, log_dir):
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)
	info_name = os.path.join(log_dir, f"{name}.info.log")
	info_handler = logging.handlers.TimedRotatingFileHandler(info_name, when="D", encoding="utf-8")
	info_handler.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
	info_handler.setFormatter(formatter)
	logger.addHandler(info_handler)
	return logger


def log_config_info(config, logger):
	logger.info("#----------Config info----------#")
	for key in dir(config):
		value = getattr(config, key)
		if not key.startswith("_") and not callable(value):
			logger.info(f"{key}: {value},")


def get_optimizer(config, model):
	return torch.optim.AdamW(
		model.parameters(),
		lr=config.lr,
		betas=config.betas,
		eps=config.eps,
		weight_decay=config.weight_decay,
		amsgrad=config.amsgrad,
	)


def get_scheduler(config, optimizer):
	return torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer,
		T_max=config.T_max,
		eta_min=config.eta_min,
		last_epoch=config.last_epoch,
	)


def save_imgs(img, msk, msk_pred, index, save_path, datasets, threshold=0.5, test_data_name=None):
	img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
	img = img / 255.0 if img.max() > 1.1 else img

	if datasets == "retinal":
		msk = np.squeeze(msk, axis=0)
		msk_pred = np.squeeze(msk_pred, axis=0)
	else:
		msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
		msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

	if test_data_name is not None:
		save_path = save_path + test_data_name + "_"

	for suffix, content, cmap in (
		("_original.png", img, None),
		("_mask.png", msk, "gray"),
		("_mask_pred.png", msk_pred, "gray"),
	):
		plt.figure()
		plt.imshow(content, cmap=cmap)
		plt.axis("off")
		plt.savefig(save_path + str(index) + suffix, bbox_inches="tight", pad_inches=0)
		plt.close()


class BCELoss(nn.Module):
	def __init__(self):
		super(BCELoss, self).__init__()
		self.bceloss = nn.BCELoss()

	def forward(self, pred, target):
		size = pred.size(0)
		pred_ = pred.view(size, -1)
		target_ = target.view(size, -1)
		return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()

	def forward(self, pred, target):
		smooth = 1
		size = pred.size(0)
		pred_ = pred.view(size, -1)
		target_ = target.view(size, -1)
		intersection = pred_ * target_
		dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
		return 1 - dice_score.sum() / size


class BceDiceLoss(nn.Module):
	def __init__(self, wb=1, wd=1):
		super(BceDiceLoss, self).__init__()
		self.bce = BCELoss()
		self.dice = DiceLoss()
		self.wb = wb
		self.wd = wd

	def forward(self, pred, target):
		return self.wd * self.dice(pred, target) + self.wb * self.bce(pred, target)


def cal_params_flops(model, size, logger):
	input_tensor = torch.randn(1, 3, size, size, device=next(model.parameters()).device)
	# THOP registers counters as buffers; do not contaminate saved checkpoints.
	flops, params = profile(copy.deepcopy(model), inputs=(input_tensor,))
	total = sum(parameter.numel() for parameter in model.parameters())
	print("flops", flops / 1e9)
	print("params", params / 1e6)
	print("Total params: %.3fM" % (total / 1e6))
	logger.info(f"flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}")
