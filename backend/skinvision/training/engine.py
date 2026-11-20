import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm


def structure_loss(pred, mask):
	weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
	wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
	wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

	pred = torch.sigmoid(pred)
	inter = ((pred * mask) * weit).sum(dim=(2, 3))
	union = ((pred + mask) * weit).sum(dim=(2, 3))
	wiou = 1 - (inter + 1) / (union - inter + 1)
	return (wbce + wiou).mean()


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, logger, config, scaler=None):
	model.train()
	loss_list = []

	for iteration, data in enumerate(train_loader):
		optimizer.zero_grad()
		images, targets = data
		images = images.to(next(model.parameters()).device, non_blocking=True).float()
		targets = targets.to(next(model.parameters()).device, non_blocking=True).float()
		if config.amp and next(model.parameters()).is_cuda:
			with autocast():
				p1, p2, p3, p4 = model(images)
				loss = sum(structure_loss(prediction, targets) for prediction in (p1, p2, p3, p4))
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			p1, p2, p3, p4 = model(images)
			loss = sum(structure_loss(prediction, targets) for prediction in (p1, p2, p3, p4))
			loss.backward()
			optimizer.step()

		loss_list.append(loss.item())
		now_lr = optimizer.state_dict()["param_groups"][0]["lr"]
		if iteration % config.print_interval == 0:
			log_info = f"train: epoch {epoch}, iter:{iteration}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}"
			print(log_info)
			logger.info(log_info)
	scheduler.step()


def val_one_epoch(test_loader, model, criterion, epoch, logger, config):
	model.eval()
	preds, gts, loss_list = [], [], []
	with torch.no_grad():
		for data in tqdm(test_loader):
			img, msk = data
			img = img.to(next(model.parameters()).device, non_blocking=True).float()
			msk = msk.to(next(model.parameters()).device, non_blocking=True).float()
			p1, p2, p3, p4 = model(img)
			loss = sum(structure_loss(prediction, msk) for prediction in (p1, p2, p3, p4))
			loss_list.append(loss.item())
			gts.append(msk.squeeze(1).cpu().detach().numpy())
			preds.append(torch.stack([p.sigmoid() for p in (p1, p2, p3, p4)]).mean(0).squeeze(1).cpu().detach().numpy())

	if epoch % config.val_interval == 0:
		preds = np.array(preds).reshape(-1)
		gts = np.array(gts).reshape(-1)
		y_pre = np.where(preds >= config.threshold, 1, 0)
		y_true = np.where(gts >= 0.5, 1, 0)
		confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
		tn, fp, fn, tp = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
		accuracy = float(tn + tp) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
		sensitivity = float(tp) / float(tp + fn) if float(tp + fn) != 0 else 0
		specificity = float(tn) / float(tn + fp) if float(tn + fp) != 0 else 0
		f1_or_dsc = float(2 * tp) / float(2 * tp + fp + fn) if float(2 * tp + fp + fn) != 0 else 0
		miou = float(tp) / float(tp + fp + fn) if float(tp + fp + fn) != 0 else 0
		log_info = f"val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy},                 specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}"
	else:
		log_info = f"val epoch: {epoch}, loss: {np.mean(loss_list):.4f}"
	print(log_info)
	logger.info(log_info)
	return np.mean(loss_list)


def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None):
	model.eval()
	preds, gts, loss_list = [], [], []
	with torch.no_grad():
		for _, data in enumerate(tqdm(test_loader)):
			img, msk = data
			img = img.to(next(model.parameters()).device, non_blocking=True).float()
			msk = msk.to(next(model.parameters()).device, non_blocking=True).float()
			p1, p2, p3, p4 = model(img)
			loss = sum(structure_loss(prediction, msk) for prediction in (p1, p2, p3, p4))
			loss_list.append(loss.item())
			gts.append(msk.squeeze(1).cpu().detach().numpy())
			preds.append(torch.stack([p.sigmoid() for p in (p1, p2, p3, p4)]).mean(0).squeeze(1).cpu().detach().numpy())

		preds = np.array(preds).reshape(-1)
		gts = np.array(gts).reshape(-1)
		y_pre = np.where(preds >= config.threshold, 1, 0)
		y_true = np.where(gts >= 0.5, 1, 0)
		confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
		tn, fp, fn, tp = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
		accuracy = float(tn + tp) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
		sensitivity = float(tp) / float(tp + fn) if float(tp + fn) != 0 else 0
		specificity = float(tn) / float(tn + fp) if float(tn + fp) != 0 else 0
		f1_or_dsc = float(2 * tp) / float(2 * tp + fp + fn) if float(2 * tp + fp + fn) != 0 else 0
		miou = float(tp) / float(tp + fp + fn) if float(tp + fp + fn) != 0 else 0
		if test_data_name is not None:
			logger.info(f"test_datasets_name: {test_data_name}")
		log_info = f"test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy},                 specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}"
		print(log_info)
		logger.info(log_info)

	return np.mean(loss_list)