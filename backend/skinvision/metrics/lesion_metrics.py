import cv2
import numpy as np


def _to_binary_mask(mask, threshold=0.5):
	if mask.ndim == 3:
		mask = mask[..., 0]
	mask = mask.astype(np.float32)
	if mask.max() > 1.0:
		bin_mask = (mask >= 127.5).astype(np.uint8)
	else:
		bin_mask = (mask >= threshold).astype(np.uint8)
	return (bin_mask * 255).astype(np.uint8)


def compute_area_ratio(mask):
	bin_mask = _to_binary_mask(mask)
	lesion_area = float(np.count_nonzero(bin_mask))
	image_area = float(bin_mask.shape[0] * bin_mask.shape[1])
	if image_area == 0:
		return 0.0
	return lesion_area / image_area * 100.0


def compute_irregularity(mask):
	bin_mask = _to_binary_mask(mask)
	contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return 0.0
	contour = max(contours, key=cv2.contourArea)
	area = float(cv2.contourArea(contour))
	if area <= 1e-8:
		return 0.0
	perimeter = float(cv2.arcLength(contour, closed=True))
	return (perimeter * perimeter) / (4.0 * np.pi * area)


def compute_color_variance(image_bgr, mask):
	bin_mask = _to_binary_mask(mask)
	roi_idx = bin_mask > 0
	if not np.any(roi_idx):
		return 0.0
	rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
	roi_pixels = rgb[roi_idx]
	channel_var = np.var(roi_pixels, axis=0)
	return float(np.mean(channel_var))


def compute_asymmetry(mask):
	bin_mask = _to_binary_mask(mask)
	if not np.any(bin_mask):
		return 0.0
	flip_h = np.fliplr(bin_mask)
	inter_h = float(np.count_nonzero(bin_mask & flip_h))
	union_h = float(np.count_nonzero(bin_mask | flip_h))
	iou_h = inter_h / union_h if union_h > 0 else 1.0
	flip_v = np.flipud(bin_mask)
	inter_v = float(np.count_nonzero(bin_mask & flip_v))
	union_v = float(np.count_nonzero(bin_mask | flip_v))
	iou_v = inter_v / union_v if union_v > 0 else 1.0
	return float((2.0 - iou_h - iou_v) / 2.0)


def compute_border_sharpness(image_bgr, mask):
	bin_mask = _to_binary_mask(mask)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
	dilated = cv2.dilate(bin_mask, kernel, iterations=1)
	eroded = cv2.erode(bin_mask, kernel, iterations=1)
	border = dilated ^ eroded
	if not np.any(border):
		return 0.0
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
	grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
	grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
	magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
	return float(np.mean(magnitude[border > 0]))


def compute_color_diversity(image_bgr, mask):
	bin_mask = _to_binary_mask(mask)
	roi_idx = bin_mask > 0
	if not np.any(roi_idx):
		return 0.0
	rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
	channel_means = np.mean(rgb[roi_idx], axis=0)
	return float(np.var(channel_means))


def compute_max_diameter(mask):
	bin_mask = _to_binary_mask(mask)
	contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return 0.0
	contour = max(contours, key=cv2.contourArea)
	_, _, w, h = cv2.boundingRect(contour)
	img_h, img_w = bin_mask.shape[:2]
	diagonal = float(np.sqrt(img_h ** 2 + img_w ** 2))
	if diagonal < 1e-8:
		return 0.0
	return float(max(w, h)) / diagonal


def extract_quant_features(image_bgr, mask):
	return {
		"area_ratio": float(compute_area_ratio(mask)),
		"irregularity_index": float(compute_irregularity(mask)),
		"color_variance": float(compute_color_variance(image_bgr, mask)),
		"asymmetry_index": float(compute_asymmetry(mask)),
		"border_sharpness": float(compute_border_sharpness(image_bgr, mask)),
		"color_diversity": float(compute_color_diversity(image_bgr, mask)),
		"max_diameter_ratio": float(compute_max_diameter(mask)),
	}