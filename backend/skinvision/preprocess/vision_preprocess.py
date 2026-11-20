import cv2
import numpy as np


def apply_clahe_lab(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
	lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
	l_channel, a_channel, b_channel = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
	l_eq = clahe.apply(l_channel)
	merged = cv2.merge((l_eq, a_channel, b_channel))
	return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def detect_hair_mask_blackhat(image_bgr, kernel_size=17, threshold=10):
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
	blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
	_, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
	return mask


def repair_hair_with_inpaint(image_bgr, hair_mask, inpaint_radius=3, method="telea"):
	method_flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
	return cv2.inpaint(image_bgr, hair_mask, inpaint_radius, method_flag)


def normalize_to_chw_float(image_bgr, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
	rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
	mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
	std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
	norm = (rgb - mean_arr) / std_arr
	return np.transpose(norm, (2, 0, 1)).astype(np.float32)


def normalize_like_training(image_bgr):
	"""The loader's global standardization cancels in its per-image min/max step."""
	rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
	normalized = (rgb - rgb.min()) / max(float(rgb.max() - rgb.min()), 1e-8) * 255.
	return normalized.transpose(2, 0, 1).astype(np.float32)


def preprocess_pipeline(
	image_bgr,
	target_size=(256, 256),
	clahe_clip_limit=2.0,
	clahe_tile_grid_size=(8, 8),
	blackhat_kernel_size=17,
	blackhat_threshold=10,
	inpaint_radius=3,
	inpaint_method="telea",
	mean=(0.5, 0.5, 0.5),
	std=(0.5, 0.5, 0.5),
):
	enhanced = apply_clahe_lab(image_bgr, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size)
	hair_mask = detect_hair_mask_blackhat(enhanced, kernel_size=blackhat_kernel_size, threshold=blackhat_threshold)
	repaired = repair_hair_with_inpaint(enhanced, hair_mask, inpaint_radius=inpaint_radius, method=inpaint_method)
	resized = cv2.resize(repaired, target_size, interpolation=cv2.INTER_LINEAR)
	chw = normalize_to_chw_float(resized, mean=mean, std=std)
	return {
		"enhanced": enhanced,
		"hair_mask": hair_mask,
		"repaired": repaired,
		"resized": resized,
		"chw": chw,
	}
