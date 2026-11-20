import os
import hashlib
from pathlib import Path
import threading

import cv2
import numpy as np
import torch

from backend.skinvision.checkpoints import load_model_weights
from backend.skinvision.metrics.lesion_metrics import extract_quant_features
from backend.skinvision.models.networks import LMDAUNet
from backend.skinvision.preprocess.vision_preprocess import preprocess_pipeline, normalize_like_training


class SkinVisionPipeline:
	"""预处理、分割与量化指标提取流程。"""

	def __init__(self, weights_path=None, threshold=0.5, use_cpu=False, normalization='rgb_minus1_1'):
		if normalization not in {'rgb_minus1_1', 'train_minmax_255'}:
			raise ValueError('Unsupported segmentation normalization')
		self.normalization = normalization
		self.weights_path = weights_path
		self.threshold = threshold
		self.device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
		self.model = None
		self.mode = "unloaded" if self._get_weights_signature() is not None else "fallback"
		self.load_error = None
		self._model_lock = threading.Lock()
		self._inference_lock = threading.RLock()
		self.weights_sha256 = None
		self._last_load_signature = None

	def _get_weights_signature(self):
		if not self.weights_path or not os.path.exists(self.weights_path):
			return None
		stat = os.stat(self.weights_path)
		return (self.weights_path, stat.st_mtime_ns, stat.st_size)

	def _try_load_model(self, signature):
		if signature is None:
			self.model = None
			self.mode = "fallback"
			self.load_error = None
			self._last_load_signature = None
			return False
		try:
			model = LMDAUNet()
			load_model_weights(model, self.weights_path, self.device)
			model.to(self.device)
			model.eval()
			self.model = model
			with open(self.weights_path, "rb") as file:
				digest = hashlib.sha256()
				for chunk in iter(lambda: file.read(1024 * 1024), b""):
					digest.update(chunk)
				self.weights_sha256 = digest.hexdigest()
			self.mode = "model"
			self.load_error = None
			self._last_load_signature = signature
			return True
		except Exception as exc:
			self.model = None
			self.mode = "fallback"
			self.load_error = str(exc)
			self._last_load_signature = signature
			return False

	def ensure_model_loaded(self):
		signature = self._get_weights_signature()
		if signature is None:
			self.model = None
			self.mode = "fallback"
			self.load_error = None
			return False
		if self.model is not None and self._last_load_signature == signature:
			return True
		if self.model is None and self._last_load_signature == signature and self.load_error:
			return False
		with self._model_lock:
			signature = self._get_weights_signature()
			if signature is None:
				self.model = None
				self.mode = "fallback"
				self.load_error = None
				self._last_load_signature = None
				return False
			if self.model is not None and self._last_load_signature == signature:
				return True
			return self._try_load_model(signature)

	@torch.no_grad()
	def _infer_by_model(self, chw):
		x = torch.from_numpy(chw).unsqueeze(0).to(self.device)
		outputs = self.model(x, mode="test")
		prob_maps = [torch.sigmoid(o) for o in outputs]
		mean_prob = torch.mean(torch.stack(prob_maps, dim=0), dim=0)
		prob = mean_prob.squeeze().detach().cpu().numpy().astype(np.float32)
		mask = (prob >= self.threshold).astype(np.uint8) * 255
		return prob, mask

	def _generate_gradcam(self, chw, resized_bgr, return_map=False):
		# The score depends on the selected activation only through out_head1.
		# Recompute that head to get the exact activation gradient without retaining
		# the entire recurrent backbone graph or accumulating parameter gradients.
		saved_feats = []
		handle = None
		try:
			def capture(module, inp, out):
				saved_feats.append(inp[0].detach())
			handle = self.model.out_head1.register_forward_hook(capture)
			x = torch.from_numpy(chw).unsqueeze(0).float().to(self.device)
			with torch.no_grad():
				outputs = self.model(x, mode="test")
			handle.remove()
			handle = None
			if not saved_feats:
				return (None, None) if return_map else None
			with torch.enable_grad():
				feat = saved_feats[0].requires_grad_(True)
				logits = self.model.out_head1(feat)
				logits = torch.nn.functional.interpolate(logits, size=outputs[0].shape[-2:], mode="bilinear", align_corners=False)
				score = logits.sigmoid().mean()
				grad, = torch.autograd.grad(score, feat)
				weights = grad.mean(dim=(2, 3), keepdim=True)
				cam = torch.relu((weights * feat).sum(dim=1)).squeeze().detach().cpu().numpy()
			if cam.max() > 1e-8:
				cam = cam / cam.max()
			target_h, target_w = resized_bgr.shape[:2]
			cam_resized = cv2.resize(cam.astype(np.float32), (target_w, target_h))
			heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
			overlay = cv2.addWeighted(resized_bgr, 0.55, heatmap, 0.45, 0)
			return (cam_resized, overlay) if return_map else overlay
		except Exception:
			return (None, None) if return_map else None
		finally:
			if handle is not None:
				handle.remove()

	def _infer_by_fallback(self, image_bgr):
		gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (5, 5), 0)
		_, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		prob = mask.astype(np.float32) / 255.0
		return prob, mask

	def run(self, image_bgr):
		with self._inference_lock:
			return self._run_locked(image_bgr)

	def _run_locked(self, image_bgr):
		prep = preprocess_pipeline(image_bgr)
		if self.normalization == 'train_minmax_255':
			prep['chw'] = normalize_like_training(prep['resized'])
		if self.ensure_model_loaded():
			prob, mask = self._infer_by_model(prep["chw"])
			gradcam_overlay = self._generate_gradcam(prep["chw"], prep["resized"])
		else:
			prob, mask = self._infer_by_fallback(prep["resized"])
			gradcam_overlay = None
		metrics = extract_quant_features(prep["resized"], mask)
		return {
			"mode": self.mode,
			"preprocessed": prep,
			"prob": prob,
			"mask": mask,
			"features": metrics,
			"gradcam_overlay": gradcam_overlay,
			"model_info": {"name": "LMDAUNet" if self.mode == "model" else "Otsu",
			               "execution": self.mode, "device": str(self.device), "load_error": self.load_error,
			               "version": "lmdaunet-portable-v1", "torch_version": torch.__version__,
			               "normalization": self.normalization, "threshold": self.threshold,
			               "preprocessing": "CLAHE/hair-inpaint/RGB/resize256",
			               "weights_sha256": self.weights_sha256 if self.mode == "model" else None,
			               "weights_file": Path(self.weights_path).name if self.weights_path else None,
			               "training_dataset": "unverified", "source": "existing project LMDAUNet / local checkpoint",
			               "explanation_status": "completed" if gradcam_overlay is not None else ("failed" if self.mode == "model" else "not_available")},
		}
	def evaluate_reliability(self, result, enabled=True, include_occlusion=False):
		from backend.skinvision.ai.reliability import analyze_reliability, occlusion_sensitivity
		if not enabled:
			return {"status": "not_evaluated", "reason": "本次未请求可靠性评估"}, None, None
		if result["mode"] != "model":
			return {"status": "not_available", "reason": "真实模型未加载；不对传统降级结果宣称模型可靠性"}, None, None
		with self._inference_lock:
			try:
				if self.model is None or result["model_info"]["weights_sha256"] != self.weights_sha256:
					raise ValueError("Model changed since segmentation")
				predict = lambda x: self._infer_by_model(x)[0]
				metadata, variance = analyze_reliability(predict, result["preprocessed"]["chw"], reference=result["prob"])
				influence = None
				if include_occlusion:
					metadata["occlusion"], influence = occlusion_sensitivity(predict, result["preprocessed"]["chw"], grid=2, reference=result["prob"])
				metadata["weights_sha256"] = self.weights_sha256
				return metadata, variance, influence
			except Exception as exc:
				return {"status": "failed", "reason": "可靠性计算失败", "error_type": type(exc).__name__}, None, None
