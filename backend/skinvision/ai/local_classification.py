"""Optional local 7-class module using a pinned public EfficientNet-B0 checkpoint.

Weights/config: adeIaide/skin-lesion-7class-efficientnet, CC BY-NC 4.0.
Only tensor data are loaded; no downloaded Python code is executed.
"""
import hashlib
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from PIL import Image
from starlette.concurrency import run_in_threadpool
import torch
from torchvision import models, transforms
from safetensors.torch import load_file

from backend.skinvision.ai.classification import LABELS, prepare_input

REPOSITORY = 'adeIaide/skin-lesion-7class-efficientnet'
REVISION = 'b5ebe73c3b89baa90ee4f991afa318385414a973'
VERSION = 'efficientnet-b0-320-v1'


def adapt_state_dict(state):
    """Map the author's wrapper to the equivalent stock torchvision network."""
    adapted = {}
    for key, value in state.items():
        if key.startswith('encoder.0.'):
            target = 'features.' + key[len('encoder.0.'):]
        elif key in {'classifier.weight', 'classifier.bias'}:
            target = key.replace('classifier.', 'classifier.1.', 1)
        else:
            raise ValueError('Unexpected public checkpoint key: ' + key)
        adapted[target] = value
    return adapted


class LocalClassifier:
    def __init__(self, directory, device='cpu'):
        self.directory = Path(directory) if directory else None
        self.device = torch.device(device)
        self.model, self.transform, self.config, self.weights_sha256 = None, None, None, None
        self.lock = threading.Lock()

    def _load(self):
        if self.model is not None:
            return
        config = json.loads((self.directory / 'config.json').read_text(encoding='utf-8'))
        if (config.get('backbone') != 'efficientnet_b0' or config.get('num_classes') != 7 or
                config.get('image_size') != 320 or config.get('input_mode') != 'RGB' or
                len(config.get('class_names', [])) != 7 or set(config['class_names']) != set(LABELS)):
            raise ValueError('Unsupported classifier configuration or class mapping')
        norm = config['normalization']
        if norm != {'mean':[.485,.456,.406], 'std':[.229,.224,.225]}:
            raise ValueError('Unexpected classifier normalization')
        path = self.directory / 'model.safetensors'
        digest = hashlib.sha256()
        with path.open('rb') as file:
            for chunk in iter(lambda: file.read(1024*1024), b''):
                digest.update(chunk)
        network = models.efficientnet_b0(weights=None, dropout=float(config['dropout']), num_classes=7)
        network.load_state_dict(adapt_state_dict(load_file(str(path))), strict=True)
        self.model = network.to(self.device).eval()
        self.transform = transforms.Compose([transforms.Resize((320,320)), transforms.ToTensor(),
                                             transforms.Normalize(norm['mean'], norm['std'])])
        self.config, self.weights_sha256 = config, digest.hexdigest()

    async def classify(self, image, mask=None, view='full'):
        return await run_in_threadpool(self._classify, image, mask, view)

    def _classify(self, image, mask, view):
        started = time.perf_counter()
        result = {'status':'not_configured', 'label':None, 'candidates':[],
                  'source':'Hugging Face / ' + REPOSITORY, 'model':'EfficientNet-B0',
                  'version':VERSION, 'revision':REVISION, 'license':'CC-BY-NC-4.0',
                  'prompt_version':None, 'request_settings':{'view':view, 'device':str(self.device)},
                  'score_semantics':'七分类 softmax 模型分数，未校准为患病概率',
                  'reason':'未配置本地分类权重', 'elapsed_ms':0.}
        if self.directory is None:
            return result
        try:
            try:
                image, info = prepare_input(image, mask, view)
            except ValueError as exc:
                result.update(status='review_required', reason=str(exc))
                return result
            result['input'] = info
            # A constant image has no spatial evidence. This narrow gate is not a
            # general skin detector and must not be described as OOD validation.
            if np.max(np.std(image.astype(np.float32), axis=(0,1))) < .01:
                result.update(status='review_required', reason='输入图像没有可用于分类的空间变化')
                return result
            result['input_sha256'] = hashlib.sha256(cv2.imencode('.png', image)[1].tobytes()).hexdigest()
            with self.lock:
                self._load()
                rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                batch = self.transform(rgb).unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    scores = self.model(batch).softmax(dim=1)[0].cpu().numpy()
                if scores.shape != (7,) or not np.isfinite(scores).all():
                    raise ValueError('Invalid classifier output')
                labels = self.config['class_names']
                candidates = [labels[int(index)] for index in np.argsort(-scores)[:3]]
                result.update(status='completed', label=candidates[0], candidates=candidates,
                              scores={label:float(score) for label,score in zip(labels,scores)},
                              weights_sha256=self.weights_sha256, preprocessing='RGB/PIL bilinear 320x320/ImageNet normalization',
                              reason='本地七分类模型候选；仅适用于研究用皮肤镜图像，不能确诊或识别所有图像类型')
        except Exception as exc:
            result.update(status='failed', label=None, candidates=[], reason='本地分类失败，需人工复核', error_type=type(exc).__name__)
        finally:
            result['elapsed_ms'] = (time.perf_counter() - started) * 1000
        return result
