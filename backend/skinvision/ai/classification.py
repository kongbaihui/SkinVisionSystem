"""Independent seven-label classification, not inferred from report prose."""
import base64
import hashlib
import json
import time

import cv2
import httpx
import numpy as np

from backend.skinvision.llm_gateway import _DASHSCOPE_URL

LABELS = {'NV': '黑色素细胞痣', 'MEL': '黑色素瘤', 'BCC': '基底细胞癌',
          'AKIEC': '日光性角化病／鲍温病', 'BKL': '良性角化性病变',
          'DF': '皮肤纤维瘤', 'VASC': '血管性病变'}
PROMPT_VERSION = 'seven-class-v1.1'
PROMPT = '''你是课程研究中的皮肤镜图像七分类模块。仅输出一个 JSON 对象，不输出 Markdown 或患病概率。
候选编码固定为 NV、MEL、BCC、AKIEC、BKL、DF、VASC。
结构：{"status":"completed","label":"NV","candidates":["NV"],"reason":"简短的可见图像依据"}。
label 必须来自上述编码，candidates 必须包含 label，且最多三个不重复编码。
图像不适用、证据不足或无法分类时也必须提供 reason，完整格式为：
{"status":"review_required","label":null,"candidates":[],"reason":"图像缺少可用于皮肤病灶分类的证据"}。
不新增疾病类别、不编造检查结果、不输出 confidence。分类仅为候选辅助分析，不能确诊。'''


def parse_prediction(text):
    text = text.strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {'status', 'label', 'candidates', 'reason'}:
        raise ValueError('Invalid classification schema')
    if data['status'] not in {'completed', 'review_required'}:
        raise ValueError('Invalid classification status')
    candidates = data['candidates']
    if (not isinstance(candidates, list) or len(candidates) > 3 or
            any(not isinstance(x, str) or x not in LABELS for x in candidates) or
            len(candidates) != len(set(candidates))):
        raise ValueError('Invalid candidate labels')
    if data['status'] == 'completed':
        if not isinstance(data['label'], str) or data['label'] not in LABELS or data['label'] not in candidates:
            raise ValueError('Invalid predicted label')
    elif data['label'] is not None:
        raise ValueError('Review status must not assert a predicted label')
    if not isinstance(data['reason'], str) or not data['reason'].strip() or len(data['reason']) > 2000:
        raise ValueError('Missing or excessive classification evidence')
    return data


def prepare_input(image, mask=None, view='full'):
    if view not in {'full', 'roi'}:
        raise ValueError('view must be full or roi')
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError('Expected a nonempty BGR image')
    info = {'view': view, 'processing_version': 'bbox-padding10-v1', 'bbox_xyxy': None}
    if view == 'full':
        return image, info
    if mask is None or mask.shape != image.shape[:2]:
        raise ValueError('ROI requires an aligned segmentation mask')
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError('ROI mask is empty')
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    px, py = max(1, round((x1-x0)*.1)), max(1, round((y1-y0)*.1))
    x0, x1 = max(0, x0-px), min(image.shape[1], x1+px)
    y0, y1 = max(0, y0-py), min(image.shape[0], y1+py)
    info['bbox_xyxy'] = [x0, y0, x1, y1]
    return image[y0:y1, x0:x1].copy(), info


class Classifier:
    def __init__(self, api_key, model='qwen3-vl-plus', timeout=60., client=None):
        self.api_key, self.model, self.timeout, self.client = api_key, model, timeout, client

    async def classify(self, image, mask=None, view='full'):
        start = time.perf_counter()
        result = {'status': 'not_configured', 'label': None, 'candidates': [],
                  'reason': '未配置分类 API', 'source': 'DashScope multimodal API',
                  'model': self.model, 'prompt_version': PROMPT_VERSION,
                  'request_settings': {'temperature': 0.0, 'view': view},
                  'raw_response': None, 'elapsed_ms': 0., 'score_semantics': None}
        if not self.api_key:
            return result
        try:
            image, info = prepare_input(image, mask, view)
            result['input'] = info
        except ValueError as exc:
            result.update(status='review_required', reason=str(exc))
            return result
        # Bound payload dimensions, and encode pixels only (no image metadata).
        ratio = min(1., 1024 / max(image.shape[:2]))
        if ratio < 1:
            image = cv2.resize(image, (max(1, round(image.shape[1]*ratio)), max(1, round(image.shape[0]*ratio))))
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            result.update(status='failed', reason='图像编码失败')
            return result
        content = encoded.tobytes()
        result['input_sha256'] = hashlib.sha256(content).hexdigest()
        payload = {'model': self.model, 'input': {'messages': [
            {'role': 'system', 'content': [{'text': PROMPT}]},
            {'role': 'user', 'content': [{'image': 'data:image/png;base64,' + base64.b64encode(content).decode()},
                                        {'text': '请独立完成七类候选分类。'}]}]},
            'parameters': {'temperature': 0.0}}
        async def request(client):
            response = await client.post(_DASHSCOPE_URL, json=payload,
                                         headers={'Authorization': 'Bearer ' + self.api_key}, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data.get('code'):
                raise ValueError('Provider returned error code: ' + str(data['code']))
            text = ''.join(x.get('text', '') for x in data['output']['choices'][0]['message']['content'])
            result['raw_response'] = text
            result['request_id'] = data.get('request_id')
            result['response_model'] = data.get('model', self.model)
            result.update(parse_prediction(text))
        try:
            if self.client is not None:
                await request(self.client)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    await request(client)
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            # Do not persist HTTP headers, credentials or account identity.
            result.update(status='failed', label=None, candidates=[], reason='分类失败，需人工复核', error_type=type(exc).__name__)
        finally:
            result['elapsed_ms'] = (time.perf_counter() - start) * 1000
        return result
