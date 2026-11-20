"""Prediction stability and occlusion effects, not disease-risk estimates."""
import time
import cv2
import numpy as np

VERSION = 'tta-flip-rotate-v1'


def _predict(predict, image):
    output = np.asarray(predict(np.ascontiguousarray(image)), dtype=np.float32)
    if output.shape != image.shape[1:] or not np.isfinite(output).all():
        raise ValueError('Prediction must be finite with matching spatial dimensions')
    if np.any(output < 0) or np.any(output > 1):
        raise ValueError('Prediction must contain probabilities in [0,1]')
    return output


def analyze_reliability(predict, chw, rotations=(-5., 5.), reference=None):
    """Return (JSON metadata, variance map) after coordinate restoration.

    `predict` must run the real segmentation model and return HxW probabilities.
    The caller must skip this function when only traditional fallback is loaded.
    Rotation pixels influenced by zero padding are excluded from that transform.
    """
    start = time.perf_counter()
    base = _predict(predict, chw) if reference is None else _predict(lambda _: reference, chw)
    height, width = base.shape
    predictions, valid = [base], [np.ones_like(base)]
    transforms = ['identity']
    for axis, name in [(2, 'horizontal_flip'), (1, 'vertical_flip')]:
        predictions.append(np.flip(_predict(predict, np.flip(chw, axis=axis)), axis=axis-1))
        valid.append(np.ones_like(base))
        transforms.append(name)
    for angle in rotations:
        matrix = cv2.getRotationMatrix2D(((width-1)/2, (height-1)/2), float(angle), 1.)
        inverse = cv2.invertAffineTransform(matrix)
        transformed = cv2.warpAffine(chw.transpose(1,2,0), matrix, (width,height), flags=cv2.INTER_LINEAR)
        predicted = _predict(predict, transformed.transpose(2,0,1))
        restored = cv2.warpAffine(predicted, inverse, (width,height), flags=cv2.INTER_LINEAR)
        coverage = cv2.warpAffine(np.ones_like(base), matrix, (width,height), flags=cv2.INTER_LINEAR)
        coverage = cv2.warpAffine(coverage, inverse, (width,height), flags=cv2.INTER_LINEAR)
        predictions.append(restored)
        valid.append((coverage >= .9999).astype(np.float32))
        transforms.append(f'rotation_{angle:g}_degrees')
    stack, weights = np.stack(predictions), np.stack(valid)
    count = weights.sum(axis=0)
    mean = (stack * weights).sum(axis=0) / count
    variance = (((stack-mean)**2) * weights).sum(axis=0) / count
    return {'status': 'completed', 'version': VERSION, 'transforms': transforms,
            'prediction_count': len(predictions), 'extra_prediction_count': len(predictions)-1,
            'elapsed_ms': (time.perf_counter()-start)*1000,
            'mean_variance': float(variance.mean()), 'p95_variance': float(np.percentile(variance,95)),
            'fully_covered_fraction': float(np.mean(count == len(predictions))),
            'semantics': '测试时变换后的预测方差，尚未校准为分割质量概率；不表示疾病危险程度'}, variance


def occlusion_sensitivity(predict, chw, grid=4, fill=0., reference=None):
    """Mask each cell and measure global mean absolute prediction change."""
    if not isinstance(grid, int) or not 1 <= grid <= min(chw.shape[1:]):
        raise ValueError('Invalid occlusion grid')
    start = time.perf_counter()
    base = _predict(predict, chw) if reference is None else _predict(lambda _: reference, chw)
    influence = np.zeros_like(base)
    ys, xs = np.linspace(0,base.shape[0],grid+1,dtype=int), np.linspace(0,base.shape[1],grid+1,dtype=int)
    effects = []
    for row in range(grid):
        for col in range(grid):
            modified = chw.copy()
            y0,y1,x0,x1 = ys[row],ys[row+1],xs[col],xs[col+1]
            modified[:,y0:y1,x0:x1] = fill
            delta = float(np.abs(_predict(predict,modified)-base).mean())
            influence[y0:y1,x0:x1] = delta
            effects.append({'bbox_xyxy':[int(x0),int(y0),int(x1),int(y1)],'mean_probability_change':delta})
    return {'status':'completed','version':'occlusion-v1','grid':grid,'fill':fill,
            'prediction_count':grid*grid+1,'extra_prediction_count':grid*grid,
            'elapsed_ms':(time.perf_counter()-start)*1000,'effects':effects,
            'semantics':'遮挡后模型预测变化；用于解释研究，不是疾病风险'}, influence
