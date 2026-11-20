"""Persist independent outputs without conflating model/API/template statuses."""
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from backend.skinvision.rule_cues import RISK_SOURCE, rule_level


def write_image(directory, filename, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError('Could not encode analysis image')
    (Path(directory) / filename).write_bytes(encoded.tobytes())
    return '/static/outputs/' + filename


def write_array(directory, filename, array):
    np.save(Path(directory) / filename, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return '/static/outputs/' + filename


def save_segmentation(task, result, directory, elapsed_ms):
    uid = task.request_id
    image, mask = result['preprocessed']['resized'], result['mask']
    tint = image.copy()
    tint[mask > 0] = (50, 180, 50)
    overlay = cv2.addWeighted(image, .65, tint, .35, 0)
    task.mask_path = write_image(directory, f'mask_{uid}.png', mask)
    task.roi_path = write_image(directory, f'roi_{uid}.png', image)
    task.overlay_path = write_image(directory, f'overlay_{uid}.png', overlay)
    task.probability_path = write_array(directory, f'prob_{uid}.npy', result['prob'])
    if result.get('gradcam_overlay') is not None:
        task.gradcam_path = write_image(directory, f'gradcam_{uid}.png', result['gradcam_overlay'])
    task.pipeline_mode = result['mode']
    task.inference_time_ms = elapsed_ms
    task.features_json = result['features']
    task.model_info_json = result.get('model_info')
    for key in ['area_ratio', 'irregularity_index', 'color_variance']:
        setattr(task, key, result['features'].get(key))
    # Overall status records a usable segmentation. Later API failures are separate.
    task.status = 'completed'
    task.updated_at = datetime.utcnow()


def save_reliability(directory, uid, metadata, variance, influence):
    metadata = dict(metadata)
    if variance is not None:
        metadata['variance_path'] = write_array(directory, f'variance_{uid}.npy', variance)
        # Fixed scale 0..0.25: variance of values bounded in [0,1], no per-image exaggeration.
        heatmap = cv2.applyColorMap(np.clip(variance / .25 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        metadata['variance_image_path'] = write_image(directory, f'variance_{uid}.png', heatmap)
        metadata['visualization_scale'] = [0., .25]
    if influence is not None:
        metadata['occlusion_path'] = write_array(directory, f'occlusion_{uid}.npy', influence)
        heatmap = cv2.applyColorMap(np.clip(influence * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        metadata['occlusion_image_path'] = write_image(directory, f'occlusion_{uid}.png', heatmap)
    return metadata


def task_features(task):
    features = dict(task.features_json or {})
    # Older tasks can have only the individual metric columns.
    for name in ('irregularity_index', 'area_ratio', 'color_variance'):
        if features.get(name) is None and getattr(task, name) is not None:
            features[name] = getattr(task, name)
    return features


def extra_fields(task):
    irregularity = task_features(task).get('irregularity_index')
    return {'classification': task.classification_json, 'reliability': task.reliability_json,
            'model_info': task.model_info_json, 'report_status': task.report_status,
            'report_metadata': task.report_metadata_json, 'probability_path': task.probability_path,
            'overlay_path': task.overlay_path, 'risk_level': rule_level(irregularity), 'risk_source': RISK_SOURCE,
            'report_version': task.report_version, 'original_path':
                f'/api/v1/tasks/{task.id}/original' if task.original_media_type else None}
