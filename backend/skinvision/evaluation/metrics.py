"""Metrics count failed/review classifications; no fabricated ground truth."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score
from scipy.stats import spearmanr

LABEL_ORDER = ['NV','MEL','BCC','AKIEC','BKL','DF','VASC']


def segmentation_metrics(probability, truth, threshold=.5):
    probability, truth = np.asarray(probability), np.asarray(truth, dtype=bool)
    if probability.shape != truth.shape or not np.isfinite(probability).all():
        raise ValueError('Invalid prediction shape or values')
    pred = probability >= threshold
    intersection = int(np.logical_and(pred, truth).sum())
    union = int(np.logical_or(pred, truth).sum())
    denominator = int(pred.sum() + truth.sum())
    return {'dice':2*intersection/denominator if denominator else 1.,
            'iou':intersection/union if union else 1., 'predicted_pixels':int(pred.sum()),
            'truth_pixels':int(truth.sum()), 'threshold':threshold}


def classification_metrics(records):
    truth = [row['truth'] for row in records]
    if not truth or any(label not in LABEL_ORDER for label in truth):
        raise ValueError('Requires valid disease labels')
    predicted = [row['result'].get('label') if row['result'].get('status') == 'completed' else '__unavailable__'
                 for row in records]
    if any(label not in LABEL_ORDER + ['__unavailable__'] for label in predicted):
        raise ValueError('Invalid prediction label')
    matrix = confusion_matrix(truth, predicted, labels=LABEL_ORDER + ['__unavailable__'])[:7]
    valid = sum(label != '__unavailable__' for label in predicted)
    return {'n':len(records), 'completed':valid, 'failed_or_review':len(records)-valid,
            'coverage':valid/len(records),
            'macro_f1':float(f1_score(truth,predicted,labels=LABEL_ORDER,average='macro',zero_division=0)),
            'recall':dict(zip(LABEL_ORDER,recall_score(truth,predicted,labels=LABEL_ORDER,average=None,zero_division=0).tolist())),
            'support':{label:truth.count(label) for label in LABEL_ORDER},
            'confusion_matrix':matrix.tolist(), 'rows_true':LABEL_ORDER,
            'columns_predicted':LABEL_ORDER+['unavailable (processing state, not disease)'],
            'accuracy':float(np.mean(np.asarray(truth)==np.asarray(predicted))),
            'failure_policy':'Failed/review/not-configured predictions count as missed true classes in primary F1/recall'}


def reliability_metrics(records, low_quality_dice=.7):
    pairs = [row for row in records if row.get('status') == 'completed']
    result = {'n':len(records), 'completed':len(pairs), 'failed':len(records)-len(pairs),
              'low_quality_dice_threshold':low_quality_dice,
              'spearman_variance_vs_error':None,'auroc_low_quality':None}
    if not pairs:
        return result
    variance = np.asarray([row['mean_variance'] for row in pairs])
    error = np.asarray([1-row['dice'] for row in pairs])
    low_quality = (1-error < low_quality_dice).astype(int)
    result['low_quality_count'] = int(low_quality.sum())
    if len(pairs)>2 and np.ptp(variance)>0 and np.ptp(error)>0:
        correlation = spearmanr(variance,error)
        result['spearman_variance_vs_error'] = float(correlation.statistic)
        result['spearman_pvalue_descriptive'] = float(correlation.pvalue)
    if len(np.unique(low_quality)) == 2:
        result['auroc_low_quality'] = float(roc_auc_score(low_quality,variance))
    else:
        result['auroc_unavailable_reason'] = 'Only one segmentation quality class observed'
    return result
