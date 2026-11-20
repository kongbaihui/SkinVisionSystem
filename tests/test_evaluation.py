import numpy as np
import pytest
from backend.skinvision.evaluation.metrics import classification_metrics, segmentation_metrics, reliability_metrics
from backend.skinvision.evaluation.splits import grouped_split


def test_failed_classification_counts_in_recall_and_confusion():
    records = [{'truth':'NV','result':{'status':'completed','label':'NV'}},
               {'truth':'MEL','result':{'status':'failed','label':None}}]
    metrics = classification_metrics(records)
    assert metrics['accuracy'] == .5 and metrics['recall']['MEL'] == 0
    assert metrics['confusion_matrix'][1][7] == 1 and metrics['failed_or_review'] == 1


def test_segmentation_metrics_and_undefined_reliability_auc():
    truth = np.array([[1,0],[0,0]], dtype=bool)
    assert segmentation_metrics(truth.astype(float), truth)['dice'] == 1
    assert segmentation_metrics(np.zeros((2,2)),truth)['dice'] == 0
    metrics = reliability_metrics([{'status':'completed','mean_variance':.1,'dice':.9}])
    assert metrics['auroc_low_quality'] is None and metrics['low_quality_count'] == 0


def test_grouped_split_keeps_all_images_of_patient_together():
    rows = [{'id':str(i),'patient_id':'p'+str(i//3),'lesion_id':'l'+str(i)} for i in range(30)]
    result = grouped_split(rows)
    assert {row['id']:row['split'] for row in result} == {row['id']:row['split'] for row in grouped_split(list(reversed(rows)))}
    for patient in {row['patient_id'] for row in result}:
        assert len({row['split'] for row in result if row['patient_id']==patient}) == 1
    with pytest.raises(ValueError,match='Group ID required'):
        grouped_split([{'id':'unknown'}])


def test_partial_patient_metadata_still_links_same_lesion():
    rows = [{'id':'a','patient_id':'p1','lesion_id':'l1'},
            {'id':'b','lesion_id':'l1'}, {'id':'c','patient_id':'p1','lesion_id':'l2'}]
    result = grouped_split(rows)
    assert len({row['group_id'] for row in result}) == 1
