from types import SimpleNamespace
from app.schemas.xai import MetricEvidence, MetricStatus
from app.voice_xai.evaluation.consistency import quality_for_analysis

def test_quality_uses_current_model_fidelity():
    metric = MetricEvidence(status=MetricStatus.available, value=0.93, scope="offline_validation", dataset_version="retrained-test")
    quality = quality_for_analysis(None, SimpleNamespace(fidelity_r2=metric))
    assert quality.surrogate_fidelity_r2.value == 0.93
    assert quality.temporal_semantic_iou.status == MetricStatus.not_applicable

def test_missing_model_metric_does_not_fabricate_r2():
    quality = quality_for_analysis(None, None)
    assert quality.surrogate_fidelity_r2.value is None
    assert quality.surrogate_fidelity_r2.status == MetricStatus.not_computed


def test_aopc_values_are_fixed_offline_references_not_current_upload_measurements():
    quality = quality_for_analysis(None, None)
    assert quality.semantic_aopc.value == 0.517231721941027
    assert quality.temporal_aopc.value == 0.1838843630005916
    for metric in (quality.semantic_aopc, quality.temporal_aopc):
        assert metric.scope == 'offline_validation'
        assert 'LA_D_2302632.wav' in metric.dataset_version
        assert 'not measured on the current upload' in metric.reason
