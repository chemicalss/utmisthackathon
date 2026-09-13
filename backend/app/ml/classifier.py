import os
import joblib
from models.schemas import BranchRegion
from ml.model import extract_features

_MODEL_PATH = "ml/branch_classifier.joblib"

# Off by default -- 25 dev-set cases isn't enough to trust a trained
# verdict yet. Flip this to True (or set the env var) once there's a
# larger labeled dataset to train on. Until then, score_branches()
# is a no-op passthrough and the hand-coded validation gate is final.
ML_FILTER_ENABLED = os.environ.get("BRANCHSEED_ML_FILTER", "false").lower() == "true"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(_MODEL_PATH)
    return _model


def score_branches(regions: list[BranchRegion], min_confidence: float = 0.5) -> list[BranchRegion]:
    """Runs after validate_branches(), before tracing. When
    ML_FILTER_ENABLED is False (the current default), this returns
    regions unchanged -- the random forest never runs, so a missing or
    undertrained model file can't silently break the pipeline."""
    if not ML_FILTER_ENABLED:
        return regions

    if not regions:
        return []
    model = _get_model()
    features = [extract_features(r) for r in regions]
    confidences = model.predict_proba(features)[:, 1]
    return [r for r, conf in zip(regions, confidences) if conf >= min_confidence]