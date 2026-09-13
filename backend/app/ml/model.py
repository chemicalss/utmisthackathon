import numpy as np
from models.schemas import BranchRegion

FEATURE_NAMES = ["persisted_mm", "direction_ok", "mean_vesselness",
                  "region_connected", "geodesic_cost"]


def extract_features(region: BranchRegion) -> np.ndarray:
    """One row per candidate. Order must match FEATURE_NAMES exactly --
    train.py and classifier.py both call this, so they can never drift
    out of sync with each other."""
    return np.array([
        region.persisted_mm,
        float(region.direction_ok),
        region.mean_vesselness,
        float(region.region_connected),
        region.geodesic_cost,
    ])