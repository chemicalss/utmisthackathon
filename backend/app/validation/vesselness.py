import numpy as np
import SimpleITK as sitk
from skimage.filters import frangi


def compute_vesselness(ct: sitk.Image, bbox, norm_percentile: float = 99.5) -> np.ndarray:
    """Frangi vesselness over just the narrow-band bounding box -- never
    the full volume, that's what keeps this inside the CPU time budget.

    Normalised against a high percentile of its own response rather than
    the exact max: a single outlier voxel (bone edge, metal artefact)
    can otherwise set the whole case's scale, which makes a fixed
    downstream threshold behave inconsistently from one case to the
    next. Percentile-based scaling keeps "1.0" meaning roughly the same
    thing -- strongly tubular -- across different volumes.
    """
    ct_arr = sitk.GetArrayFromImage(ct)
    roi = ct_arr[bbox].astype(np.float32)
    vesselness = frangi(roi, sigmas=range(1, 3), black_ridges=False)
    scale = np.percentile(vesselness, norm_percentile)
    return np.clip(vesselness / (scale + 1e-6), 0.0, 1.0)