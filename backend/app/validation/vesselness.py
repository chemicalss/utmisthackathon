import numpy as np
import SimpleITK as sitk
from skimage.filters import frangi


def compute_vesselness(ct: sitk.Image, bbox) -> np.ndarray:
    """Frangi vesselness over just the narrow-band bounding box -- never
    the full volume, that's what keeps this inside the CPU time budget."""
    ct_arr = sitk.GetArrayFromImage(ct)
    roi = ct_arr[bbox].astype(np.float32)
    vesselness = frangi(roi, sigmas=range(1, 3), black_ridges=False)
    v_min, v_max = vesselness.min(), vesselness.max()
    return (vesselness - v_min) / (v_max - v_min + 1e-6)