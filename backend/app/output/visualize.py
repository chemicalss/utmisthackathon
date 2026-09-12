import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""Qualitative visual check for a case's predictions: three maximum-
intensity-projection panels (axial / coronal / sagittal) showing the
supplied aorta mask, every detected ostium, and its daughter-direction
arrow. This is for human sanity-checking, not a diagnostic viewer --
projections stay in voxel-index space so they stay correct regardless
of a volume's physical orientation."""

# (projection axis in the (z, y, x) array, title, x-label, y-label)
_VIEWS = [
    (0, "Axial (top-down)", "x (voxel)", "y (voxel)"),
    (1, "Coronal (front)", "x (voxel)", "z (voxel)"),
    (2, "Sagittal (side)", "y (voxel)", "z (voxel)"),
]


def _project_point(voxel_xyz, axis: int):
    x, y, z = voxel_xyz
    if axis == 0:
        return x, y
    if axis == 1:
        return x, z
    return y, z


def save_visual_check(ct: sitk.Image, aorta_mask: sitk.Image, result: dict,
                       out_path, window=(-100.0, 400.0), arrow_length_mm: float = 20.0) -> None:
    ct_arr = np.clip(sitk.GetArrayFromImage(ct).astype(np.float32), *window)
    mask_arr = sitk.GetArrayFromImage(aorta_mask).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (axis, title, xlabel, ylabel) in zip(axes, _VIEWS):
        ax.imshow(ct_arr.max(axis=axis), cmap="gray", origin="lower")
        ax.contour(mask_arr.max(axis=axis), levels=[0.5], colors="red", linewidths=1.0)

        for d in result["daughters"]:
            ostium_mm = np.array(d["ostium_xyz_mm"])
            direction = np.array(d["direction_xyz"])
            # The real seed is only 5 mm out -- too short to see at this
            # zoom level, so the arrow is drawn stretched to
            # arrow_length_mm along the same (already unit) direction.
            # This is a display-only exaggeration; nothing in the
            # underlying prediction is changed.
            arrow_tip_mm = ostium_mm + direction * arrow_length_mm
            ostium_idx = ct.TransformPhysicalPointToIndex(tuple(ostium_mm))
            tip_idx = ct.TransformPhysicalPointToIndex(tuple(arrow_tip_mm))
            ox, oy = _project_point(ostium_idx, axis)
            sx, sy = _project_point(tip_idx, axis)
            ax.plot(ox, oy, "o", color="yellow", markersize=6, markeredgecolor="black")
            ax.annotate("", xy=(sx, sy), xytext=(ox, oy),
                        arrowprops=dict(arrowstyle="->", color="cyan", linewidth=1.5))
            ax.annotate(d["instance_id"], (ox, oy), color="yellow", fontsize=7,
                        xytext=(3, 3), textcoords="offset points")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.suptitle(f"Case: {result['case_id']}  |  {len(result['daughters'])} daughter branch(es) detected  "
                 "(red = supplied aorta mask, yellow = ostium, cyan arrow = daughter direction)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
