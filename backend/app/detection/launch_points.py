import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from models.schemas import LaunchPoint


def find_launch_points(ct: sitk.Image, aorta_mask: sitk.Image,
                        shell_out_mm: float = 3.0,
                        min_cluster_voxels: int = 3,
                        crop_margin_slices: int = 2) -> list[LaunchPoint]:
    ct_arr = sitk.GetArrayFromImage(ct)                 # z, y, x
    mask_arr = sitk.GetArrayFromImage(aorta_mask).astype(bool)
    spacing = np.array(ct.GetSpacing())[::-1]           # -> z, y, x order to match the array

    # 1. surface shell = mask minus its erosion
    eroded = ndimage.binary_erosion(mask_arr, iterations=1)
    surface = mask_arr & ~eroded

    # 2. drop the flat crop faces at the top/bottom of the supplied segment.
    #    NOTE: this assumes an axial crop. If a case is cropped at an angle, replace this with a test on the local surface normal instead (normals that are ~parallel to the mask's principal axis near the bounding-box extremes are crop faces, not ostia).
    z_present = np.where(mask_arr.any(axis=(1, 2)))[0]
    z_min, z_max = z_present.min(), z_present.max()
    cut_plane = np.zeros_like(surface)
    cut_plane[z_min:z_min + crop_margin_slices] = True
    cut_plane[z_max - crop_margin_slices + 1:z_max + 1] = True
    surface &= ~cut_plane

    # 3. outward normal field: gradient of the distance-to-mask transform, computed in physical mm (sampling=spacing) so anisotropy doesn't skew it
    dist_out = ndimage.distance_transform_edt(~mask_arr, sampling=spacing)
    gz, gy, gx = np.gradient(dist_out, *spacing)
    normals = np.stack([gz, gy, gx], axis=-1)
    norm_len = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, norm_len, out=np.zeros_like(normals), where=norm_len > 1e-6)

    # 4. probe a short distance outward from each surface voxel; keep it if it stays close to aortic-blood-pool intensity rather than dropping into fat/muscle
    shell_vox = max(1, int(round(shell_out_mm / spacing.mean())))
    intensity_floor = 148   # HU tolerance below the aortic pool

    hot = np.zeros_like(surface)
    for z, y, x in zip(*np.where(surface)):
        n = normals[z, y, x]
        if np.linalg.norm(n) < 1e-6:
            continue
        samples = []
        for step in range(1, shell_vox + 1):
            q = np.round([z, y, x] + n * step).astype(int)
            if not (0 <= q[0] < ct_arr.shape[0] and 0 <= q[1] < ct_arr.shape[1] and 0 <= q[2] < ct_arr.shape[2]):
                break
            samples.append(ct_arr[tuple(q)])
        if samples and np.mean(samples) > intensity_floor:
            hot[z, y, x] = True

    # 5. cluster surviving voxels -> one candidate launch point per cluster
    labeled, n_labels = ndimage.label(hot, structure=np.ones((3, 3, 3)))
    out = []
    for lbl in range(1, n_labels + 1):
        coords = np.argwhere(labeled == lbl)
        if len(coords) < min_cluster_voxels:
            continue
        centroid = coords.mean(axis=0)
        # Nearest voxel actually in the cluster, used as voxel_idx -- the
        # rounded centroid itself can fall outside the (possibly curved)
        # cluster, landing on a voxel with no reliable normal.
        nearest = coords[np.argmin(np.linalg.norm(coords - centroid, axis=1))]
        z, y, x = (int(v) for v in nearest)
        physical = np.array(ct.TransformContinuousIndexToPhysicalPoint(
            (centroid[2], centroid[1], centroid[0])            # (x, y, z) order!!!
        ))
        # Average the normal over the cluster's own voxels -- every one of
        # them already passed the near-zero-norm check above, so this is
        # always well-defined (unlike sampling a single, possibly
        # off-cluster, voxel).
        cluster_normals = normals[coords[:, 0], coords[:, 1], coords[:, 2]]
        avg_normal = cluster_normals.mean(axis=0)
        avg_normal /= (np.linalg.norm(avg_normal) + 1e-9)
        out.append(LaunchPoint(
            voxel_idx=(z, y, x),
            physical_xyz=physical,
            outward_normal=avg_normal,
            support=len(coords),
        ))
    return out