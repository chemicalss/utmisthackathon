import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

"""Small shared geometry helpers used once proximal tracing has produced
a physical-space centreline: walking a fixed arc length along it (for the
daughter seed), estimating the local vessel radius from the branch's own
voxel set, and thinning that voxel set down to a 1-voxel skeleton for
bifurcation detection."""


def unit_vector(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def point_at_arclength(physical_path, target_mm: float):
    """Walk a polyline of mm points (in travel order, starting at/near the
    ostium) and return the point at `target_mm` arc length from the first
    point, plus the unit tangent of the segment it falls on.

    Interpolates linearly between samples. If the path is shorter than
    `target_mm` (e.g. a proximal trace cut short by an early
    bifurcation), extrapolates past the last point along its final
    segment's direction rather than clamping -- the seed is defined as a
    fixed 5 mm offset, so falling back to a closer point would misstate
    it by construction rather than just losing some precision.
    """
    pts = np.asarray(physical_path, dtype=float)
    if len(pts) == 0:
        return None, None
    if len(pts) == 1:
        return pts[0].copy(), None

    cum = 0.0
    for i in range(1, len(pts)):
        seg = pts[i] - pts[i - 1]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-9:
            continue
        if cum + seg_len >= target_mm:
            frac = (target_mm - cum) / seg_len
            point = pts[i - 1] + frac * seg
            return point, unit_vector(seg)
        cum += seg_len

    tangent = unit_vector(pts[-1] - pts[-2])
    point = pts[-1] + tangent * (target_mm - cum)
    return point, tangent


def _local_voxel_array(voxels_xyz):
    """Build a local (z, y, x) boolean array (1-voxel background padding
    on every side) from global (x, y, z) integer voxel indices, plus the
    `mins` offset needed to map local coordinates back to global ones."""
    voxels = np.asarray(list(voxels_xyz), dtype=int)
    mins = voxels.min(axis=0)
    maxs = voxels.max(axis=0)
    shape_xyz = (maxs - mins) + 3
    local = np.zeros(shape_xyz[::-1], dtype=bool)  # (z, y, x)
    local_idx = voxels - mins + 1
    local[local_idx[:, 2], local_idx[:, 1], local_idx[:, 0]] = True
    return local, mins


def local_radius_mm(voxels_xyz, seed_voxel_xyz, spacing_xyz) -> float:
    """Estimate the vessel radius (mm) at `seed_voxel_xyz` as the distance
    from that voxel to the nearest voxel outside the traced branch region
    -- i.e. the radius of the largest ball centred there that still fits
    inside the branch. This needs no re-scan of the CT: it reuses the
    per-branch voxel set the segmentation stage already produced.

    voxels_xyz: iterable of global (x, y, z) integer voxel indices
                belonging to the branch.
    seed_voxel_xyz: (x, y, z) integer voxel index of the daughter seed.
    spacing_xyz: (sx, sy, sz) mm, from ct.GetSpacing().
    """
    voxels = np.asarray(list(voxels_xyz), dtype=int)
    if len(voxels) == 0:
        return 0.0

    local, mins = _local_voxel_array(voxels)
    spacing_zyx = np.asarray(spacing_xyz, dtype=float)[::-1]
    dist = ndimage.distance_transform_edt(local, sampling=spacing_zyx)

    shape_xyz = np.array(local.shape[::-1])
    seed_local = np.asarray(seed_voxel_xyz, dtype=int) - mins + 1
    seed_local = np.clip(seed_local, 0, shape_xyz - 1)
    x, y, z = seed_local
    return float(dist[z, y, x])


def skeletonize_voxels(voxels_xyz) -> list[tuple]:
    """Thin a branch's volumetric voxel region down to a 1-voxel-wide 3D
    skeleton, returned as global (x, y, z) voxel-index tuples.

    Bifurcation detection needs this, not the raw region: the watershed
    territory is a solid blob (a Voronoi-style split of the narrow band
    between seeds), so almost every interior voxel has 2+ neighbours
    that count as "forward" purely because the region has volume -- that
    makes a naive neighbour-count check fire on the very first step, not
    at an actual branch point. On the thinned skeleton, the count only
    jumps where the structure actually splits.
    """
    voxels = np.asarray(list(voxels_xyz), dtype=int)
    if len(voxels) == 0:
        return []
    local, mins = _local_voxel_array(voxels)
    skeleton = skeletonize(local)
    skel_local_zyx = np.argwhere(skeleton)
    skel_global_xyz = skel_local_zyx[:, ::-1] + mins - 1
    return [tuple(int(v) for v in row) for row in skel_global_xyz]
