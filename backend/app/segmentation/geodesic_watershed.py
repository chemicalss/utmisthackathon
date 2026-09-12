import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from models.schemas import LaunchPoint, OutwardTrack, BranchRegion
from validation.vesselness import compute_vesselness


def unpack_contract(ct: sitk.Image, aorta_mask: sitk.Image):
    """Pulls every variable named in the team contract. spacing/origin/
    direction come straight from the sitk.Image -- SimpleITK already
    returns these in (x, y, z) world order, so there's no reversing
    needed here; the reversal only happens locally, for numpy calls
    that operate on the (z, y, x)-indexed arrays."""
    ct_array = sitk.GetArrayFromImage(ct)                             # (z, y, x)
    aorta_mask_array = sitk.GetArrayFromImage(aorta_mask).astype(bool)  # (z, y, x)
    spacing = ct.GetSpacing()      # (x, y, z) mm
    origin = ct.GetOrigin()        # (x, y, z) mm
    direction = ct.GetDirection()  # 9-tuple orientation matrix
    return ct_array, aorta_mask_array, spacing, origin, direction


def merge_nearby_launch_points(points: list[LaunchPoint],
                                min_separation_mm: float = 4.0) -> list[LaunchPoint]:
    n = len(points)
    if n <= 1:
        return points
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i].physical_xyz - points[j].physical_xyz) < min_separation_mm:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for indices in groups.values():
        if len(indices) == 1:
            merged.append(points[indices[0]])
            continue
        members = [points[k] for k in indices]
        total_support = sum(m.support for m in members)
        avg_xyz = sum(m.physical_xyz * m.support for m in members) / total_support
        avg_normal = sum(m.outward_normal * m.support for m in members) / total_support
        avg_normal /= (np.linalg.norm(avg_normal) + 1e-9)
        anchor = max(members, key=lambda m: m.support)  # need a real voxel_idx to seed from
        merged.append(LaunchPoint(anchor.voxel_idx, avg_xyz, avg_normal, total_support))
    return merged


def _narrow_band(aorta_mask_array: np.ndarray, spacing_zyx: np.ndarray, band_mm: float = 12.0):
    dist_out = ndimage.distance_transform_edt(~aorta_mask_array, sampling=spacing_zyx)
    band = (dist_out > 0) & (dist_out <= band_mm)
    bbox = ndimage.find_objects(band.astype(int))[0]
    return band, bbox


def _voxel_graph(cost, spacing_zyx):
    shape = cost.shape
    idx = np.arange(cost.size).reshape(shape)
    offsets = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dz, dy, dx) != (0, 0, 0)]
    rows, cols, weights = [], [], []
    for dz, dy, dx in offsets:
        step_len = np.linalg.norm(np.array([dz, dy, dx]) * spacing_zyx)
        z0, z1 = max(0, -dz), shape[0] - max(0, dz)
        y0, y1 = max(0, -dy), shape[1] - max(0, dy)
        x0, x1 = max(0, -dx), shape[2] - max(0, dx)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue
        src = idx[z0:z1, y0:y1, x0:x1]
        dst = idx[z0+dz:z1+dz, y0+dy:y1+dy, x0+dx:x1+dx]
        w = 0.5 * (cost[z0:z1, y0:y1, x0:x1] + cost[z0+dz:z1+dz, y0+dy:y1+dy, x0+dx:x1+dx]) * step_len
        valid = np.isfinite(w)
        rows.append(src[valid].ravel()); cols.append(dst[valid].ravel()); weights.append(w[valid].ravel())
    rows, cols, weights = np.concatenate(rows), np.concatenate(cols), np.concatenate(weights)
    return csr_matrix((weights, (rows, cols)), shape=(cost.size, cost.size))


def geodesic_watershed(ct: sitk.Image, aorta_mask: sitk.Image,
                        tracks: list[OutwardTrack], band_mm: float = 12.0) -> list[BranchRegion]:
    """Geodesic Path Search + Geodesic Watershed. Stops at region masks
    and diagnostics for validation -- no seed point, no direction, no
    radius. Those need a traced centreline, which is the next stage."""
    ct_array, aorta_mask_array, spacing_xyz, _, _ = unpack_contract(ct, aorta_mask)
    spacing_zyx = np.array(spacing_xyz)[::-1]

    accepted_tracks = [t for t in tracks if t.accepted]
    launch_points = [t.launch for t in accepted_tracks]
    if not launch_points:
        return []

    band, bbox = _narrow_band(aorta_mask_array, spacing_zyx, band_mm)
    vesselness = compute_vesselness(ct, bbox)
    cost = 1.0 - 0.95 * vesselness
    cost[~band[bbox]] = np.inf
    graph = _voxel_graph(cost, spacing_zyx)

    seed_indices = [
        np.ravel_multi_index(tuple(lp.voxel_idx[i] - bbox[i].start for i in range(3)), cost.shape)
        for lp in launch_points
    ]
    dist_matrix = dijkstra(graph, indices=seed_indices, directed=False)
    owner = np.argmin(dist_matrix, axis=0).reshape(cost.shape)
    reachable = np.isfinite(np.min(dist_matrix, axis=0)).reshape(cost.shape)

    regions = []
    for i, track in enumerate(accepted_tracks):
        region = (owner == i) & reachable & band[bbox]
        if region.sum() < 5:
            continue
        _, n_components = ndimage.label(region, structure=np.ones((3, 3, 3)))
        persisted_mm = (
            np.linalg.norm(np.diff(track.path_physical, axis=0), axis=1).sum()
            if len(track.path_physical) > 1 else 0.0
        )
        regions.append(BranchRegion(
            launch=track.launch, region=region, bbox=bbox,
            persisted_mm=persisted_mm, direction_ok=track.accepted,
            mean_vesselness=float(vesselness[region].mean()),
            region_connected=(n_components == 1),
            geodesic_cost=float(dist_matrix[i][region.ravel()].mean()),
        ))
    return regions


def _region_to_voxel_list(region: np.ndarray, bbox) -> list[tuple]:
    """Local (z, y, x) region mask -> GLOBAL (x, y, z) voxel-index tuples.
    Global indices in (x, y, z) order mean your teammate can hand a tuple
    straight to ct.TransformIndexToPhysicalPoint(...) without needing to
    know this module's internal bbox cropping."""
    local_coords = np.argwhere(region)
    offsets = np.array([bbox[0].start, bbox[1].start, bbox[2].start])
    global_coords = local_coords + offsets
    return [(int(x), int(y), int(z)) for z, y, x in global_coords]


def export_branches(regions: list[BranchRegion], ct: sitk.Image) -> list[dict]:
    """Final formatting step, run after validation. Produces the shared
    handoff contract exactly. seed_xyz_mm is deliberately None -- filling
    it in is proximal tracing's job, not this module's."""
    branches = []
    for branch_id, r in enumerate(regions, start=1):
        ostium_xyz_mm = tuple(ct.TransformIndexToPhysicalPoint((
            int(r.launch.voxel_idx[2]), int(r.launch.voxel_idx[1]), int(r.launch.voxel_idx[0])
        )))
        branches.append({
            "branch_id": branch_id,
            "seed_xyz_mm": None,
            "ostium_xyz_mm": ostium_xyz_mm,
            "voxels": _region_to_voxel_list(r.region, r.bbox),
            "geodesic_cost": r.geodesic_cost,
        })
    return branches