from dataclasses import dataclass
import numpy as np

@dataclass
class LaunchPoint:
    voxel_idx: tuple[int, int, int]   # (z, y, x) on the aorta surface
    physical_xyz: np.ndarray          # mm
    outward_normal: np.ndarray        # unit vector, physical space
    support: int                      # connected-component size backing this candidate

@dataclass
class OutwardTrack:
    launch: LaunchPoint
    accepted: bool
    path_physical: np.ndarray
    rejection_reason: str = ""

@dataclass
class BranchInstance:
    """One daughter branch, in exactly the shape the submission format
    requires (see output/formatter.py)."""
    instance_id: str
    parent_instance_id: str
    ostium_xyz_mm: list[float]
    seed_xyz_mm: list[float]
    radius_mm: float
    direction_xyz: list[float]

@dataclass
class BranchRegion:
    launch: LaunchPoint
    region: np.ndarray      # bool mask, local to bbox — the watershed's ownership region for this branch
    bbox: tuple             # 3 slice objects mapping region back to the full CT array
    persisted_mm: float
    direction_ok: bool
    mean_vesselness: float
    region_connected: bool
    geodesic_cost: float    # mean Dijkstra distance from seed to its own region