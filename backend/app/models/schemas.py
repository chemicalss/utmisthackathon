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
    instance_id: str
    ostium_xyz_mm: np.ndarray
    seed_xyz_mm: np.ndarray
    radius_mm: float
    direction_xyz: np.ndarray

@dataclass
class BranchCandidate:
    branch: BranchInstance
    persisted_mm: float        # how far the outward walk got before stopping
    direction_ok: bool         # outward_search didn't reject it for a sharp turn / lost signal
    mean_vesselness: float     # 0-1, averaged along the traced skeleton
    skeleton_connected: bool   # single connected piece, no gaps

@dataclass
class BranchRegion:
    launch: LaunchPoint
    region: np.ndarray      # bool mask, local to bbox — the watershed's ownership region for this branch
    bbox: tuple             # 3 slice objects mapping region back to the full CT array
    persisted_mm: float
    direction_ok: bool
    mean_vesselness: float
    region_connected: bool

@dataclass
class BranchRegion:
    launch: LaunchPoint
    region: np.ndarray
    bbox: tuple
    persisted_mm: float
    direction_ok: bool
    mean_vesselness: float
    region_connected: bool
    geodesic_cost: float   # new -- mean Dijkstra distance from seed to its own region