import numpy as np
import SimpleITK as sitk
from models.schemas import LaunchPoint, OutwardTrack

#Greedy centerline walker 

def _sample_intensity(ct: sitk.Image, physical_xyz) -> float:
    idx = ct.TransformPhysicalPointToContinuousIndex(tuple(physical_xyz))
    size = ct.GetSize()
    ix, iy, iz = [int(round(v)) for v in idx]
    if not (0 <= ix < size[0] and 0 <= iy < size[1] and 0 <= iz < size[2]):
        return -3000.0  # out of bounds -> treat as air
    return ct.GetPixel(ix, iy, iz)


def _perpendiculars(direction):
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    p1 = np.cross(direction, arbitrary)
    p1 /= np.linalg.norm(p1)
    p2 = np.cross(direction, p1)
    return p1, p2


def walk_outward(ct: sitk.Image, launch: LaunchPoint,
                  step_mm: float = 0.5, max_mm: float = 12.0,
                  min_required_mm: float = 5.0,
                  intensity_floor: float = 150.0,
                  max_turn_deg: float = 60.0) -> OutwardTrack:
    """Cheap first-pass filter: re-centre on the brightest local blob each
    step, and bail if the path goes dark, drifts off, or turns too sharply
    (a proxy for 'direction consensus'). Not the final centerline — that
    comes from the geodesic step, which also separates nearby branches."""
    pos = launch.physical_xyz.copy()
    direction = launch.outward_normal / (np.linalg.norm(launch.outward_normal) + 1e-9)
    path = [pos.copy()]
    travelled = 0.0

    while travelled < max_mm:
        perp1, perp2 = _perpendiculars(direction)
        best_offset, best_val = np.zeros(3), -np.inf
        for r in (0.0, 0.3, 0.6):
            for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                offset = r * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
                val = _sample_intensity(ct, pos + offset)
                if val > best_val:
                    best_val, best_offset = val, offset
        if best_val < intensity_floor:
            break
        pos = pos + best_offset
        candidate = pos + direction * step_mm
        if _sample_intensity(ct, candidate) < intensity_floor:
            break
        step_dir = candidate - path[-1]
        if np.linalg.norm(step_dir) > 1e-6:
            step_dir /= np.linalg.norm(step_dir)
            turn = np.degrees(np.arccos(np.clip(np.dot(step_dir, direction), -1, 1)))
            if turn > max_turn_deg:
                break
            direction = 0.7 * direction + 0.3 * step_dir
            direction /= np.linalg.norm(direction)
        pos = candidate
        path.append(pos.copy())
        travelled += step_mm

    accepted = travelled >= min_required_mm
    return OutwardTrack(
        launch=launch, accepted=accepted, path_physical=np.array(path),
        rejection_reason="" if accepted else f"lost centreline at {travelled:.1f} mm",
    )