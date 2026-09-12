from models.schemas import BranchRegion
from validation.persistence import check_persistence
from validation.direction import check_direction
from validation.continuity import check_continuity


def validate_branches(regions: list[BranchRegion],
                       min_persisted_mm: float = 5.0,
                       min_vesselness: float = 0.3) -> list[BranchRegion]:
    validated = []
    for region in regions:
        if not check_persistence(region, min_persisted_mm):
            continue
        if not check_direction(region):
            continue
        if region.mean_vesselness < min_vesselness:
            continue
        if not check_continuity(region):
            continue
        validated.append(region)
    return validated
