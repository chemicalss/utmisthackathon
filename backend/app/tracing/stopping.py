import numpy as np


def get_neighbors_26(voxel, shape):
    """Return 26-connected neighboring voxels."""

    x, y, z = voxel
    z_size, y_size, x_size = shape

    neighbors = []

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):

                if dx == 0 and dy == 0 and dz == 0:
                    continue

                nx = x + dx
                ny = y + dy
                nz = z + dz

                if (
                    0 <= nx < x_size
                    and 0 <= ny < y_size
                    and 0 <= nz < z_size
                ):
                    neighbors.append((nx, ny, nz))

    return neighbors


def _spur_length(start, exclude, branch_set, shape, max_voxels):
    """How many steps a branch starting at `start` keeps going (without
    re-entering `exclude`) before it dead-ends, capped at `max_voxels`.

    Skeletonization routinely leaves tiny one- or two-voxel spurs that
    aren't real anatomy; without this check, every spur would register
    as a second "forward continuation" and the very first one along the
    centreline would always be (mis)read as a bifurcation.
    """

    visited = {exclude, start}
    frontier = [start]
    depth = 1

    while frontier and depth < max_voxels:
        next_frontier = []
        for node in frontier:
            for neighbor in get_neighbors_26(node, shape):
                if neighbor in branch_set and neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
        if not next_frontier:
            break
        frontier = next_frontier
        depth += 1

    return depth


def find_first_bifurcation(centerline, branch_voxels, min_spur_voxels=3):
    """
    Find the first point where the vessel region
    has multiple forward directions.

    Returns the bifurcation voxel or None.
    """

    branch_set = set(branch_voxels)

    if len(centerline) < 3:
        return None

    shape = (
        max(v[2] for v in branch_voxels) + 1,
        max(v[1] for v in branch_voxels) + 1,
        max(v[0] for v in branch_voxels) + 1,
    )

    for i in range(1, len(centerline) - 1):

        previous = np.array(centerline[i - 1])
        current = np.array(centerline[i])

        direction = current - previous

        if np.linalg.norm(direction) == 0:
            continue

        forward_neighbors = []

        for neighbor in get_neighbors_26(
            tuple(current),
            shape
        ):

            if neighbor not in branch_set:
                continue

            movement = np.array(neighbor) - current

            if np.linalg.norm(movement) == 0:
                continue

            # Neighbor is considered forward if it
            # generally follows the current direction.
            if np.dot(direction, movement) > 0:
                forward_neighbors.append(neighbor)

        # Ignore forward continuations that immediately dead-end --
        # those are skeletonization spurs, not real branches.
        real_forward = [
            n for n in forward_neighbors
            if _spur_length(n, tuple(current), branch_set, shape, min_spur_voxels) >= min_spur_voxels
        ]

        # More than one forward continuation
        # suggests a bifurcation.
        if len(real_forward) >= 2:
            return tuple(current)

    return None


def _truncate_to_arclength(centerline, ct, max_mm):
    """Trim a voxel-index centerline to at most `max_mm` of physical arc
    length measured from its first point."""

    if len(centerline) < 2:
        return centerline

    physical = [
        np.array(ct.TransformIndexToPhysicalPoint(tuple(map(int, v))))
        for v in centerline
    ]

    cum = 0.0
    for i in range(1, len(physical)):
        cum += np.linalg.norm(physical[i] - physical[i - 1])
        if cum >= max_mm:
            return centerline[:i + 1]

    return centerline


def stop_at_bifurcation(branch, ct, max_mm=10.0):
    """
    Stop the traced branch at its first bifurcation, or after `max_mm`
    of proximal path if no bifurcation occurs first -- matching the
    challenge brief's "up to 10 mm beyond the ostium or until the first
    downstream bifurcation, whichever occurs first".
    """

    centerline = branch["centerline"]
    # Bifurcation detection needs the thin skeleton, not the full blobby
    # watershed region -- see utils.geometry.skeletonize_voxels. Falls
    # back to "voxels" for callers (e.g. this module's own __main__ demo)
    # that hand in an already-thin, hand-built voxel set directly.
    branch_voxels = branch.get("skeleton_voxels", branch["voxels"])

    bifurcation = find_first_bifurcation(
        centerline,
        branch_voxels
    )

    # No bifurcation found -- cap at max_mm of proximal path instead.
    if bifurcation is None:

        return {
            **branch,
            "bifurcation_found": False,
            "bifurcation_xyz_mm": None,
            "final_centerline": _truncate_to_arclength(centerline, ct, max_mm)
        }

    # Find where the bifurcation occurs
    bifurcation_index = centerline.index(
        bifurcation
    )

    # Convert voxel index → physical mm
    bifurcation_xyz_mm = tuple(
        ct.TransformIndexToPhysicalPoint(
            tuple(map(int, bifurcation))
        )
    )

    # Keep the centerline up to the bifurcation, further capped at
    # max_mm in case the bifurcation itself lies beyond it.
    final_centerline = _truncate_to_arclength(
        centerline[:bifurcation_index + 1], ct, max_mm
    )

    return {
        **branch,
        "bifurcation_found": True,
        "bifurcation_xyz_mm": bifurcation_xyz_mm,
        "final_centerline": final_centerline
    }


if __name__ == "__main__":

    from pathlib import Path
    import SimpleITK as sitk

    DATA_DIR = Path("backend/app/data/subject001")

    ct_path = DATA_DIR / "orig1.nii"
    mask_path = DATA_DIR / "mask1.nii"

    ct = sitk.ReadImage(str(ct_path))
    mask = sitk.ReadImage(str(mask_path))


    # ============================================================
    # TEST 1: NO BIFURCATION
    # ============================================================

    test_branch_1 = {

        "branch_id": 1,

        "seed_xyz_mm": None,

        "ostium_xyz_mm": tuple(
            ct.TransformIndexToPhysicalPoint(
                (250, 250, 80)
            )
        ),

        "voxels": [
            (250, 250, 80),
            (251, 250, 80),
            (252, 250, 80),
            (253, 250, 80),
            (254, 250, 80),
            (255, 249, 80),
            (256, 248, 80),
        ],

        "geodesic_cost": 42.7,

        "centerline": [
            (250, 250, 80),
            (251, 250, 80),
            (252, 250, 80),
            (253, 250, 80),
            (254, 250, 80),
            (255, 249, 80),
        ]
    }


    result_1 = stop_at_bifurcation(
        test_branch_1,
        ct
    )


    print("\n========================================")
    print("TEST 1: NO BIFURCATION")
    print("========================================")

    print(
        "Branch ID:",
        result_1["branch_id"]
    )

    print(
        "Bifurcation found:",
        result_1["bifurcation_found"]
    )

    print(
        "Bifurcation XYZ (mm):",
        result_1["bifurcation_xyz_mm"]
    )

    print(
        "Original centerline length:",
        len(result_1["centerline"])
    )

    print(
        "Final centerline length:",
        len(result_1["final_centerline"])
    )


    # ============================================================
    # TEST 2: BIFURCATION
    # ============================================================

    test_branch_2 = {

        "branch_id": 2,

        "seed_xyz_mm": None,

        "ostium_xyz_mm": tuple(
            ct.TransformIndexToPhysicalPoint(
                (250, 250, 80)
            )
        ),

        "voxels": [

            # Main vessel
            (250, 250, 80),
            (251, 250, 80),
            (252, 250, 80),
            (253, 250, 80),
            (254, 250, 80),

            # First daughter
            (255, 249, 80),
            (256, 248, 80),

            # Second daughter
            (255, 251, 80),
            (256, 252, 80),

            # Continue parent direction
            (255, 250, 80),
        ],

        "geodesic_cost": 35.0,

        # IMPORTANT:
        # There is a point AFTER 254.
        # This allows the algorithm to determine
        # the forward direction at 254.
        "centerline": [
            (250, 250, 80),
            (251, 250, 80),
            (252, 250, 80),
            (253, 250, 80),
            (254, 250, 80),
            (255, 250, 80),
        ]
    }


    result_2 = stop_at_bifurcation(
        test_branch_2,
        ct
    )


    print("\n========================================")
    print("TEST 2: BIFURCATION")
    print("========================================")

    print(
        "Branch ID:",
        result_2["branch_id"]
    )

    print(
        "Bifurcation found:",
        result_2["bifurcation_found"]
    )

    print(
        "Bifurcation XYZ (mm):",
        result_2["bifurcation_xyz_mm"]
    )

    print(
        "Original centerline length:",
        len(result_2["centerline"])
    )

    print(
        "Final centerline length:",
        len(result_2["final_centerline"])
    )

    print("\nFinal centerline:")

    for voxel in result_2["final_centerline"]:
        print(" ", voxel)
