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


def find_first_bifurcation(centerline, branch_voxels):
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

        # More than one forward continuation
        # suggests a bifurcation.
        if len(forward_neighbors) >= 2:
            return tuple(current)

    return None


def stop_at_bifurcation(branch, ct):
    """
    Stop the traced branch at its first bifurcation.
    """

    centerline = branch["centerline"]
    branch_voxels = branch["voxels"]

    bifurcation = find_first_bifurcation(
        centerline,
        branch_voxels
    )

    # No bifurcation found
    if bifurcation is None:

        return {
            **branch,
            "bifurcation_found": False,
            "bifurcation_xyz_mm": None,
            "final_centerline": centerline
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

    # Keep the centerline up to the bifurcation
    final_centerline = centerline[
        :bifurcation_index + 1
    ]

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
