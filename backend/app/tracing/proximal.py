import numpy as np
import SimpleITK as sitk
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def get_neighbors_26(voxel, shape):
    """Return 26-connected neighbors.

    voxel = (x, y, z)
    shape = NumPy shape (z, y, x)
    """

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


def build_branch_graph(branch_voxels):
    """Create a graph connecting neighboring branch voxels."""

    voxels = list(branch_voxels)
    branch_set = set(voxels)

    voxel_to_id = {
        voxel: i
        for i, voxel in enumerate(voxels)
    }

    rows = []
    cols = []
    weights = []

    shape = (
        max(v[2] for v in voxels) + 1,
        max(v[1] for v in voxels) + 1,
        max(v[0] for v in voxels) + 1,
    )

    for voxel in voxels:

        i = voxel_to_id[voxel]

        for neighbor in get_neighbors_26(voxel, shape):

            if neighbor not in branch_set:
                continue

            j = voxel_to_id[neighbor]

            distance = np.linalg.norm(
                np.array(neighbor) - np.array(voxel)
            )

            rows.append(i)
            cols.append(j)
            weights.append(distance)

    graph = csr_matrix(
        (weights, (rows, cols)),
        shape=(len(voxels), len(voxels))
    )

    return graph, voxels, voxel_to_id


def find_start_voxel(branch, ct):
    """Find the branch voxel closest to the candidate ostium."""

    branch_voxels = branch["voxels"]

    ostium_index = ct.TransformPhysicalPointToIndex(
        branch["ostium_xyz_mm"]
    )

    start_voxel = min(
        branch_voxels,
        key=lambda voxel: np.linalg.norm(
            np.array(voxel) - np.array(ostium_index)
        )
    )

    return start_voxel


def trace_proximal(branch, ct):
    """
    Trace the validated branch from the ostium outward.

    Returns the branch with an ordered centerline.
    """

    branch_voxels = branch["voxels"]

    if len(branch_voxels) < 2:
        return {
            **branch,
            "centerline": branch_voxels
        }

    graph, voxels, voxel_to_id = build_branch_graph(
        branch_voxels
    )

    start_voxel = find_start_voxel(branch, ct)
    start_id = voxel_to_id[start_voxel]

    distances, predecessors = dijkstra(
        graph,
        indices=start_id,
        return_predecessors=True
    )

    reachable = np.where(np.isfinite(distances))[0]

    if len(reachable) == 0:
        return {
            **branch,
            "centerline": []
        }

    # Farthest point in the branch region
    end_id = reachable[
        np.argmax(distances[reachable])
    ]

    # Reconstruct path backwards
    path = []

    current = end_id

    while current != start_id:

        path.append(voxels[current])

        previous = predecessors[current]

        if previous < 0:
            break

        current = previous

    path.append(voxels[start_id])

    path.reverse()

    return {
        **branch,
        "centerline": path
    }


def centerline_to_physical(centerline, ct):
    """Convert (x,y,z) voxel indices to physical mm coordinates."""

    return [
        tuple(
            ct.TransformIndexToPhysicalPoint(
                tuple(map(int, voxel))
            )
        )
        for voxel in centerline
    ]

if __name__ == "__main__":
    from pathlib import Path

    # Load test CT
    DATA_DIR = Path("backend/app/data/subject001")

    ct_path = DATA_DIR / "orig1.nii"
    mask_path = DATA_DIR / "mask1.nii"

    ct = sitk.ReadImage(str(ct_path))
    mask = sitk.ReadImage(str(mask_path))

    # --------------------------------------------------
    # Temporary test branch
    # --------------------------------------------------
    #
    # Replace these with an actual branch from your
    # watershed/validation output when available.
    #

    test_branch = {
        "branch_id": 1,

        "seed_xyz_mm": None,

        "ostium_xyz_mm": tuple(
            ct.TransformIndexToPhysicalPoint((250, 250, 80))
        ),

        "voxels": [
            (250, 250, 80),
            (251, 250, 80),
            (252, 250, 80),
            (253, 251, 80),
            (254, 251, 80),
            (255, 252, 80),
        ],

        "geodesic_cost": 42.7,
    }

    # --------------------------------------------------
    # Run proximal tracing
    # --------------------------------------------------

    traced_branch = trace_proximal(
        test_branch,
        ct
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("Branch ID:", traced_branch["branch_id"])

    print(
        "Number of branch voxels:",
        len(traced_branch["voxels"])
    )

    print(
        "Number of centerline voxels:",
        len(traced_branch["centerline"])
    )

    print("Centerline:")

    for voxel in traced_branch["centerline"]:
        print(" ", voxel)

    # Convert centerline to physical coordinates
    centerline_mm = centerline_to_physical(
        traced_branch["centerline"],
        ct
    )

    print("\nCenterline in mm:")

    for point in centerline_mm:
        print(" ", point)
