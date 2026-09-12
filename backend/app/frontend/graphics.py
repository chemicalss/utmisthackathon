#!/usr/bin/env python3
"""
Branchseed Challenge - baseline daughter-branch detector.

Given a CT volume and a binary aorta mask, detects candidate daughter
arteries leaving the aorta and reports each one's ostium centre, seed
point, local radius, and direction, all in physical millimetres.

Usage:
    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json

This is a classical-image-processing baseline (no GPU, no internet,
no pretrained weights): it uses a Frangi vesselness filter to highlight
tube-like bright structures, then looks for ones touching the outside
of the aorta wall.
"""

import argparse
import json
import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.filters import frangi


def load_volumes(image_path, mask_path):
    """Read the CT and aorta mask. SimpleITK arrays come back as (z, y, x)."""
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    image_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    return image, mask_arr, image_arr


def build_search_shell(mask_arr, dilation_voxels=6):
    """A branch can only start at the aorta's outer wall, so restrict the
    search to a thin band just outside the supplied mask rather than the
    whole volume."""
    dilated = ndimage.binary_dilation(mask_arr, iterations=dilation_voxels)
    shell = dilated & (~mask_arr)
    return shell


def vesselness_map(image_arr, sigmas=(1, 2, 3)):
    """Frangi filter: scores each voxel by how tube-like its neighbourhood
    looks (arteries score high, blobs/flat regions score low)."""
    ptp = image_arr.max() - image_arr.min()
    norm = (image_arr - image_arr.min()) / (ptp + 1e-6)
    return frangi(norm, sigmas=sigmas, black_ridges=False)


def find_candidates(vessel_map, shell, mask_arr, min_voxels=15):
    """Threshold the vesselness map inside the search shell and split it
    into separate connected blobs - each one is a candidate branch stub."""
    if shell.any():
        thresh = vessel_map[shell].mean() + vessel_map[shell].std()
    else:
        thresh = vessel_map.mean()

    candidate_mask = (vessel_map > thresh) & shell
    labeled, n = ndimage.label(candidate_mask)

    z_indices = np.where(mask_arr.any(axis=(1, 2)))[0]
    z_min, z_max = z_indices[0], z_indices[-1]

    blobs = []
    for i in range(1, n + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) < min_voxels:
            continue
        centroid_z = coords[:, 0].mean()
        # skip blobs sitting right at the flat cropped top/bottom faces -
        # those are scan-edge artifacts, not real branch origins
        if abs(centroid_z - z_min) < 3 or abs(centroid_z - z_max) < 3:
            continue
        blobs.append(coords)
    return blobs, candidate_mask


def ostium_and_direction(coords, mask_arr):
    """Ostium = the blob voxel closest to the aorta wall.
    Direction = vector from the ostium toward the blob's far end."""
    aorta_coords = np.argwhere(mask_arr)
    centroid = coords.mean(axis=0)

    dists_to_centroid = np.linalg.norm(aorta_coords - centroid, axis=1)
    anchor = aorta_coords[dists_to_centroid.argmin()]

    blob_dists = np.linalg.norm(coords - anchor, axis=1)
    ostium_vox = coords[blob_dists.argmin()]
    far_vox = coords[blob_dists.argmax()]

    direction = far_vox.astype(float) - ostium_vox.astype(float)
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
    return ostium_vox, direction


def estimate_seed_and_radius(ostium_vox, direction, candidate_mask, spacing, step_mm=5.0):
    """Step outward ~5mm along the branch direction (respecting voxel
    spacing) to get the seed point, then measure the local lumen radius
    there with a distance transform."""
    # spacing from SimpleITK is (x, y, z); array/coords are (z, y, x)
    voxel_step = np.array([step_mm / spacing[2], step_mm / spacing[1], step_mm / spacing[0]])
    seed_vox = ostium_vox + direction * voxel_step

    dist_map = ndimage.distance_transform_edt(
        candidate_mask, sampling=(spacing[2], spacing[1], spacing[0])
    )
    z, y, x = [int(round(c)) for c in seed_vox]
    z = np.clip(z, 0, dist_map.shape[0] - 1)
    y = np.clip(y, 0, dist_map.shape[1] - 1)
    x = np.clip(x, 0, dist_map.shape[2] - 1)
    radius_mm = float(dist_map[z, y, x])
    if radius_mm <= 0:
        radius_mm = 1.0  # fallback so we never report zero
    return seed_vox, radius_mm


def voxel_to_physical(image, vox_zyx):
    """SimpleITK wants (x, y, z) index order and returns mm coordinates."""
    idx = (int(round(vox_zyx[2])), int(round(vox_zyx[1])), int(round(vox_zyx[0])))
    return image.TransformIndexToPhysicalPoint(idx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the CT volume (.nii/.nii.gz)")
    parser.add_argument("--aorta-mask", required=True, help="Path to the binary aorta mask")
    parser.add_argument("--output", required=True, help="Where to write the prediction JSON")
    args = parser.parse_args()

    image, mask_arr, image_arr = load_volumes(args.image, args.aorta_mask)
    spacing = image.GetSpacing()  # (x, y, z) mm per voxel

    shell = build_search_shell(mask_arr)
    vessel_map = vesselness_map(image_arr)
    candidates, candidate_mask = find_candidates(vessel_map, shell, mask_arr)

    daughters = []
    for i, coords in enumerate(candidates, start=1):
        ostium_vox, direction = ostium_and_direction(coords, mask_arr)
        seed_vox, radius_mm = estimate_seed_and_radius(
            ostium_vox, direction, candidate_mask, spacing
        )

        ostium_mm = voxel_to_physical(image, ostium_vox)
        seed_mm = voxel_to_physical(image, seed_vox)

        dir_mm = np.array(seed_mm) - np.array(ostium_mm)
        norm = np.linalg.norm(dir_mm)
        direction_xyz = (dir_mm / norm).tolist() if norm > 0 else [0.0, 0.0, 1.0]

        daughters.append({
            "instance_id": f"branch_{i:03d}",
            "parent_instance_id": "aorta",
            "ostium_xyz_mm": [round(v, 2) for v in ostium_mm],
            "seed_xyz_mm": [round(v, 2) for v in seed_mm],
            "radius_mm": round(radius_mm, 2),
            "direction_xyz": [round(v, 4) for v in direction_xyz],
        })

    case_id = args.image.split("/")[-1].split(".")[0]
    result = {
        "case_id": case_id,
        "parent": {"instance_id": "aorta"},
        "daughters": daughters,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {len(daughters)} candidate daughters to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
