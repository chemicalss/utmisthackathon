# Aortic Branch Detection

Detects every artery that leaves the abdominal aorta directly in a CT
angiogram, given only the CT volume and a binary mask of the parent
aortic lumen, and reports each one as an independent daughter instance
(ostium, seed point, radius, direction) in the challenge's JSON format.

## Setup

```
pip install -r requirements.txt
```

## Run

```
python run.py --image orig1.nii --aorta-mask mask1.nii --output prediction.json
```

Optionally save a qualitative visual check (aorta mask outline, detected
ostia, daughter-direction arrows) alongside the prediction:

```
python run.py --image orig1.nii --aorta-mask mask1.nii --output prediction.json --visual check.png
```

## Method

Classical image processing throughout -- no learned model, no GPU. Six
stages, chained in `app/pipeline/branch_detection.py`:

1. **Launch points** (`app/detection/launch_points.py`) -- find
   candidate ostia on the aortic surface shell by walking a short
   distance outward along the surface normal and keeping spots that
   stay at blood-pool intensity (rather than dropping into fat/muscle).
   Crop faces from a partial aortic segment are excluded.
2. **Outward walk** (`app/detection/outward_search.py`) -- a cheap
   greedy centreline walk from each launch point, as a first-pass
   filter: rejects points where the signal goes dark, drifts, or turns
   too sharply within ~12 mm.
3. **Geodesic watershed** (`app/segmentation/geodesic_watershed.py`) --
   partitions a narrow band around the aorta between the surviving
   launch points via multi-source Dijkstra over a Frangi-vesselness
   cost field, giving each candidate branch its own voxel region.
4. **Validation** (`app/validation/validator.py`) -- keeps a region
   only if it persisted far enough, its walk didn't get rejected, its
   region is one connected piece, and its traced path is sufficiently
   vessel-like (mean Frangi vesselness along the walked path, not
   averaged over the whole region -- see Known limitations).
5. **Proximal tracing** (`app/tracing/proximal.py`,
   `app/tracing/stopping.py`) -- thins each validated region to a
   1-voxel skeleton, finds the farthest-point path from the ostium
   along it, and stops that path at the first genuine bifurcation (spur
   voxels from skeletonization are pruned so they aren't mistaken for
   one) or 10 mm, whichever comes first.
6. **Measurement + formatting** (`app/utils/geometry.py`,
   `app/output/formatter.py`) -- walks 5 mm along that path for the
   daughter seed (extrapolating past the traced path if it's shorter,
   since the seed is a fixed 5 mm offset by definition), estimates the
   local radius as the largest inscribed ball at the seed within the
   branch's own voxel region, and takes the direction as the unit
   vector from ostium to seed.

All physical coordinates are produced via
`SimpleITK.TransformIndexToPhysicalPoint`, never raw voxel indices.

## Runtime

~60 seconds on one CPU core for a 512x512x174 volume (the one dev-set
case available locally), comfortably within a 4-core/8 GB budget. The
dominant costs are the per-surface-voxel outward-normal probe in stage
1 and the Dijkstra solve in stage 3, both scoped to a narrow band
around the aorta rather than the full volume.

## Known limitations

- **Vesselness threshold is a starting point, not a tuned value.** No
  reference annotations were available in this repo to score
  precision/recall, so `validation/validator.py`'s `min_vesselness`
  default was set by inspecting the achievable range on the one real
  case available (subject001), not calibrated against ground truth.
  Revisit it once the dev-set reference annotations are available.
- **Competing seeds can starve a real branch's watershed region.**
  Because branch regions are carved out as a Voronoi-style split of the
  narrow band between all surviving launch points, a branch with a
  close neighbour can end up with a region too short to skeletonize
  past a couple of millimetres, undershooting the 5 mm seed offset
  (handled by extrapolation, not by discarding the branch). A
  seed-competition-aware watershed would be a more thorough fix.
- **Visual checks:** the "at least three cases" visual-check
  requirement could only be produced for one case (`subject001`) --
  it's the only sample data present in this repo (`app/data/` is
  gitignored). The `--visual` flag works on any case; running it on two
  more only needs their data added under `app/data/`.

## Tests

```
pip install -r requirements-dev.txt
pytest backend/tests
```

`tests/test_pipeline.py` is an end-to-end smoke test that runs the full
pipeline on whatever case data is present under `app/data/` and checks
the output against the required schema; it skips itself if no case data
is present (e.g. in CI, since that directory isn't committed).
