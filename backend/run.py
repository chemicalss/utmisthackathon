#!/usr/bin/env python
"""Command-line entry point for the aortic-branch-detection pipeline.

Usage:
    python run.py --image orig1.nii --aorta-mask mask1.nii --output prediction.json

Optionally save a qualitative visual check (aorta mask + detected ostia +
daughter-direction arrows) alongside the prediction:
    python run.py --image orig1.nii --aorta-mask mask1.nii --output prediction.json --visual check.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

from preprocessing.volume import load_volume  # noqa: E402
from pipeline.branch_detection import run_case_from_volumes  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect every artery arising directly from a supplied parent-aorta mask in a CT volume."
    )
    parser.add_argument("--image", required=True, help="Path to the CT volume (NIfTI).")
    parser.add_argument("--aorta-mask", required=True, dest="aorta_mask",
                         help="Path to the binary parent-aorta mask (NIfTI).")
    parser.add_argument("--output", required=True, help="Path to write the prediction JSON to.")
    parser.add_argument("--case-id", default=None, dest="case_id",
                         help="Case identifier to embed in the output (defaults to the image's parent folder name).")
    parser.add_argument("--visual", default=None,
                         help="Optional path (e.g. check.png) to save a 3-panel visual check: "
                              "aorta mask, detected ostia, and daughter-direction arrows.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading CT from {args.image}...", flush=True)
    start = time.time()
    ct, aorta_mask = load_volume(args.image, args.aorta_mask)
    print(f"Volumes loaded ({time.time()-start:.1f}s)", flush=True)

    case_id = args.case_id or Path(args.image).parent.name or Path(args.image).stem
    result = run_case_from_volumes(ct, aorta_mask, case_id)
    elapsed = time.time() - start

    print(f"\n[10] Writing prediction JSON...", flush=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"     Output: {out_path}", flush=True)

    if args.visual:
        print(f"[11] Generating visual check...", flush=True)
        from output.visualize import save_visual_check
        visual_path = Path(args.visual)
        visual_path.parent.mkdir(parents=True, exist_ok=True)
        save_visual_check(ct, aorta_mask, result, visual_path)
        print(f"     Visual: {visual_path}", flush=True)

    print(f"\n[DONE] {len(result['daughters'])} daughter branch(es) detected in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
