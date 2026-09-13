import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

from ml.model import extract_features
from pipeline.branch_detection import detect_and_segment_regions  # returns BranchRegion list, pre-export
import SimpleITK as sitk


def _matches_reference(predicted_xyz, reference_ostia, tolerance_mm=5.0) -> bool:
    for ref_xyz in reference_ostia:
        if np.linalg.norm(np.array(predicted_xyz) - np.array(ref_xyz)) < tolerance_mm:
            return True
    return False


def build_training_set(dev_cases_dir: str):
    """dev_cases_dir/<case>/orig.nii, aorta_mask.nii, reference.json"""
    X, y = [], []
    for case_dir in Path(dev_cases_dir).iterdir():
        ct = sitk.ReadImage(str(case_dir / "orig.nii"))
        aorta_mask = sitk.ReadImage(str(case_dir / "aorta_mask.nii"))
        reference = json.loads((case_dir / "reference.json").read_text())
        reference_ostia = [d["ostium_xyz_mm"] for d in reference["daughters"]]

        regions = detect_and_segment_regions(ct, aorta_mask)  # pre-validation candidates
        for region in regions:
            ostium_xyz = ct.TransformIndexToPhysicalPoint((
                int(region.launch.voxel_idx[2]),
                int(region.launch.voxel_idx[1]),
                int(region.launch.voxel_idx[0]),
            ))
            label = int(_matches_reference(ostium_xyz, reference_ostia))
            X.append(extract_features(region))
            y.append(label)
    return np.array(X), np.array(y)


def train(dev_cases_dir: str, model_out_path: str = "ml/branch_classifier.joblib"):
    X, y = build_training_set(dev_cases_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,       # number of trees voting
        max_depth=6,            # keeps trees shallow -- small dataset, avoid overfitting
        min_samples_leaf=3,
        class_weight="balanced",  # false positives will outnumber real branches; don't let the model just always predict "fake"
        random_state=42,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"train accuracy: {train_acc:.3f}, held-out accuracy: {test_acc:.3f}")

    joblib.dump(clf, model_out_path)
    print(f"saved model to {model_out_path}")


if __name__ == "__main__":
    train(dev_cases_dir="data/dev_cases")