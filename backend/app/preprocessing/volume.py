from pathlib import Path
import SimpleITK as sitk


def load_volume(ct_path, mask_path):
    ct = sitk.ReadImage(str(ct_path))
    mask = sitk.ReadImage(str(mask_path))

    if ct.GetSize() != mask.GetSize():
        raise ValueError("CT and mask have different dimensions")

    if ct.GetSpacing() != mask.GetSpacing():
        raise ValueError("CT and mask have different spacing")

    return ct, mask


def main():
    # Testing with one patient
    DATA_DIR = Path("backend/app/data/subject001")

    ct_path = DATA_DIR / "orig1.nii"
    mask_path = DATA_DIR / "mask1.nii"

    ct, mask = load_volume(ct_path, mask_path)

    print("CT size:", ct.GetSize())
    print("Mask size:", mask.GetSize())
    print("Spacing:", ct.GetSpacing())
    print("Origin:", ct.GetOrigin())
    print("Direction:", ct.GetDirection())


if __name__ == "__main__":
    main()
