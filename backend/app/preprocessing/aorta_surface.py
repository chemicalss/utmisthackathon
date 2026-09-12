import SimpleITK as sitk

def extract_aorta_surface(mask):
    binary_mask = mask > 0
    surface = sitk.LabelContour(binary_mask)

    return surface

def main():
    from pathlib import Path

    # Testing with one patient
    DATA_DIR = Path("backend/app/data/subject001")
    mask_path = DATA_DIR / "mask1.nii"

    mask = sitk.ReadImage(str(mask_path))

    surface = extract_aorta_surface(mask)

    surface_array = sitk.GetArrayFromImage(surface)

    print("Surface size:", surface.GetSize())
    print("Number of surface voxels:", (surface_array > 0).sum())


if __name__ == "__main__":
    main()
