import nibabel as nib
import numpy as np
import sys

def check_nii_img(nii_img_path):
    # Load the NIfTI file
    img_nii_file = nib.load(nii_img_path)
    img_nii_data = img_nii_file.get_fdata()

    print("Loaded:", nii_img_path)

    # Check ndim
    print("Image ndim:", img_nii_file.ndim)
    # Check shape
    print("Image shape:", img_nii_file.shape)
    # Identify unique labels
    unique_labels = np.unique(img_nii_data)
    print("Unique labels in the image:", len(unique_labels))
    print("Labels:", unique_labels)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to the NIfTI file as an argument.")
        sys.exit(1)

    nii_img_path = sys.argv[1]
    check_nii_img(nii_img_path)