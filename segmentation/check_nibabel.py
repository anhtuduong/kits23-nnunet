import nibabel as nib

IMG_NII_PATH = "dataset/case_00000/imaging.nii.gz"

# Load the NIfTI file
img_nii_file = nib.load(IMG_NII_PATH)

# Check ndim
print("Image ndim:", img_nii_file.ndim)
# Check shape
print("Image shape:", img_nii_file.shape)