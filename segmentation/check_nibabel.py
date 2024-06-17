import nibabel as nib

IMG_NII_PATH = "/home/toto/Projects/kits23/nnUNet_raw/Dataset079_KiTS2023HistogramPreprocessed/imagesTr original/case_00000_0000.nii.gz"

# Load the NIfTI file
img_nii_file = nib.load(IMG_NII_PATH)

# Check ndim
print("Image ndim:", img_nii_file.ndim)