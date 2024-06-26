import os
import nibabel as nib

IMAGES_PATH = "nnUNet_raw/Dataset079_KiTS2023HistogramPreprocessed/imagesTr"

def convert_4d_to_3d(nii_img_path):
    # Load the 4D NIfTI image
    image_4d = nib.load(nii_img_path)
    
    # Get the data array from the image
    image_4d_data = image_4d.get_fdata()
    
    # Select a specific volume (e.g., the first volume)
    # Assuming the 4th dimension is the time dimension, we select the first time point
    image_3d_data = image_4d_data[..., 1]
    
    # Create a new NIfTI image with the 3D data
    image_3d = nib.Nifti1Image(image_3d_data, affine=image_4d.affine, header=image_4d.header)
    
    # Overwrite the original 4D NIfTI image with the 3D NIfTI image
    nib.save(image_3d, nii_img_path)
    

if __name__ == "__main__":

    print("Converting 4D NIfTI images to 3D NIfTI images...")

    # Get the list of NIfTI files in the images directory
    nii_files = [f for f in os.listdir(IMAGES_PATH) if f.endswith('.nii.gz')]
    
    count = 1

    # Iterate over the NIfTI files
    for nii_file in nii_files:
        nii_img_path = os.path.join(IMAGES_PATH, nii_file)
        
        # Convert the 4D NIfTI image to a 3D NIfTI image
        convert_4d_to_3d(nii_img_path)

        print(f"Converted {count}/{len(nii_files)}")
    
    print("All 4D NIfTI images converted to 3D NIfTI images.")