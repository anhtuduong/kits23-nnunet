import os
import nibabel as nib
import numpy as np

PREPROCESSED_PATH = "segmentation/results/2024_06_17_10_10"
NPY_IMG = "cropped_image.npy"
NPY_SEG = "cropped_segmentation.npy"
OUTPUT_PATH = "segmentation/converted/2024_06_17_10_10"
NII_IMG = "imaging.nii.gz"
NII_SEG = "segmentation.nii.gz"

def get_subfolders(directory):
    try:
        # List all directories in the specified directory
        subfolders = [f.name for f in os.scandir(directory) if f.is_dir()]
        return subfolders
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
        return []
    except PermissionError:
        print(f"Permission denied to access the directory '{directory}'.")
        return []
    
def convert_npy_to_nii_gz(npy_path, output_path):
    data = np.load(npy_path)
    img = nib.Nifti1Image(data, np.eye(4))  # Save axis for data (just identity)
    img.header.get_xyzt_units()
    img.to_filename(output_path)  # Save as NiBabel file


if __name__ == "__main__":
    
    # Create the output directory if it does not exist
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    # Get the list of subfolders in the npy directory
    subfolders = get_subfolders(PREPROCESSED_PATH)
    count = 1
    
    # Iterate over the subfolders
    for subfolder in subfolders:
        print(f"Processing: {count}/{len(subfolders)}")
        # Define the path to the current subfolder in the npy directory
        subfolder_path = os.path.join(PREPROCESSED_PATH, subfolder)
        
        # Define the path to the current subfolder in the nii.gz directory
        output_subfolder_path = os.path.join(OUTPUT_PATH, subfolder)
        
        # Create the output subfolder if it does not exist
        if not os.path.exists(output_subfolder_path):
            os.makedirs(output_subfolder_path)
        
        # Define the path to the current .npy file
        npy_img_path = os.path.join(subfolder_path, NPY_IMG)
        npy_seg_path = os.path.join(subfolder_path, NPY_SEG)
        
        # Define the output path for the corresponding .nii.gz file
        output_img_path = os.path.join(output_subfolder_path, NII_IMG)
        output_seg_path = os.path.join(output_subfolder_path, NII_SEG)
        
        # Convert the .npy file to .nii.gz format
        convert_npy_to_nii_gz(npy_img_path, output_img_path)
        convert_npy_to_nii_gz(npy_seg_path, output_seg_path)

        print(f"Converted '{npy_img_path}' to '{output_img_path}'")
        print(f"Converted '{npy_seg_path}' to '{output_seg_path}'")
        
        count += 1
    
    print("Conversion complete.")