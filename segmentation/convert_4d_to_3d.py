import os
import nibabel as nib
import numpy as np

def convert_4d_to_3d(nii_img_path, output_path, volume_index=0):
    # Load the 4D NIfTI image
    image_4d = nib.load(nii_img_path)
    
    # Get the data array from the image, ensure dtype matches the original
    image_4d_data = image_4d.get_fdata()
    
    # Check the original data type and unique values
    original_dtype = image_4d_data.dtype
    
    # Select a specific volume (e.g., the first volume)
    image_3d_data = image_4d_data[..., volume_index]
    
    # Cast to the original data type if necessary
    if image_3d_data.dtype != original_dtype:
        image_3d_data = image_3d_data.astype(original_dtype)
    
    # Create a new NIfTI image with the 3D data
    image_3d = nib.Nifti1Image(image_3d_data, affine=image_4d.affine, header=image_4d.header)
    
    # Save the 3D NIfTI image
    nib.save(image_3d, output_path)

    print(f"ndim: {image_3d.ndim} shape: {image_3d.shape} unique labels: {len(np.unique(image_3d_data))} labels: {np.unique(image_3d_data)}")

def convert_4d_to_3d_integer(nii_img_path, output_path, volume_index=0):
    # Load the 4D NIfTI image
    image_4d = nib.load(nii_img_path)
    
    # Get the data array from the image
    image_4d_data = image_4d.get_fdata()
    
    # Convert the data to integer type
    image_4d_data = image_4d_data.astype(np.int32)
    
    # Select a specific volume (e.g., the first volume)
    image_3d_data = image_4d_data[..., volume_index]
    
    # Create a new NIfTI image with the 3D data
    image_3d = nib.Nifti1Image(image_3d_data, affine=image_4d.affine, header=image_4d.header)
    
    # Save the 3D NIfTI image
    nib.save(image_3d, output_path)
    
    print(f"ndim: {image_3d.ndim} shape: {image_3d.shape} unique labels: {len(np.unique(image_3d_data))} labels: {np.unique(image_3d_data)}")

