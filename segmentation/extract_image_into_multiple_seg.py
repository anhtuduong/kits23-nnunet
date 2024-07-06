import os
import argparse
import nibabel as nib
import numpy as np

def extract_image_into_multiple_seg(NII_IMG_PATH, OUTPUT_DIR):
    print("Loading: ", NII_IMG_PATH)

    # Load the combined segmentation NIfTI file
    combined_img = nib.load(NII_IMG_PATH)
    combined_data = combined_img.get_fdata()

    # Identify unique labels in the combined image
    unique_labels = np.unique(combined_data)

    print("Unique labels in the combined image:", len(unique_labels))
    print("Labels:", unique_labels)


    # Exclude the background label (usually 0)
    unique_labels = unique_labels[unique_labels != 0]

    # Extract and save individual label images
    for label in unique_labels:
        # Create an empty array for the current label
        label_data = np.zeros(combined_data.shape)

        # Set the voxels that belong to the current label
        label_data[combined_data == label] = 1

        # Create a new NIfTI image for the current label
        label_img = nib.Nifti1Image(label_data, affine=combined_img.affine)

        # Save the label image as a new NIfTI file
        label_filename = f'label_{int(label)}.nii.gz'
        # Create a new directory if it does not exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        nib.save(label_img, os.path.join(OUTPUT_DIR, label_filename))

        print(f"Label {int(label)} NIfTI file saved as '{label_filename}'")

    print("All individual label NIfTI files have been saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate confusion matrices and plots for evaluation.')
    parser.add_argument('input_groundtruth', type=str, help='input ground truth dataset location')
    parser.add_argument('input_prediction', type=str, help='input prediction dataset location')
    parser.add_argument('dataset_id', type=int, help='dataset id')
    parser.add_argument('case_id', type=int, help='case id')

    args = parser.parse_args()

    input_groundtruth = args.input_groundtruth
    input_prediction = args.input_prediction
    output_dir = f"segmentation/visualize_segmentation/dataset{args.dataset_id}/case_{args.case_id}"

    extract_image_into_multiple_seg(input_groundtruth, os.path.join(output_dir, "groundtruth"))
    extract_image_into_multiple_seg(input_prediction, os.path.join(output_dir, "prediction"))

    
