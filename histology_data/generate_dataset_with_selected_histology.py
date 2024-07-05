import json
import os
import shutil

# Path to the original dataset folder
dataset_folder = 'dataset_selected_histology_preprocessed'

# Output folder for the new dataset
output_folder = 'dataset_selected_histology_preprocessed_2'

# Path to the JSON file containing case information
json_file = 'histology_data/kits23_histology_data_selected.json'

# Load the JSON data
with open(json_file, 'r') as f:
    data = json.load(f)

# Iterate over each entry in the JSON data
for entry in data:
    case_id = entry['case_id']
    histologic_subtype = entry['tumor_histologic_subtype']
    
    # Check if histologic_subtype is not 'other'
    if histologic_subtype != 'other':
        # Source and destination paths
        source_folder = os.path.join(dataset_folder, case_id)
        destination_folder = os.path.join(output_folder, case_id)
        
        # Create destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)
        
        # Copy imaging.nii.gz and segmentation.nii.gz to the new location
        shutil.copy(os.path.join(source_folder, 'imaging.nii.gz'), destination_folder)
        shutil.copy(os.path.join(source_folder, 'segmentation.nii.gz'), destination_folder)
        
        print(f"Copied {case_id} with subtype {histologic_subtype}.")

print("Dataset generation complete.")
