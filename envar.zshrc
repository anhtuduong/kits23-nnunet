echo "Saving environment variables..."

export root_dir=$(pwd)
echo "Root directory: $root_dir"

export nnUNet_raw="$root_dir/nnUNet_raw"
export nnUNet_preprocessed="$root_dir/nnUNet_preprocessed"
export nnUNet_results="$root_dir/nnUNet_results"

export dataset_0="$root_dir/dataset"
export dataset_1="$root_dir/dataset_preprocessed"
export dataset_2="$root_dir/dataset_histology_preprocessed"
export dataset_3="$root_dir/dataset_selected_histology_preprocessed"

echo "Location of nnUNet_raw: $nnUNet_raw"
echo "Location of nnUNet_preprocessed: $nnUNet_preprocessed"
echo "Location of nnUNet_results: $nnUNet_results"
echo "Location of dataset_0: $dataset_0"
echo "Location of dataset_1: $dataset_1"
echo "Location of dataset_2: $dataset_2"
echo "Location of dataset_3: $dataset_3"
