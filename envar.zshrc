echo "Saving environment variables..."

export root_dir=$(pwd)
echo "Root directory: $root_dir"

export nnUNet_raw="$root_dir/nnUNet_raw"
export nnUNet_preprocessed="$root_dir/nnUNet_preprocessed"
export nnUNet_results="$root_dir/nnUNet_results"
export kits23_dataset="$root_dir/dataset"
export kits23_dataset_histology_preprocessed="$root_dir/dataset_histology_preprocessed"

echo "Location of nnUNet_raw: $nnUNet_raw"
echo "Location of nnUNet_preprocessed: $nnUNet_preprocessed"
echo "Location of nnUNet_results: $nnUNet_results"
echo "Location of kits23_dataset: $kits23_dataset"
echo "Location of kits23_dataset_histology_preprocessed: $kits23_dataset_histology_preprocessed"
