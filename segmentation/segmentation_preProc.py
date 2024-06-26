import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk
import os
import numpy as np
import json
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.stats import norm

from segmentation.resample_image import resample_image
from segmentation.convert_npy_to_nii_gz import convert_npy_to_nii_gz
from segmentation.convert_4d_to_3d import convert_4d_to_3d

# Resolve paths
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH


class KidneyDatasetPreprocessor:
    class KidneyCasePreprocessor:
        def __init__(self, case_path, dataset_preprocessor):
            """
            Constructor of class KidneyCasePreprocessor:
            Args:
                case_path (str): path of cases (usually see as os.path.join() to join the many path strings into 1 string)
                dataset_preprocessor
            """
            self.case_path = os.path.join(case_path)
            self.dataset_preprocessor = dataset_preprocessor

            print("Processing case: " + self.case_path)

        def load_data(self):
            self.image = sitk.ReadImage(os.path.join(self.case_path, 'imaging.nii.gz'))
            self.segmentation = sitk.ReadImage(os.path.join(self.case_path, 'segmentation.nii.gz'))

        def resample_data(self):
            self.image_resampled_xyz = resample_image(self.image, [1.0, 1.0, 1.0])
            self.segmentation_resampled_xyz = resample_image(self.segmentation, [1.0, 1.0, 1.0])

        def range_normalize(self):
            # Number of quantiles (2^16 for 16-bit precision)
            num_quantiles = 2**16 + 1

            # Initialize 3-channel image with the same spatial dimensions as the input image
            # and dtype float to accommodate [0, 1] range at 16-bit precision
            normalized_image = np.zeros((*self.image_resampled_xyz.shape, 3), dtype=np.float32)

            # Specify the labels of interest (skipping the background) 
            # (0 = background), (1: kidney), (2: tumor), (3: cyst)
            labels_of_interest = [1, 2, 3]

            for i, label in enumerate(labels_of_interest):
                params = self.dataset_preprocessor.gaussian_fits[label]
                mean, std = params['mean'], params['std']

                # Compute the quantiles for the Gaussian distribution
                quantiles = norm.ppf(np.linspace(0.5 / num_quantiles, 1 - 0.5 / num_quantiles, num_quantiles), loc=mean, scale=std)

                # Clip the image intensities to the range defined by the extreme quantiles
                # outside the range = its extreme quantiles (below [0] = [0])(upper [-1] = [-1])
                clipped_image = np.clip(self.image_resampled_xyz, quantiles[0], quantiles[-1])

                # Interpolate the original intensities to [0, 1] based on the quantiles
                interp_func = np.interp(clipped_image, quantiles, np.linspace(0, 1, num_quantiles))

                # Store the interpolated image in the corresponding channel i among the multichannel of the normalized_image
                normalized_image[..., i] = interp_func

            self.clipped_image_stack = np.clip(normalized_image, 0, 1).astype(np.float32)

        def crop_data(self):
            # Find the bounding box of non-zero regions in the segmentation
            segmentation_array = self.segmentation_resampled_xyz
            nz = np.nonzero(segmentation_array)

            # provide additional margin of 63 pixels around the segmented region
            min_coords = np.max([np.min(nz, axis=1) - 63, [0, 0, 0]], axis=0)  # Ensure minimum coordinates are not negative
            max_coords = np.min([np.max(nz, axis=1) + 63, segmentation_array.shape], axis=0)  # Ensure maximum coordinates do not exceed the image size

            # Use the bounding box to crop the image and segmentation
            self.cropped_image = self.clipped_image_stack[min_coords[0]:max_coords[0],
                                                          min_coords[1]:max_coords[1],
                                                          min_coords[2]:max_coords[2]]
            self.cropped_segmentation = segmentation_array[min_coords[0]:max_coords[0],
                                                           min_coords[1]:max_coords[1],
                                                           min_coords[2]:max_coords[2]]

        def add_histology_data(self):
            # Get the unique labels in the cropped segmentation
            unique_labels = np.unique(self.cropped_segmentation)

            # Get the tumor histologic subtype for this case
            case_id = os.path.basename(self.case_path)
            histology_info = next((item for item in self.dataset_preprocessor.histology_data if item['case_id'] == case_id), None)
            if histology_info is not None:
                subtype = histology_info['tumor_histologic_subtype']
                if subtype in self.dataset_preprocessor.label_mapping:
                    subtype_label = self.dataset_preprocessor.label_mapping[subtype]
                    if 2 in unique_labels:  # Check if tumor label is present
                        tumor_mask = (self.cropped_segmentation == 2)
                        self.cropped_segmentation[tumor_mask] = subtype_label

            self.cropped_segmentation = self.cropped_segmentation[..., np.newaxis]

        def save_data(self):
            # Derive case_id from the case_path
            case_id = os.path.basename(self.case_path)
            output_folder = os.path.join(OUTPUT_FOLDER, case_id)
            os.makedirs(output_folder, exist_ok=True)

            # Save data as NiBabel files
            convert_npy_to_nii_gz(self.cropped_image, os.path.join(output_folder, "imaging.nii.gz"))
            convert_npy_to_nii_gz(self.cropped_segmentation, os.path.join(output_folder, "segmentation.nii.gz"))

            # Convert 4D NIfTI images to 3D NIfTI images
            convert_4d_to_3d(os.path.join(output_folder, "imaging.nii.gz"))

    def __init__(self, source_folder, histogram_path, histology_data_path, label_mapping_path):
        """
        Constructor of class KidneyDatasetPreprocessor
        Args:
            source_folder (str): path to the source data folder that will be used in the preprocess
            histogram_path (str): path to the histogram data file
            histology_data_path (str): path to the histology data file
            label_mapping_path (str): path to the label mapping file
        """

        print("KidneyDatasetPreprocessor INITIALIZING...")

        self.source_folder = source_folder
        self.histogram_path = histogram_path
        self.histology_data_path = histology_data_path
        self.label_mapping_path = label_mapping_path

        print("source_folder: " + source_folder)
        print("histogram_path: " + histogram_path)
        print("histology_data_path: " + histology_data_path)
        print("label_mapping_path: " + label_mapping_path)

        # Load histogram data
        # (pickling: serialize convert obj to binary/bytes);
        # (.item(): to get the value of a key in a dict = to convert a loaded Numpy array back to its original obj form)
        self.histograms = np.load(self.histogram_path, allow_pickle=True).item()

        # Generate array from [-1000,2001)
        self.bin_edges = np.arange(-1000, 2001)  # Fixed bin edges

        # create empty dictionary
        self.gaussian_fits = {}

        # use the func compute_gaussian_fits
        self.compute_gaussian_fits()

        # Load label mapping data
        self.load_label_mapping()

        # Load histology data
        self.load_histology_data()

        # Extract case IDs to be processed
        self.case_ids_to_process = [entry['case_id'] for entry in self.histology_data]

    # to estimate Gaussian distribution parameters from self.histograms
    def compute_gaussian_fits(self):
        for label, histogram in self.histograms.items():
            bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

            try:
                # Fit a Gaussian distribution to the histogram
                popt, _ = curve_fit(norm.pdf, bin_centers, histogram, p0=[0, 1])

                # extract fitted params (mean = popt[0], std = popt[1])
                mean, std = popt
                self.gaussian_fits[label] = {'mean': mean, 'std': std}
            except RuntimeError:
                print(f"Warning: Gaussian fit did not converge for label {label}")
                self.gaussian_fits[label] = {'mean': 0, 'std': 1}  # Use default values if fit fails

    def load_label_mapping(self):
        with open(self.label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)['labels']

        self.histology_labels = []
        for key, value in self.label_mapping.items():
            if isinstance(value, list) and key != 'background':
                self.histology_labels.extend(value)

    def load_histology_data(self):
        with open(self.histology_data_path, 'r') as f:
            self.histology_data = json.load(f)

    def process_cases(self):
        count = 1
        # Iterate through each case in the source folder
        for case in os.listdir(self.source_folder):
            case_path = os.path.join(self.source_folder, case)

            # Process only if the case ID is in the list of cases to process
            if case in self.case_ids_to_process:
                case_processor = self.KidneyCasePreprocessor(case_path, self)
                case_processor.load_data()
                case_processor.resample_data()
                case_processor.range_normalize()
                case_processor.crop_data()
                case_processor.add_histology_data()
                case_processor.save_data()

                print(f"Processed case: {count}/{len(self.case_ids_to_process)}")
                count += 1


# Usage example
if __name__ == "__main__":
    SOURCE_FOLDER = "dataset"
    HISTOGRAM_PATH = "hists/histogram_counts.npy"
    HISTOLOGY_DATA_PATH = "histology_data/kits23_histology_data.json"
    LABEL_MAPPING_PATH = "segmentation/labels.json"
    OUTPUT_FOLDER = "dataset_histology_preprocessed"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    LOG_FILE_PATH = os.path.join(OUTPUT_FOLDER, "preprocessing_" + datetime.now().strftime("%Y_%m_%d_%H_%M") + ".log")

    preprocessor = KidneyDatasetPreprocessor(SOURCE_FOLDER, HISTOGRAM_PATH, HISTOLOGY_DATA_PATH, LABEL_MAPPING_PATH)
    preprocessor.process_cases()
