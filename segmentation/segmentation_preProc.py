
import matplotlib.pyplot as plt
import random
import SimpleITK as sitk
import os 
from scipy import ndimage
from scipy.ndimage import binary_opening, generate_binary_structure
import time
import numpy as np 
from importlib import reload
import json
import cv2
from datetime import datetime
import shutil
from scipy.optimize import curve_fit
from scipy.stats import norm

from segmentation.resample_image import resample_image
from segmentation.convert_npy_to_nii_gz import convert_npy_to_nii_gz

# Resolve paths
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
SEGMENTATION_RESULTS = os.path.join(ROOT, "segmentation", "results")
os.makedirs(SEGMENTATION_RESULTS, exist_ok=True)
OUTPUT_FOLDER = os.path.join(SEGMENTATION_RESULTS, datetime.now().strftime("%Y_%m_%d_%H_%M"))
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
LOG_FILE_PATH = os.path.join(OUTPUT_FOLDER, "preprocessing_" + datetime.now().strftime("%Y_%m_%d_%H_%M") + ".log")


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
            #(0 = background), (1: kidney), (2: tumor), (3: cyst)
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
            min_coords = np.max([np.min(nz, axis=1) - 63, [0,0,0]], axis=0)  # Ensure minimum coordinates are not negative
            max_coords = np.min([np.max(nz, axis=1) + 63, segmentation_array.shape], axis=0)  # Ensure maximum coordinates do not exceed the image size

            # Use the bounding box to crop the image and segmentation
            self.cropped_image = self.clipped_image_stack[min_coords[0]:max_coords[0],
                                                          min_coords[1]:max_coords[1],
                                                          min_coords[2]:max_coords[2]]
            self.cropped_segmentation = segmentation_array[min_coords[0]:max_coords[0],
                                                           min_coords[1]:max_coords[1],
                                                           min_coords[2]:max_coords[2]]

        def add_histology_data(self):
            # TODO: add another channel for histology data
            pass

        def save_data(self):
            # Derive case_id from the case_path
            case_id = os.path.basename(self.case_path)
            output_folder = os.path.join(OUTPUT_FOLDER, case_id)
            os.makedirs(output_folder, exist_ok=True)

            # TODO Save data as NiBabel files
            convert_npy_to_nii_gz(self.cropped_image, os.path.join(output_folder, "imaging.nii.gz"))
            convert_npy_to_nii_gz(self.cropped_segmentation, os.path.join(output_folder, "segmentation.nii.gz"))

            # # Save the cropped data
            # np.save(os.path.join(output_folder, 'cropped_image.npy'), self.cropped_image)
            # np.save(os.path.join(output_folder, 'cropped_segmentation.npy'), self.cropped_segmentation)
            
 
    def __init__(self, source_folder, histogram_path, histology_data_path): 
        """
        Constructor of class KidneyDatasetPreprocessor
        Args:
            source_folder (str): path to the source data folder that will be used in the preprocess
            histogram_path (str): path to the histogram data file
        """

        print("KidneyDatasetPreprocessor INITIALIZING...")

        self.source_folder = source_folder
        self.histogram_path = histogram_path
        self.histology_data_path = histology_data_path
        
        print("source_folder: " + source_folder)
        print("histogram_path: " + histogram_path)

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

    # to estimate Gaussian distribution parameters from self.histograms
    def compute_gaussian_fits(self):
        # Calculate bin centers as the midpoints of bin_edges for fitting
        # bin_centers contain an array of values show the midpoint between each pair of consecutive bin edges
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
        # label: key, histogram: value (loop through every items in histogram dictionary, as shown as "histogram.items()"") 
        # while histograms = np.load().item() is a single item retrieve from the list, then histograms.items() is get a whole list
        for label, histogram in self.histograms.items():
            # Estimate the mean and standard deviation from the histogram
            # Note: This estimation assumes the histogram approximates a Gaussian distribution
            # np.average(): computes weighted average of bin_centers, where weights are provided by the array histogram.
            mean = np.average(bin_centers, weights=histogram)
            variance = np.average((bin_centers - mean)**2, weights=histogram)
            sigma = np.sqrt(variance)
            
            # Alternatively, use norm.fit() to estimate the distribution parameters directly from sample data
            # This would require reconstructing the sample data from the histogram, which might not be straightforward
            
            # Store the estimated Gaussian parameters
            self.gaussian_fits[label] = {'mean': mean, 'std': sigma}

    def process_dataset(self):
        # List all subfolders in the source folder to determine the total number of cases
        # The below code is to cal the total cases by cal the length of the list of items exist in the source_folder file after filter out any files but the files that are directories.
        total_cases = len([name for name in os.listdir(self.source_folder) # iterates over each item in the list of files obtained from the source folder
                           if os.path.isdir(os.path.join(self.source_folder, name))]) # check if each item is a directory "".isdir()", if True then constructs the full path to the item ".join()"
        processed_cases = 0

        print(f"Total cases: {total_cases}")

        # Iterate over each subfolder in the source folder
        # If the item in the source folder is not a subfolder (not a case folder), it continues to the next iteration.
        # aka filter out files and only process directories within the source folder.
        for case_folder in os.listdir(self.source_folder):
            case_path = os.path.join(self.source_folder, case_folder)
            if not os.path.isdir(case_path):
                continue

            # Check if the target folder for this case already exists to determine if it should be skipped
            target_folder = os.path.join(OUTPUT_FOLDER, case_folder)
            if os.path.exists(target_folder):
                print(f"Skipping already processed case: {case_folder}")
                continue

            # Process the case
            casePreproc = self.KidneyCasePreprocessor(case_path, self)
            casePreproc.load_data()
            casePreproc.resample_data()

            # Process the original case
            self.process_case(casePreproc, processed_cases)

            processed_cases += 1

            print("PROCESSED " + str(processed_cases) + " / " + str(total_cases) + " CASES")


    def process_case(self, casePreproc, processed_cases):
        casePreproc.range_normalize()
        casePreproc.crop_data()
        casePreproc.add_histology_data()
        casePreproc.save_data()

        # # Save slices as images for a few cases
        # if processed_cases <= 4:
        #     self.save_slices_as_images(casePreproc)

    def save_slices_as_images(self, casePreproc):
        # Derive case_id from the case_path
        case_id = os.path.basename(casePreproc.case_path)
        output_folder = os.path.join(OUTPUT_FOLDER, case_id, "slices")
        os.makedirs(output_folder, exist_ok=True)

        # Save each slice of the image and segmentation
        for i in range(casePreproc.cropped_image.shape[2]):  # Iterate over z-axis (slices)
            # Save image slice
            plt.imsave(os.path.join(output_folder, f"img_slice_{i:03d}.png"), casePreproc.cropped_image[:, :, i], cmap='gray')

            # Save segmentation slice
            plt.imsave(os.path.join(output_folder, f"seg_slice_{i:03d}.png"), casePreproc.cropped_segmentation[:, :, i])

if __name__ == "__main__": 
    # Path to raw dataset
    dataset_path = "dataset_test"
    # Path to the histogram data file
    histogram_path = "hists/histogram_counts.npy"
    # Path to the histology data file
    histology_data_path = "histology_data/kits23_histology_data.json"
    # Path to the label mapping file
    label_mapping_path = "segmentation/labels.json"

    # Create an instance of the preprocessor
    preprocessor = KidneyDatasetPreprocessor(dataset_path, histogram_path, histology_data_path)

    # Process dataset
    preprocessor.process_dataset()

    # Copy the preprocessing script to the output folder
    script_name = os.path.basename(__file__)
    shutil.copy(script_name, OUTPUT_FOLDER)


