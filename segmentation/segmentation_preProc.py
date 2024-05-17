
import matplotlib.pyplot as plt
import random
# SimpleITK: Simplified, open-source to Insight Toolkit(ITK) image analysis toolkit
import SimpleITK as sitk
# os = operating system
import os 
from scipy import ndimage
from scipy.ndimage import binary_opening, generate_binary_structure
import time
# numpy deal with arrays
import numpy as np 
from importlib import reload
#import debug_helpers as db
import json
import cv2
from datetime import datetime
# shutil helps in automating process of copying and removal of files and directories
import shutil
from scipy.optimize import curve_fit
from scipy.stats import norm

def resample_image(image, new_spacing=[1.0, 1.0, 1.0]):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Convert image to a numpy array to check the number of unique values
    image_array = sitk.GetArrayFromImage(image)
    num_unique_values = len(np.unique(image_array))

    # Determine if the input is an image or segmentation
    # Assuming segmentation will have a significantly smaller number of unique values
    if num_unique_values > 10:  # Threshold for distinguishing images from segmentations
        # Image: Use B-spline for x and y, Linear for z
        new_size_xy = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size[0:2], original_spacing[0:2], new_spacing[0:2])]
        new_size = [new_size_xy[0], new_size_xy[1], original_size[2]]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing([new_spacing[0], new_spacing[1], original_spacing[2]])
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkBSpline)

        resampled_image_xy = resampler.Execute(image)

        # Resample in z dimension using linear interpolation
        new_size_z = [new_size_xy[0], new_size_xy[1], int(round(original_size[2] * original_spacing[2] / new_spacing[2]))]
        
        resampler.SetSize(new_size_z)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetInterpolator(sitk.sitkLinear)

        resampled_image_xyz = resampler.Execute(resampled_image_xy)

        return sitk.GetArrayFromImage(resampled_image_xyz)

    else:
       # Process segmentation using distance map interpolation
        labels = np.unique(image_array)
        
        # Compute the size of the resampled image based on new_spacing
        new_size = [int(np.round(original_size[dim] * original_spacing[dim] / new_spacing[dim])) for dim in range(3)]
        
        # Initialize max_distance_map and label_map with dimensions of the resampled image
        new_size_reordered = [new_size[2], new_size[1], new_size[0]]  # Reordering to (z, y, x)
        max_distance_map = np.full(new_size_reordered, -np.inf)
        label_map = np.zeros(new_size_reordered, dtype=np.int16)
        
        for label in labels:
            if label == 0:  # Skip background
                continue
            
            # Create binary mask for current label
            binary_mask = (image_array == label).astype(np.int8)
            binary_mask_sitk = sitk.GetImageFromArray(binary_mask)
            binary_mask_sitk.CopyInformation(image)
            
            # Compute the signed distance map for the current label
            distance_map = sitk.SignedMaurerDistanceMap(binary_mask_sitk, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)
            
            # Resample the distance map as an image
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(image)  # Use original image to copy some properties
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetSize([int(round(original_size[dim] * original_spacing[dim] / new_spacing[dim])) for dim in range(3)])
            resampler.SetTransform(sitk.Transform())
            resampler.SetInterpolator(sitk.sitkLinear)
            resampled_distance_map_sitk = resampler.Execute(distance_map)
            
            # After converting the resampled SimpleITK distance map to a NumPy array
            resampled_distance_map = sitk.GetArrayFromImage(resampled_distance_map_sitk)

            # Now resampled_distance_map has the same dimension order as max_distance_map and label_map
            # Proceed with your comparison and label assignment
            label_mask = resampled_distance_map > max_distance_map
            max_distance_map[label_mask] = resampled_distance_map[label_mask]
            label_map[label_mask] = label
            
        # Resample the distance map as an image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)  # Use original image to copy some properties
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize([int(round(original_size[dim] * original_spacing[dim] / new_spacing[dim])) for dim in range(3)])
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkLinear)
        foreground_mask = resampler.Execute(image>0)

        # Background processing
        label_map[sitk.GetArrayFromImage(foreground_mask) == 0] = 0

        return label_map

class KidneyDatasetPreprocessor:
    class KidneyCasePreprocessor:
        def __init__(self, case_path, dataset_preprocessor):
            """
            Constructor of class KidneyCasePreprocessor:
            Args:
                case_path (str): path of cases (usually see as os.path.join() to join the many path strings into 1 string)
                dataset_preprocessor (TODO)
            """
            self.case_path = os.path.join(case_path)
            self.dataset_preprocessor = dataset_preprocessor

        def load_data(self):
            self.image = sitk.ReadImage(os.path.join(self.case_path, 'imaging.nii.gz'))
            self.segmentation = sitk.ReadImage(os.path.join(self.case_path, 'segmentation.nii.gz'))

        def resample_data(self):
            self.image_resampled_xyz = resample_image(self.image, [1.0, 1.0, 1.0])
            self.segmentation_resampled_xyz = resample_image(self.segmentation, [1.0, 1.0, 1.0])
        
        def range_normalize(self):
            # Number of quantiles (2^16 for 16-bit precision)
            num_quantiles = 2**16 + 1

            # Initialize a 3-channel image with the same spatial dimensions as the input image
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
    
            # Rescale to 16-bit precision (TODO)
            # converts the data type of the array elements to float32 ? (WHY 16-BIT PRECISION COMMENT ABOVE??)
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

        def save_data(self):
            # Derive case_id from the case_path
            case_id = os.path.basename(self.case_path)
            date_str = datetime.now().strftime("%Y_%m_%d")
            output_folder = os.path.join('C:\\Data\\2023_09_KidneyCancer\\kits23-main', date_str, case_id)
            os.makedirs(output_folder, exist_ok=True)

            # Save the cropped data
            np.save(os.path.join(output_folder, 'cropped_image.npy'), self.cropped_image)
            np.save(os.path.join(output_folder, 'cropped_segmentation.npy'), self.cropped_segmentation)
 
    def __init__(self, source_folder, hist_path): 
        """
        Constructor of class KidneyDatasetPreprocessor
        Args:
            source_folder (str): path to the source data folder that will be used in the preprocess
            hist_path (str): path to the histogram data file
        """

        self.source_folder = source_folder
        self.hist_path = hist_path
        
        # Load histogram data
        # (pickling: serialize convert obj to binary/bytes);
        # (.item(): to get the value of a key in a dict = to convert a loaded Numpy array back to its original obj form)                                                                   
        self.histograms = np.load(self.hist_path, allow_pickle=True).item() 
        
        # Generate array from [-1000,2001)
        self.bin_edges = np.arange(-1000, 2001)  # Fixed bin edges

        # create empty dictionary
        self.gaussian_fits = {} 

        # return date and time when the code is executed
        self.date_str = datetime.now().strftime("%Y_%m_%d")

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

        # Iterate over each subfolder in the source folder
        # If the item in the source folder is not a subfolder (not a case folder), it continues to the next iteration.
        # aka filter out files and only process directories within the source folder.
        for case_folder in os.listdir(self.source_folder):
            case_path = os.path.join(self.source_folder, case_folder)
            if not os.path.isdir(case_path):
                continue

            # Check if the target folder for this case already exists to determine if it should be skipped
            target_folder = os.path.join('C:\\Data\\2023_09_KidneyCancer\\kits23-main', self.date_str, case_folder)
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
            print(f"Processed {processed_cases} out of {total_cases} cases.")

    def process_case(self, casePreproc, processed_cases):
        casePreproc.range_normalize()
        casePreproc.crop_data()
        casePreproc.save_data()

        # Save slices as images for a few cases
        if processed_cases <= 4:  # Considering 0-based indexing (TODO)
            self.save_slices_as_images(casePreproc)

    def save_slices_as_images(self, casePreproc):
        # Derive case_id from the case_path
        case_id = os.path.basename(casePreproc.case_path)
        output_folder = os.path.join('C:\\Data\\2023_09_KidneyCancer\\kits23-main', self.date_str, case_id, "slices")
        os.makedirs(output_folder, exist_ok=True)

        # Save each slice of the image and segmentation
        for i in range(casePreproc.cropped_image.shape[2]):  # Iterate over z-axis (slices)
            # Save image slice
            plt.imsave(os.path.join(output_folder, f"img_slice_{i:03d}.png"), casePreproc.cropped_image[:, :, i], cmap='gray')

            # Save segmentation slice
            plt.imsave(os.path.join(output_folder, f"seg_slice_{i:03d}.png"), casePreproc.cropped_segmentation[:, :, i])

if __name__ == "__main__": 
    # Path to your dataset
    dataset_path = "dataset" 
    hist_path = 'C:\\Code\\2023_09_KC\\local\\hists\\histogram_counts.npy'

    # Create an instance of the preprocessor
    preprocessor = KidneyDatasetPreprocessor(dataset_path, hist_path)

    # Process dataset
    preprocessor.process_dataset()

    # Copy the preprocessing script to the output folder
    script_name = os.path.basename(__file__)
    shutil.copy(script_name, os.path.join('C:\\Data\\2023_09_KidneyCancer\\kits23-main', datetime.now().strftime("%Y_%m_%d")))

## Classification table
#classification_table = {
#    "multilocular_cystic_rcc": "MALIGN",
#    "oncocytoma": "BENIGN",
#    "clear_cell_papillary_rcc": "MALIGN",
#    "collecting_duct_undefined": "MALIGN",
#    "angiomyolipoma": "BENIGN",
#    "spindle_cell_neoplasm": "MALIGN",
#    "mest": "MALIGN",
#    "chromophobe": "MALIGN",
#    "papillary": "MALIGN",
#    "rcc_unclassified": "MALIGN",
#    "clear_cell_rcc": "MALIGN"
#}
