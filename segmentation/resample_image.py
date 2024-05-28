import SimpleITK as sitk
import numpy as np
from utils.log import Log as log

def resample_image(image, new_spacing=[1.0, 1.0, 1.0]):
    """
    This function resamples a 3D medical image or segmentation
    to a new spacing, distinguishing between typical images
    and segmentations based on the number of unique values in the image data.
    - For images, it uses B-spline interpolation for x and y, and linear interpolation for z.
    - For segmentations, it uses distance map interpolation to maintain label boundaries accurately.

    @param image: The input 3D SimpleITK image to be resampled.
    @param new_spacing: The desired output spacing (default is [1.0, 1.0, 1.0]).
    @return: The resampled 3D NumPy array.
    """

    log.debug_highlight("Resampling image...")

    # Get the original image spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    log.debug("Original spacing: " + str(original_spacing))
    log.debug("Original size: " + str(original_size))
    
    # Convert image to a numpy array to check the number of unique values
    image_array = sitk.GetArrayFromImage(image)
    num_unique_values = len(np.unique(image_array))

    log.debug("Number of unique values: " + str(num_unique_values))
    # -----------------------------------------------------------------------------------------------

    # Determine if the input is an image or segmentation:

    # If the input is identified as a typical image (more than 10 unique values),
    # use B-spline for x and y, Linear for z
    if num_unique_values > 10:

        log.debug("It's an image. Resampling using B-spline interpolation for x and y, and linear interpolation for z...")

        # Calculates the new size in the x and y dimensions
        new_size_xy = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size[0:2], original_spacing[0:2], new_spacing[0:2])]
        new_size = [new_size_xy[0], new_size_xy[1], original_size[2]]

        # Configures the resampler for the xy-plane using B-spline interpolation
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing([new_spacing[0], new_spacing[1], original_spacing[2]])
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkBSpline)

        # Resamples the image in the xy-plane
        resampled_image_xy = resampler.Execute(image)

        # Resample in z dimension using linear interpolation
        new_size_z = [new_size_xy[0], new_size_xy[1], int(round(original_size[2] * original_spacing[2] / new_spacing[2]))]
        
        # Configures the resampler for the z-dimension using linear interpolation
        resampler.SetSize(new_size_z)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetInterpolator(sitk.sitkLinear)

        # Resamples the image in the z-dimension
        resampled_image_xyz = resampler.Execute(resampled_image_xy)

        # Returns the resampled image as a NumPy array
        return sitk.GetArrayFromImage(resampled_image_xyz)
    # -----------------------------------------------------------------------------------------------

    # If the input is identified as a segmentation (10 or fewer unique values),
    # use distance map interpolation to maintain label boundaries accurately
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