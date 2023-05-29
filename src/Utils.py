import numpy as np 
import pydicom
import os
import scipy.ndimage
from pydicom import dcmread
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from scipy.ndimage import zoom


def extract_masks(pixel_array, threshold=40):
    '''
        Extract mask from a dicom file in RGB and annotations in colors
        since for grey colors the RGB channels are equal, calculating the difference for the 3 channels
        
    '''
    # Calculate the maximum and minimum RGB channel values for each pixel
    img = pixel_array.copy()
    max_vals = np.max(img, axis=2)
    min_vals = np.min(img, axis=2)

    # Check for pixels where the difference between maximum and minimum values is less than or equal to 2
    diff = max_vals - min_vals
    equal_diff_pixels = diff <= threshold

    # Set the equal difference pixels to black (0, 0, 0)
    img[equal_diff_pixels] = [0, 0, 0]
    binary_mask = np.where(img > 0, 255, 0).astype(np.uint8)

    return binary_mask

def generate_masks(tomask_path, save_path):
    for folder in os.listdir(tomask_path):
        folder_path = tomask_path+folder+"/"
        for file in os.listdir(folder_path):
            file_path = folder_path+file
            
            save_dir = os.path.join(save_path, folder)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            saved_filename = os.path.join(save_path, folder,file.replace('.dcm', '_mask.png'))
            if not os.path.exists(saved_filename):
                try: 
                    ds = dcmread(file_path)
                    pixel_array = ds.pixel_array
                    mask = extract_masks(pixel_array)
                    cv2.imwrite(saved_filename, mask)
                except:
                    print(f'Error loading {file_path}')


def transform_to_hu(ds):
    '''
    Return an ndarray of the the pixel values transformed to Hounsfield Units (HU)
    using the Rescale Intercept and Rescale Slope values from the DICOM metadata.

    Parameters:
    -----------
    ds: Pydicom object
    '''
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    image = ds.pixel_array
    hu_image = image * slope + intercept

    return hu_image

def set_window(img, hu=[-800.,1200.]):
    '''
    Performs windowing on the image by mapping the pixel values to a normalized range 
    between 0 and 1 based on the specified HU range.

    Parameters: 
    ----------
    img: array_like
        The image to apply windowing
    hu: array_like
        The range of the windowing 

    Returns: 
    newimg: array_like
        The transformed image array

    '''
    window = np.array(hu)
    newimg = (img-window[0]) / (window[1]-window[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def zero_center(hu_image):
    hu_image = hu_image - np.mean(hu_image)
    return hu_image


def resize_dcm(ds, size = (256, 256)):
    # Extract the pixel data
    pixel_data = ds.pixel_array
    # Define the desired output size
    output_size = (256, 256)  # Specify the desired width and height

    # Resize the pixel data using the zoom function from scipy.ndimage
    resized_data = zoom(pixel_data, (output_size[0] / pixel_data.shape[0], output_size[1] / pixel_data.shape[1]))

    # Update DICOM attributes
    ds.Rows, ds.Columns = output_size
    ds.PixelData = resized_data.astype(np.uint16).tobytes()
    ds.RescaleIntercept = 0  # Reset the rescale intercept to 0 if necessary

    plt.imshow(ds.pixel_array, cmap='gray')
    return ds

def crop_dicom(ds, output_path, x_start, x_end, y_start, y_end):
    # Load DICOM image
    dicom_image = ds.copy()

    # Get pixel data
    pixel_array = dicom_image.pixel_array

    # Crop the image
    cropped_array = pixel_array[y_start:y_end, x_start:x_end]

    # Update DICOM metadata
    dicom_image.Rows, dicom_image.Columns = cropped_array.shape
    dicom_image.PixelData = cropped_array.tobytes()

    return dicom_image