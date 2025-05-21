import gzip
import os
import sys
import time  # possibly don't need
import numpy as np
import pickle
import hashlib
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

import nibabel as nib
from scipy import ndimage

# Dimensions of neuroimages after resizing
STANDARD_DIM1 = int(200 * 0.85)
STANDARD_DIM2 = int(200 * 0.85)
STANDARD_DIM3 = int(150 * 0.85)
DIMESIONS = (STANDARD_DIM1, STANDARD_DIM2, STANDARD_DIM3)

# Maximum number of images per patient
MAX_NUM_IMAGES = 4

CACHE_PATH = 'cache'
os.makedirs(CACHE_PATH, exist_ok=True)


class MRIData(Dataset):
    """
    MRI data
    The dictionaries AD_Img_Dict.pkl and MCI_Img_Dict.pkl contain key-value
      pairs of the following:
      key:      subject ID
      value:    paths to relevant images (i.e. multiple images per subject)
    These dictionaries are converted to arrays and passed into this dataset,
    where the paths will be accessed and their neuroimages processed into tensors.
    """

    def __init__(self, root_dir, data_array):
        """
        Args:
            root_dir (string): directory of all the images
            data_array (list): array that contains one [key, [value]*] entry for each patient,
                               where key:       subject ID
                                     value:     paths to patient's MRI .nii neuroimages
        """
        self.root_dir = root_dir
        self.data_array = data_array

    def __len__(self):
        """
        Returns length of dataset       (required by DataLoader)
        """
        return len(self.data_array)  # the number of patients in the dataset

    def __getitem__(self, index):
        """
        Allows indexing of dataset      (required by DataLoader)
        Returns a tensor that contains the patient's MRI neuroimages and their diagnoses (AD or MCI)
        """
        # Get current_patient, where [0] is their ID and [1] is their list of images
        current_patient_images_label = self.data_array[index]
        # List to store the individual image tensors
        # The last element in the current patient's array is the classification
        # print(patient_label)
        # For each image path, process the .nii image using nibabel
        image_dict = self.get(current_patient_images_label, index)

        return image_dict

    def cache0(self, current_patient_images_label: list):
        images_list = []
        patient_images = current_patient_images_label[:-1]
        patient_label = current_patient_images_label[-1]
        for image_path in patient_images:
            # print(image_path) #FIXME: delete this
            file_name = os.path.join(self.root_dir, image_path)
            # file_name = r'G:\university\arshad\payan_name\open_source_projects\Alzheimers-DL-Network\data_sample\Data\MCI_to_AD\022_S_1394\MIDAS_Whole_Brain_Mask\2007-05-29_14_24_28.0\S34317\ADNI_022_S_1394_MR_MIDAS_Whole_Brain_Mask_Br_20120814182221239_S34317_I323573.nii'
            neuroimage = nib.load(file_name)  # Loads proxy image
            # Extract the N-D array containing the image data from the nibabel image object
            image_data = neuroimage.get_fdata()  # Retrieves array data
            # Resize and interpolate image
            image_size = image_data.shape  # Store dimensions of N-D array
            current_dim1 = image_size[0]
            current_dim2 = image_size[1]
            current_dim3 = image_size[2]
            # Calculate scale factor for each direction
            scale_factor1 = STANDARD_DIM1 / float(current_dim1)
            scale_factor2 = STANDARD_DIM2 / float(current_dim2)
            scale_factor3 = STANDARD_DIM3 / float(current_dim3)
            # Resize image (spline interpolation)
            image_data = ndimage.zoom(
                image_data, (scale_factor1, scale_factor2, scale_factor3)
            )
            # print("Resize success") #FIXME: delete this
            # Convert image data to a tensor
            image_data_tensor = torch.Tensor(image_data)
            images_list.append(image_data_tensor)

        # Add padding to make all final tensors the same size
        num_images = len(images_list)
        while len(images_list) < MAX_NUM_IMAGES:
            padding_array = np.zeros((STANDARD_DIM1, STANDARD_DIM2, STANDARD_DIM3))
            padding_tensor = torch.Tensor(padding_array)
            images_list.append(padding_tensor)

        if len(images_list) > MAX_NUM_IMAGES:
            print(
                "Error: More than 10 images for one individual patient. Update MAX_NUM_IMAGES in data_loader.py"
            )

        # Convert the list of individual image tensors to a tensor itself
        images_tensor = torch.stack(images_list, dim=0)

        # Return a dictionary with the images tensor and the label
        image_dict = {
            'images': images_tensor,
            'label': patient_label,
            'num_images': num_images,
        }

        return image_dict

    def get(self, current_patient_images_label: list, index: int):
        # print(f'\t*start get: {index}')
        time0 = time.time()
        using_cache=False

        hash_str = hashlib.sha256(
            pickle.dumps(current_patient_images_label)
        ).hexdigest()
        cache_file_path = os.path.join(CACHE_PATH, hash_str)
        cache_health_file_path = os.path.join(CACHE_PATH, f'{hash_str}.healthy')
        if os.path.exists(cache_file_path) and os.path.exists(cache_health_file_path):
            using_cache = True
            with gzip.open(cache_file_path, "rb") as file:
                image_dict = pickle.load(file)
        else:
            using_cache = True = False
            image_dict = self.cache0(current_patient_images_label)
            with gzip.open(cache_file_path, "wb", compresslevel=3) as file:
                pickle.dump(image_dict, file)
            open(cache_health_file_path, 'w').close()
        
        
        time0 = time.time() - time0
        print(f'\t\t*got: {index}, took {time0:.3f}s')
        if not using_cache and time0 < 2:
            patient_images = current_patient_images_label[:-1]
            for image_path in patient_images:
                size_mb = os.path.getsize(image_path) / (1024 ** 2)
                print(f'\t\t image:\n\t\t  {os.path.basename(image_path)}')
                print(f'\t\t size: {size_mb:.3f} MB')
        return image_dict
