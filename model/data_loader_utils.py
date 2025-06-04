

from concurrent.futures import Future, ProcessPoolExecutor
import gzip
import hashlib
import os
import pickle
import shutil
import sys
import time
import traceback
import uuid
import nibabel
import numpy as np
from scipy import ndimage
import torch

from .const import *

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Insert the parent directory at the beginning of sys.path
sys.path.insert(0, parent_dir)
import utils

def calculate(root_dir, current_patient_images_label: list):
    images_list = []
    patient_images = current_patient_images_label[:-1]
    patient_label = current_patient_images_label[-1]

    cache_count = 0
    total_count = len(patient_images)

    for image_path in patient_images:
        file_path: str = os.path.join(root_dir, image_path)
        hash_str = hashlib.sha256(file_path.encode('utf-8')).hexdigest()
        cache_path = os.path.join(CACHE_SINGLE_PATH, hash_str)
        cache_path_healthy = cache_path + '.healthy'

        if os.path.exists(cache_path):
            os.remove(cache_path)

        if os.path.exists(cache_path_healthy):
            cache_count += 1
            with gzip.open(cache_path_healthy, "rb") as file:
                image_data_tensor = pickle.load(file)
        else:
            use_temp_file = not file_path.endswith('.nii')
            if use_temp_file:
                temp_file_path = os.path.join(TEMP_PATH, uuid.uuid4().hex + '.nii')
                shutil.copy(file_path, temp_file_path)
            else:
                temp_file_path = file_path

            neuroimage = nibabel.load(temp_file_path)  # type: ignore # Loads proxy image
            # Extract the N-D array containing the image data from the nibabel image object
            image_data = neuroimage.get_fdata()  # type: ignore # Retrieves array data
            # Resize and interpolate image
            if use_temp_file:
                os.remove(temp_file_path)
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

            with gzip.open(cache_path, "wb", compresslevel=1) as file:
                image_dict_bytes = pickle.dumps(image_data_tensor)
                size_before_mb = len(image_dict_bytes) / (1024**2)
                file.write(image_dict_bytes)

            os.rename(cache_path, cache_path_healthy)
            size_after_mb = os.path.getsize(cache_path_healthy) / (1024**2)

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

    return image_dict, cache_count, total_count

def get(
    root_dir,
    current_patient_images_label: list,
    index: int,
    all_items_count: int,
    clear=False,
):
    # print(f'\t*start get: {index}')
    time0 = time.time()
    image_dict, cache_count, total_count = calculate(
        root_dir,
        current_patient_images_label,
    )
    time0 = time.time() - time0
    print(
        f'\t * got index: {index}/{all_items_count}, used cache: {cache_count}/{total_count}, took {time0:.3f}s'
    )

    # if cache_count == 0:
    #     print(
    #         f'\t * got: {index}, using_cache: {using_cache}, size from {size_before_mb:.2f} to {size_after_mb:.2f}, took {time0:.3f}s'
    #     )
    #     if not using_cache and time0 < 2:
    #         patient_images = current_patient_images_label[:-1]
    #         for image_path in patient_images:
    #             size_mb = os.path.getsize(image_path) / (1024**2)
    #             print(f'\t\t image:\n\t\t  {os.path.basename(image_path)}')
    #             print(f'\t\t size: {size_mb:.3f} MB')
   
    if clear:
        utils.clear()
        return None
    
    return image_dict

def cache_all_multiprocess(root_dir, data_array):
    print('* start caching all images...')
    executor = ProcessPoolExecutor(6)
    futures: list[Future] = []

    for index in range(len(data_array)):
        current_patient_images_label = data_array[index]
        future = executor.submit(
            get,
            root_dir,
            current_patient_images_label,
            index,
            len(data_array),
            True,
        )
        futures.append(future)

    print('\ttasks submitted, waiting for them to finish')
    for index, future in enumerate(futures):
        try:
            future.result()
        except Exception as e:
            traceback.print_exception(e)
        # print(f'\r\t* {index+1}/{len(futures)} done!', end='')
        if index % 20 == 0:
            utils.clear()

    print(f'\n\tcaching done!')

    executor.shutdown(True)