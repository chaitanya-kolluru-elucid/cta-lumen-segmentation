import glob
import nibabel as nib
import glob
import numpy as np
import os
from tqdm import tqdm

def get_training_metadata(image_list, masks_list):

    pixel_spacing = np.zeros((len(image_list), 3))

    print('Getting training metadata')

    for k in range(len(image_list)):
        nii = nib.load(image_list[k])
        sx, sy, sz = nii.header.get_zooms()

        pixel_spacing[k,:] = [sx, sy, sz]

    median_pixel_spacing = np.median(pixel_spacing, axis = 0)

    fg_intensity_vals = []

    for k in tqdm(range(len(image_list))):
        image_data = nib.load(image_list[k]).get_fdata()
        mask_data = nib.load(masks_list[k]).get_fdata()

        assert image_data.shape == mask_data.shape

        fg_intensity_vals.append(image_data[mask_data > 0])

    fg_intensity_vals = np.concatenate([i_vals for i_vals in fg_intensity_vals])
    
    fg_intensity_metrics =  [np.mean(fg_intensity_vals), np.std(fg_intensity_vals), np.percentile(fg_intensity_vals, 99.5), np.percentile(fg_intensity_vals, 0.5)]

    return tuple(median_pixel_spacing), fg_intensity_metrics