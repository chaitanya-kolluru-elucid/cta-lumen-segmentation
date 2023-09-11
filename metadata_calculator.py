import glob
import nibabel as nib
import glob
import numpy as np
import os
from tqdm import tqdm

def check_orientation(ct_image, ct_arr):
    """
    Check the NIfTI orientation, and flip to  'RPS' if needed.
    :param ct_image: NIfTI file
    :param ct_arr: array file
    :return: array after flipping
    """
    x, y, z = nib.aff2axcodes(ct_image.affine)
    if x != 'R':
        ct_arr = np.flip(ct_arr, axis=0)
    if y != 'A':
        ct_arr = np.flip(ct_arr, axis=1)
    if z != 'S':
        ct_arr = np.flip(ct_arr, axis=2)
    return ct_arr

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
        nifti_image = nib.load(image_list[k])
        nifti_mask = nib.load(masks_list[k])

        image_data = nifti_image.get_fdata()
        mask_data = nifti_mask.get_fdata()

        image_data = check_orientation(nifti_image, image_data)
        mask_data = check_orientation(nifti_mask, mask_data)        

        assert image_data.shape == mask_data.shape

        fg_intensity_vals.append(image_data[mask_data > 0])

    fg_intensity_vals = np.concatenate([i_vals for i_vals in fg_intensity_vals])
    
    fg_intensity_metrics =  [np.mean(fg_intensity_vals), np.std(fg_intensity_vals), np.percentile(fg_intensity_vals, 99.5), np.percentile(fg_intensity_vals, 0.5)]

    return tuple(median_pixel_spacing), fg_intensity_metrics