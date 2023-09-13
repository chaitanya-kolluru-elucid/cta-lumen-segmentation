import SimpleITK as sitk
from loguru import logger
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm 
import os
from whs_inference import run_pred_monai
import glob

def crop_image(cta_image_path, whs_path, crop_image_save_path, annotation_image_path = None, crop_annotation_save_path = None ):

    cta_reader = sitk.ImageFileReader()
    cta_reader.SetFileName(cta_image_path)
    cta_reader.LoadPrivateTagsOn()
    cta_reader.ReadImageInformation()

    logger.info("----------CTA Image Info---------:")
    logger.info("Image Size: {0}".format(cta_reader.GetSize()))
    logger.info("Image Physical Origin: {0}".format(cta_reader.GetOrigin()))
    logger.info("Image Spacing: {0}".format(cta_reader.GetSpacing()))
    logger.info("Image PixelType: {0}".format(sitk.GetPixelIDValueAsString(cta_reader.GetPixelID())))
    logger.info("Image Direction: {0}".format(cta_reader.GetDirection()))

    cta = cta_reader.Execute()

    if annotation_image_path is not None:
        annotation_reader = sitk.ImageFileReader()
        annotation_reader.SetFileName(annotation_image_path)
        annotation_reader.LoadPrivateTagsOn()
        annotation_reader.ReadImageInformation()

        logger.info("----------Annotation Image Info---------:")
        logger.info("Image Size: {0}".format(annotation_reader.GetSize()))
        logger.info("Image Physical Origin: {0}".format(annotation_reader.GetOrigin()))
        logger.info("Image Spacing: {0}".format(annotation_reader.GetSpacing()))
        logger.info("Image PixelType: {0}".format(sitk.GetPixelIDValueAsString(annotation_reader.GetPixelID())))
        logger.info("Image Direction: {0}".format(annotation_reader.GetDirection()))

        annotation_image = annotation_reader.Execute()

        # Resample the annotations to ensure that they are on the same grid (origin, spacing, direction)
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(cta.GetSpacing())    
        resampler.SetOutputDirection(cta.GetDirection())
        resampler.SetOutputOrigin(cta.GetOrigin())
        resampler.SetSize(cta.GetSize())

        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)

        annotation_image = resampler.Execute(annotation_image)

    whs_reader = sitk.ImageFileReader()
    whs_reader.SetFileName(whs_path)
    whs_reader.LoadPrivateTagsOn()
    whs_reader.ReadImageInformation()

    logger.info("----------WHS Image Info---------:")
    logger.info("Image Size: {0}".format(whs_reader.GetSize()))
    logger.info("Image Physical Origin: {0}".format(whs_reader.GetOrigin()))
    logger.info("Image Spacing: {0}".format(whs_reader.GetSpacing()))
    logger.info("Image PixelType: {0}".format(sitk.GetPixelIDValueAsString(whs_reader.GetPixelID())))
    logger.info("Image Direction: {0}".format(whs_reader.GetDirection()))

    whs_mask = nib.load(whs_path)

    logger.info("----------WHS Image Info from Nibabel---------:")
    logger.info("Image Size: {0}".format(whs_mask.header.get_data_shape()))
    logger.info("Image Spacing: {0}".format(whs_mask.header.get_zooms()))
    logger.info("Image PixelType: {0}".format(whs_mask.header.get_data_dtype()))
    logger.info("Image Direction: {0}".format(whs_mask.affine))

    whs_mask_array = np.array(whs_mask.dataobj)

    # MONAI Whole heart segmentation (TotalSegmentator)
    # "background": "0",
    # "Aorta": "7",
    # "Myocardium_of_LV": "44",
    # "LA": "45",
    # "LV": "46",
    # "RA": "47",
    # "RV": "48",
    # "PAT": "49"

    seg_rv = np.where(whs_mask_array == 48)
    seg_la = np.where(whs_mask_array == 45)
    seg_myo = np.where(whs_mask_array == 44)    
    seg_ra = np.where(whs_mask_array == 47)

    # Check direction ijk to ras matrix (since we read with nibabel)
    direction_matrix = np.sign(whs_mask.affine[:3, :3])
    i_to_r = direction_matrix[0,0] > 0
    j_to_a = direction_matrix[1,1] > 0

    # Get left, right, anterior and posterior extremes based on masks
    left_coordinate = np.min(seg_myo[0]) - 60 if i_to_r else np.max(seg_myo[0]) + 60
    right_coordinate = np.max(seg_ra[0]) + 60 if i_to_r else np.min(seg_ra[0]) - 60

    anterior_coordinate = max(np.max(seg_myo[1]), np.max(seg_rv[1])) + 60 if j_to_a else min(np.min(seg_myo[1]), np.min(seg_rv[1])) - 60
    posterior_coordinate = np.min(seg_la[1]) - 60 if j_to_a else np.max(seg_la[1]) + 60

    # Ensure all coordinates are within the size of the image
    left_coordinate = np.clip(left_coordinate, 0, whs_mask.header.get_data_shape()[0] - 1)
    right_coordinate = np.clip(right_coordinate, 0, whs_mask.header.get_data_shape()[0] - 1)
    
    anterior_coordinate = np.clip(anterior_coordinate, 0, whs_mask.header.get_data_shape()[1] - 1)
    posterior_coordinate = np.clip(posterior_coordinate, 0, whs_mask.header.get_data_shape()[1] - 1)

    # Find min max coordinates for each axis
    min_x_index = min(left_coordinate, right_coordinate)
    max_x_index = max(left_coordinate, right_coordinate)

    min_y_index = min(anterior_coordinate, posterior_coordinate)
    max_y_index = max(anterior_coordinate, posterior_coordinate)
    
    # Create and save cropped CTA
    cropped_cta = cta[min_x_index:max_x_index, min_y_index:max_y_index, :]
    sitk.WriteImage(cropped_cta, crop_image_save_path)

    if annotation_image_path is not None:

        cropped_annotations = annotation_image[min_x_index:max_x_index, min_y_index:max_y_index, :]

        if crop_annotation_save_path is None:
            print('Provide a path to save the cropped annotations')
            return
        
        sitk.WriteImage(cropped_annotations, crop_annotation_save_path)

def generate_crop_images(nifti_image_filelist, annotation_filelist = None, whs_paths = None):

    if annotation_filelist is not None:
        assert len(nifti_image_filelist) == len(annotation_filelist)

    assert len(whs_paths) == len(nifti_image_filelist)

    print('Cropping images')

    for k in tqdm(range(len(nifti_image_filelist))):

        parent_dir = nifti_image_filelist[k].split('/')[-2]
        folder_to_save_images = os.path.abspath(os.path.join(os.path.dirname(nifti_image_filelist[k]), '..', 'crop_' + parent_dir))

        if not os.path.exists(folder_to_save_images):
            os.makedirs(folder_to_save_images)

        image_stack_name = nifti_image_filelist[k].split('/')[-1]
        crop_image_save_path = os.path.join(folder_to_save_images, image_stack_name)

        annotation_image_path = None
        crop_annotation_save_path = None

        if annotation_filelist is not None:
            parent_dir = annotation_filelist[k].split('/')[-2]
            annotation_image_path = annotation_filelist[k]
            folder_to_save_annotations = os.path.abspath(os.path.join(os.path.dirname(annotation_filelist[k]), '..', 'crop_' + parent_dir))

            if not os.path.exists(folder_to_save_annotations):
                os.makedirs(folder_to_save_annotations)

            annotation_stack_name = annotation_filelist[k].split('/')[-1]

            crop_annotation_save_path = os.path.join(folder_to_save_annotations, annotation_stack_name)

        crop_image(nifti_image_filelist[k], whs_paths[k], crop_image_save_path, annotation_filelist[k], crop_annotation_save_path = crop_annotation_save_path )

if __name__ == '__main__':
    
    # Provide folders for raw CTA images (as .nii.gz) and location to save whole heart segmentations
    nifti_image_dir = './data/imagesTr'
    annotations_dir = './data/labelsTr'
    whs_pred_dir = os.path.join('./whsPreds', nifti_image_dir.split('/')[-1])

    # MONAI - whole body CT segmentation model from Zoo
    #whs_segmentation_bundle_path = './monai_whole_body_segmentation/'
    #run_pred_monai(model_folder=whs_segmentation_bundle_path, input_dir = nifti_image_dir, output_dir=whs_pred_dir)

    nifti_image_filelist = sorted(glob.glob(os.path.join(nifti_image_dir , '*.nii.gz')))
    annotations_filelist = sorted(glob.glob(os.path.join(annotations_dir , '*.nii.gz')))

    assert len(nifti_image_filelist) == len(annotations_filelist)
    whs_paths = sorted(glob.glob(os.path.join(whs_pred_dir, '*.nii.gz')))

    # Crop images based on the segmentation result
    generate_crop_images(nifti_image_filelist, annotation_filelist = annotations_filelist, whs_paths = whs_paths)