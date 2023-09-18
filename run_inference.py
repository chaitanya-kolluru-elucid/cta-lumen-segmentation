import argparse
import torch
import os
import glob
import pickle

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, NormalizeIntensityd, Orientationd, Spacingd, Invertd
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import nibabel as nib
from monai.inferers import sliding_window_inference
from scipy.spatial.distance import dice
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def run_inference(results_dir, test_images_dir, test_preds_dir, training_args):

    if not os.path.exists(test_preds_dir):
        os.makedirs(test_preds_dir)

    # Get the GPU device
    device = torch.device("cuda:0")

    # Load model architecture
    model = torch.load(os.path.join(results_dir, 'model_architecture.pt'))
    model = model.to(device)

    # Load the weights checkpoint
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_val_metric_model.pt')))
    model.eval()

    # Create appropriate dictionary lists
    test_images = sorted(glob.glob(os.path.join(test_images_dir, '*.nii.gz')))
    test_dicts = [{"image": image} for image in test_images]

    # Load the image metadata from the training set
    with open(os.path.join(results_dir, 'median_pixel_spacing.pkl'), 'rb') as f:
        median_pixel_spacing = pickle.load(f)

    with open(os.path.join(results_dir, 'fg_intensity_metrics.pkl'), 'rb') as f:
        fg_intensity_metrics = pickle.load(f)

    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=fg_intensity_metrics[3],
                a_max=fg_intensity_metrics[2],
                b_min=fg_intensity_metrics[3],
                b_max=fg_intensity_metrics[2],
                clip=True,
            ),
            NormalizeIntensityd(
                keys=["image"],
                subtrahend = fg_intensity_metrics[0],
                divisor = fg_intensity_metrics[1],
            ),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=median_pixel_spacing, mode=("bilinear")),
        ]
        )

    test_ds = Dataset(data=test_dicts, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

    post_transforms = Compose(
        [
        Invertd(
        keys="pred",
        transform=test_transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=True,
        to_tensor=True,
        ),
        ]
        )

    print('Running inference on ' + str(len(test_images)) + ' cases.')
    with torch.no_grad():
        for test_data in test_loader:
            test_inputs = test_data["image"].to(device)
            roi_size = training_args.train_roi_size
            sw_batch_size = 4
            test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

            volume = torch.argmax(test_data[0]["pred"], dim=0).detach().cpu()

            image = nib.load(test_inputs.meta["filename_or_obj"][0])

            nifti_volume = nib.Nifti1Image(volume, image.affine, image.header)
            filename = test_inputs.meta["filename_or_obj"][0].split('/')[-1]

            pred_filename = filename.split('.')[0] + '_seg_out.nii.gz'

            nib.save(nifti_volume, os.path.join(test_preds_dir, pred_filename))

def sort_key(x):
    splits = x.split('/')[-1]
    return len(splits), splits

def calculate_metrics(test_preds_dir, test_labels_dir, num_label_classes = 3, compute_cm = False):

        predsList = sorted(glob.glob(os.path.join(test_preds_dir, '*.nii.gz')), key = sort_key)
        labelsList = sorted(glob.glob(os.path.join(test_labels_dir, '*.nii.gz')), key = sort_key)

        assert len(predsList) == len(labelsList)

        dice_vals = np.zeros((num_label_classes,len(predsList)))

        confusion_matrix_var = np.zeros((num_label_classes, num_label_classes, len(predsList)))

        print('Prediction image list: ')
        print(predsList)

        print('Computing metrics')

        for k in tqdm(range(len(predsList))):

            current_prediction = nib.load(predsList[k]).get_fdata()
            current_label = nib.load(labelsList[k]).get_fdata()

            current_prediction = np.reshape(current_prediction, (-1,))
            current_label = np.reshape(current_label, (-1,))

            for label_class in range(num_label_classes):
                prediction_mask = current_prediction == label_class
                label_mask = current_label == label_class

                dice_vals[label_class, k] = 1 - dice(prediction_mask, label_mask)

            if compute_cm:
                if num_label_classes == 2:
                    current_label[current_label == 2] = 0
                    current_prediction[current_prediction == 2] = 0

                confusion_matrix_var[:,:, k] = confusion_matrix(current_label, current_prediction)

        return dice_vals, confusion_matrix_var

# Hausdorff distance
# hd += compute_hausdorff_distance(y_pred = val_outputs[0], y = val_labels[0], include_background=False, distance_metric='euclidean', spacing=[0.40519333, 0.40519333, 0.625])

if __name__ == '__main__':

    num_label_classes = 2

    # Parse user specified arguments
    parser = argparse.ArgumentParser(description='Run inference using a segmentation model for CTA images.')
    parser.add_argument('-test_images_dir', type=str, default='crop_imagesTs_asoca', help='Path to the test images directory.')
    parser.add_argument('-test_labels_dir', type=str, default='crop_labelsTs_asoca', help='Path to the test labels directory.')
    parser.add_argument('-model_run_datetime', type=str, default='17092023_152632', help='Date time string that is the name of the results folder to use as the model for this inference run.')
    parser.add_argument('-compute_confusion_matrix', type=bool, default=False, help='Flag to compute confusion matrix')

    args = parser.parse_args()

    args.train_results_folder = os.path.join('./results', args.model_run_datetime)
    args.test_preds_dir = os.path.join('./segPreds', args.test_images_dir, args.model_run_datetime)
    args.test_images_dir = os.path.join('./data', args.test_images_dir)
    args.test_labels_dir = os.path.join('./data', args.test_labels_dir)

    # Get training arguments from the results directory
    with open(os.path.join(args.train_results_folder, 'post_training_args.pkl'), 'rb') as f:
        training_args = pickle.load(f)

    # Run inference
    #run_inference(results_dir=args.train_results_folder, test_images_dir=args.test_images_dir, test_preds_dir=args.test_preds_dir, training_args=training_args)

    # Calculate metrics (Dice, ASD, confusion matrix)
    dice_metric, confusion_matrix_var = calculate_metrics(test_preds_dir= args.test_preds_dir, test_labels_dir = args.test_labels_dir, num_label_classes = num_label_classes, compute_cm = args.compute_confusion_matrix)

    # Print dice values
    print('Dice metric:')
    print(dice_metric)

    print('Mean Dice value across classes:')
    print(np.mean(dice_metric, axis=1))

    print('Overall confusion matrix')
    print(np.sum(confusion_matrix_var, axis=2))
