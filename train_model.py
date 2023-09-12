from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    RandAffined,
    NormalizeIntensityd,
    Lambda,
    MapTransform,
    RandCropByLabelClassesd,
    RandGaussianNoised,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandHistogramShiftd,
    RandFlipd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, SegResNet, AttentionUnet, SegResNetVAE
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, get_confusion_matrix
from monai.losses import DiceLoss, DiceCELoss,GeneralizedDiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, PersistentDataset
from monai.config import print_config
from monai.metrics import compute_hausdorff_distance

import argparse
import wandb
import torch
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from datetime import datetime
import math
from sklearn.metrics import confusion_matrix
import metadata_calculator
import pickle

from losses import TI_Loss, cldice

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# Print MONAI config
print_config()

def training_run(args, results_dir):

    num_classes = args.num_classes

    # Get list of training images
    images = sorted(glob.glob(os.path.join(args.data_dir, args.images_dir, "*.nii.gz")))

    # Get list of training labels
    labels = sorted(glob.glob(os.path.join(args.data_dir, args.labels_dir, '*' + args.mask_suffix + '.nii.gz')))

    # Set random seed and shuffle image list
    random.seed(3)
    random.shuffle(images)

    # Set random seed and shuffle label list
    random.seed(3)
    random.shuffle(labels)

    if not len(images) == len(labels):
        print('Please verify that the number of images and labels are the same. Exiting.')
        return
    
    num_cases = len(images)
    num_validation_cases = math.floor(0.2 * num_cases)

    # Create data dictionaries, training for 10 cases, validation on 2 cases
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]
    train_files, val_files = data_dicts[:-num_validation_cases], data_dicts[-num_validation_cases:]

    # Get training metadata, used for image normalization, save to pickle files, useful for inference
    calculate_metadata = False
    if os.path.exists(os.path.join(args.metadata_dir, 'median_pixel_spacing.pkl')):
        with open(os.path.join(args.metadata_dir, 'median_pixel_spacing.pkl'), 'rb') as f:
            median_pixel_spacing = pickle.load(f)

    else:
        calculate_metadata = True

    if os.path.exists(os.path.join(args.metadata_dir, 'fg_intensity_metrics.pkl')):
        with open(os.path.join(args.metadata_dir, 'fg_intensity_metrics.pkl'), 'rb') as f:
            fg_intensity_metrics = pickle.load(f)
    else:
        calculate_metadata = True

    if calculate_metadata:
        median_pixel_spacing, fg_intensity_metrics = metadata_calculator.get_training_metadata(images, labels)
    
    with open(os.path.join(results_dir, 'median_pixel_spacing.pkl'), 'wb') as f:
        pickle.dump(median_pixel_spacing, f)

    with open(os.path.join(results_dir, 'fg_intensity_metrics.pkl'), 'wb') as f:
        pickle.dump(fg_intensity_metrics, f)

    set_determinism(seed=0)

    # Transforms for the training dataset
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
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
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=median_pixel_spacing, mode=("bilinear", "nearest")),
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=args.train_roi_size,
                num_classes=num_classes,
                ratios=args.crop_ratios,
                num_samples=int(np.sum(args.crop_ratios)), 
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
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
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=median_pixel_spacing, mode=("bilinear", "nearest")),
        ]
    )

    # Save a sample from the validation dataset to confirm mask looks accurate, save to results directory
    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image, label = (check_data["image"][0][0], check_data["label"][0][0])

    nifti_image = nib.Nifti1Image(((image * fg_intensity_metrics[1]) + fg_intensity_metrics[0]).astype(np.uint16) , affine=np.eye(4))
    nifti_label = nib.Nifti1Image(label.astype(np.uint16), affine=np.eye(4))

    nib.save(nifti_image, os.path.join(results_dir, 'Validation data image check.nii.gz'))
    nib.save(nifti_label, os.path.join(results_dir, 'Validation data label check.nii.gz'))

    # Create training and validation data loaders
    train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir = './cache')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=6)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=os.cpu_count())

    # Get the GPU device
    device = torch.device("cuda:0")

    # Get network architecture
    if args.architecture == 'UNet':
        model = UNet(spatial_dims=3, in_channels=1, out_channels=num_classes, channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2, norm=Norm.BATCH, dropout=0.2).to(device)

    elif args.architecture == 'SegResNet':
        model = SegResNet(spatial_dims=3, in_channels=1, out_channels=num_classes, norm=Norm.BATCH).to(device)

    elif args.architecture == 'AttentionUnet':
        model = AttentionUnet(spatial_dims=3, in_channels=1, out_channels=num_classes, channels=[16, 32, 64, 128, 256], strides=(2, 2, 2, 2, 2), dropout=0.0).to(device)

    else:
        print('Model architecture not found, ensure that the architecture is either UNet, SegResNet or AttentionUnet. Exiting.')
        return
    
    # Save the model architecture to the results folder
    torch.save(model, os.path.join(results_dir, 'model_architecture.pt'))
    
    # Get loss function
    if args.loss == 'DiceCE':
        loss_function = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, batch=False, ce_weight=torch.Tensor(args.ce_weights).to(device)) 

    elif args.loss == 'Topological':
        dice_ce_loss = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, batch=False, ce_weight=torch.Tensor(args.ce_weights).to(device)) 

        ti_loss_function = TI_Loss.TI_Loss(dim=3, connectivity=26, inclusion=[[2,1]], exclusion=[], min_thick=1)

        loss_function = lambda outputs, labels: dice_ce_loss(outputs, labels) + (1e-6) * ti_loss_function(outputs, labels)
    
    elif args.loss == 'clDice':
        loss_function = cldice.soft_dice_cldice_ce(iter_=10, num_classes = num_classes, lumen_class=1)

    else:
        print('Loss function not found, ensure that the loss is one of DiceCE, Topological or clDice. Exiting.')
        return        

    optimizer = torch.optim.Adam(model.parameters(), args.lr) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    if 'Dice' in args.metrics:
        metric = DiceMetric(include_background=False, reduction="mean") 
    else:
        print('Alternate metrics not yet implemented.Exiting')
        return

    max_epochs = args.epochs
    val_interval = 2
    best_metric = -np.Inf
    best_metric_epoch = -1
    epoch_loss_values = []
    val_loss_values = []
    metric_values = []

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)]) 
    post_label = Compose([AsDiscrete(to_onehot=num_classes)])

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            torch.cuda.empty_cache()

        train_loss /= step
        epoch_loss_values.append(train_loss)
        print(f"epoch {epoch + 1} average loss: {train_loss:.4f}")
        wandb.log({'train/loss':train_loss}, step=epoch)
        scheduler.step()

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            step = 0
            with torch.no_grad():
                for val_data in val_loader:
                    step += 1
                    val_inputs, val_labels = (val_data["image"].to(device), val_data["label"].to(device))
                    roi_size = args.train_roi_size
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
          
                    val_loss += loss_function(val_outputs, val_labels).item()
                    
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    
                    # Compute metric for current iteration
                    metric(y_pred=val_outputs, y=val_labels)

                    torch.cuda.empty_cache()

                val_loss /= step
                val_loss_values.append(val_loss)
                wandb.log({'val/loss':val_loss}, step=epoch)

                print(f'Dice values on the validation set (rows are samples, cols are classes): ', metric.get_buffer().cpu())
                
                # aggregate the final mean dice result
                val_metric = metric.aggregate().item()
                wandb.log({'val/metric':val_metric}, step=epoch)

                # reset the status for next validation round
                metric.reset()

                metric_values.append(val_metric)
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(results_dir, "best_val_metric_model.pt"))
                    print("saved new best metric model")

                print(f"current epoch: {epoch + 1} \ncurrent val mean loss: {val_loss:.4f} \ncurrent val mean dice: {val_metric:.4f} \nbest val mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")

    plt.figure("train", (12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Mean training Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)

    plt.subplot(1, 3, 2)
    plt.title("Mean validation Loss")
    x = [(i + 1)*val_interval for i in range(len(val_loss_values))]
    y = val_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)

    plt.subplot(1, 3, 3)
    plt.title("Mean validation Dice metric")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)

    plt.savefig(os.path.join(results_dir, 'Loss curves.png'))

if __name__ == '__main__':

    # Parse user specified arguments
    parser = argparse.ArgumentParser(description='Train a segmentation model for CTA images')
    parser.add_argument('-architecture', type=str, default='UNet', help='Network architecture')
    parser.add_argument('-loss', type=str, default='DiceCE', help='Network loss function')
    parser.add_argument('-epochs', type=int, default=50, help='Number of epochs for trianing')
    parser.add_argument('-lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('-metrics', nargs='*', default='Dice', type=str, help='Metrics to collect')
    parser.add_argument('-data_dir', type=str, default='./data', help='Relative path to the training/val dataset.')
    parser.add_argument('-results_dir', type=str, default='./results', help='Relative path to the results folder.')
    parser.add_argument('-images_dir', type=str, default='crop_imagesTr', help='Directory name containing training/val images')
    parser.add_argument('-labels_dir', type=str, default='crop_labelsTr', help='Directory name containing training/val labels')
    parser.add_argument('-train_roi_size', type=tuple, default=(96, 96, 96), help='Size of ROI used for training')
    parser.add_argument('-num_classes', type=int, default=3, help='Number of classes to predict, including background.')
    parser.add_argument('-mask_suffix', type=str, default='lumen-wall-mask', help='Suffix for the label file, used to select labels.')
    parser.add_argument('-crop_ratios', type=list, default=[0,2,2], help='Used to calculate probability of picking a crop with center pixel of certain class and number of crops per sample.')
    parser.add_argument('-ce_weights', type=list, default=[1,1,1], help='Weights of classes for cross entropy loss.')
    parser.add_argument('-metadata_dir', type=str, default='./metadata/hc_musc_olvz', help='Location to read the training metadata pickle files (median pixel spacing and fg intensities)')

    args = parser.parse_args()

    # Start wandb to track this run
    config = {"learning_rate": args.lr, "architecture":args.architecture, "loss":args.loss, 
              "epochs":args.epochs, "batch size":args.batch_size, "metrics":args.metrics, "roi_size":args.train_roi_size, 
              "crop ratios":args.crop_ratios, "ce weights":args.ce_weights}
    
    wandb.init(project='single-level-branching', name='initial-run-attention-unet', config=config)

    # Create a results directory for current run with date time
    results_dir = os.path.join(args.results_dir, datetime.now().strftime("%d%m%Y_%H%M%S"))

    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # Save training args to the results folder
    with open(os.path.join(results_dir, 'training_args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Start a training run
    training_run(args, results_dir)