import torch.nn as nn
import torch
import os
from monai.transforms import MapTransform
import shutil

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
    CastToTyped
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, SegResNet, AttentionUnet, SegResNetVAE, DynUNet
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
from pytz import timezone
import math
from sklearn.metrics import confusion_matrix
import metadata_calculator
import pickle

from losses import TI_Loss, cldice, dsv
import trainer
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# Print MONAI config
print_config()

class RemoveCalc(MapTransform):
    def __init__(self, keys, calc_val = 2, bg_val = 0):
        super().__init__(keys, False)
        self.keys = keys
        self.calc_val = calc_val
        self.bg_val = bg_val

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key][data[key] == self.calc_val] = self.bg_val
        return data

def plot_loss_curves(epoch_loss_values, val_loss_values, val_interval, metric_values, results_dir):

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

def post_training_run(post_train_images_dir, post_train_labels_dir, pre_train_results_dir, post_train_results_dir, args):

    # Load the model and weights
    model = torch.load(os.path.join(pre_train_results_dir, 'model_architecture.pt'))
    model.load_state_dict(torch.load(os.path.join(pre_train_results_dir, 'best_val_metric_model.pt')))

    if args.convertModelToTwoChannelOutput:

        # Change the last layer of the model to be two class segmentation instead of 3 class
        if args.architecture == 'UNet':
            model.model._modules['2']._modules['0'].conv = nn.ConvTranspose3d(32, 2, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1))
            model.model._modules['2']._modules['0']._modules['adn']._modules['N'] = nn.BatchNorm3d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            model.model._modules['2']._modules['1'].conv.unit0.conv = nn.Conv3d(2, 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        elif args.architecture == 'DynUNet':
            model.output_block.conv.conv = nn.Conv3d(32, 2, kernel_size=(1,1,1), stride=(1,1,1))
            model.deep_supervision_heads._modules['0'].conv.conv = nn.Conv3d(64, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1))
            model.deep_supervision_heads._modules['1'].conv.conv = nn.Conv3d(128, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1))
            model.deep_supervision_heads._modules['2'].conv.conv = nn.Conv3d(256, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1))
            model.skip_layers.next_layer.next_layer.next_layer.super_head.conv.conv = nn.Conv3d(256, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1))
            model.skip_layers.next_layer.next_layer.super_head.conv.conv = nn.Conv3d(128, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1))
            model.skip_layers.next_layer.super_head.conv.conv = nn.Conv3d(64, 2, kernel_size=(1, 1, 1), stride=(1, 1, 1))

        else:
            print('Alternate architectures not yet implemented for a different class number post-training. Exiting.')
            return
    
    # Get the GPU device
    device = torch.device("cuda:0")

    # Push the model to the GPU
    model = model.to(device)

    num_classes = 2

    # Get list of training images
    images = sorted(glob.glob(os.path.join(post_train_images_dir, "*.nii.gz")))

    # Get list of training labels
    labels = sorted(glob.glob(os.path.join(post_train_labels_dir, '*.nii.gz')))

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

    # Create data dictionaries, training and validatoin
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
    
    with open(os.path.join(post_train_results_dir, 'median_pixel_spacing.pkl'), 'wb') as f:
        pickle.dump(median_pixel_spacing, f)

    with open(os.path.join(post_train_results_dir, 'fg_intensity_metrics.pkl'), 'wb') as f:
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
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
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
            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
        ]
    )

    if args.removeCalcFromLabels:
        train_transforms.insert(-1, RemoveCalc(keys=["image", "label"], calc_val = 2, bg_val = 0))
        val_transforms.insert(-1, RemoveCalc(keys=["image", "label"], calc_val = 2, bg_val = 0))

    # Save a sample from the validation dataset to confirm mask looks accurate, save to results directory
    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image, label = (check_data["image"][0][0], check_data["label"][0][0])

    nifti_image = nib.Nifti1Image(((image * fg_intensity_metrics[1]) + fg_intensity_metrics[0]).astype(np.uint16) , affine=np.eye(4))
    nifti_label = nib.Nifti1Image(label.astype(np.uint16), affine=np.eye(4))

    nib.save(nifti_image, os.path.join(post_train_results_dir, 'Validation data image check.nii.gz'))
    nib.save(nifti_label, os.path.join(post_train_results_dir, 'Validation data label check.nii.gz'))

    # Create training and validation data loaders
    train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir = './cache')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=6)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=6)

    # Save the model architecture to the results folder
    torch.save(model, os.path.join(post_train_results_dir, 'model_architecture.pt'))
    
    # Get loss function
    if args.loss == 'DiceCE':
        loss_function = DiceCELoss(include_background=args.include_bg_in_loss, to_onehot_y=True, softmax=True, batch=args.dice_batch_reduction, ce_weight=torch.Tensor(args.ce_weights).to(device)) 

    elif args.loss == 'Topological':
        dice_ce_loss = DiceCELoss(include_background=args.include_bg_in_loss, to_onehot_y=True, softmax=True, batch=args.dice_batch_reduction, ce_weight=torch.Tensor(args.ce_weights).to(device)) 

        ti_loss_function = TI_Loss.TI_Loss(dim=3, connectivity=26, inclusion=[[2,1]], exclusion=[], min_thick=1)

        loss_function = lambda outputs, labels: dice_ce_loss(outputs, labels) + (1e-6) * ti_loss_function(outputs, labels)
    
    elif args.loss == 'clDice':
        loss_function = cldice.soft_dice_cldice_ce(iter_=20, num_classes = num_classes, lumen_class=1, include_bg = args.include_bg_in_loss)

    elif args.loss == 'DeepSupervisionLoss':
            loss_function = dsv(include_background=args.include_bg_in_loss, to_onehot_y=True, softmax=True, batch=args.dice_batch_reduction, ce_weight=torch.Tensor(args.ce_weights).to(device))
        
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
        print('train/loss: ', train_loss)
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
                print("val/loss: ", val_loss)

                print(f'Dice values on the validation set (rows are samples, cols are classes): ', metric.get_buffer().cpu())
                
                # aggregate the final mean dice result
                val_metric = metric.aggregate().item()                
                print("val/metric: ", val_metric)

                # reset the status for next validation round
                metric.reset()

                metric_values.append(val_metric)
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(post_train_results_dir, "best_val_metric_model.pt"))
                    print("saved new best metric model")

                print(f"current epoch: {epoch + 1} \ncurrent val mean loss: {val_loss:.4f} \ncurrent val mean dice: {val_metric:.4f} \nbest val mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")

    plot_loss_curves(epoch_loss_values, val_loss_values, val_interval, metric_values, post_train_results_dir)

if __name__ == '__main__':

    pre_train_results_dir = './results/12092023_005708'

    post_train_images_dir = './data/crop_imagesTr_mlb_in_house_plus_rsip'
    post_train_labels_dir = './data/crop_labelsTr_mlb_in_house_plus_rsip'

    # Create a results directory for current run with date time
    tz = timezone('US/Eastern')
    date_time = datetime.now(tz).strftime("%d%m%Y_%H%M%S")
    print('Creating the results directory in ./results/' + date_time)
    post_train_results_dir = os.path.join('./results', date_time)

    if not os.path.exists(post_train_results_dir):
        os.makedirs(post_train_results_dir)

    # Read training args from pre_train results folder
    with open(os.path.join(pre_train_results_dir, 'training_args.pkl'), 'rb') as f:
        args = pickle.load(f)

    # Copy the training args pkl file from pre_train to post_train folder
    shutil.copy(os.path.join(pre_train_results_dir, 'training_args.pkl'), os.path.join(pre_train_results_dir, 'training_args.pkl'))

    # Post training specific argument defaults
    args.epochs = 500
    args.batch_size = 2
    args.crop_ratios = [2, 2, 2]
    args.ce_weights = [1, 1, 1]
    args.include_bg_in_loss = True
    args.dice_batch_reduction = True

    args.convertModelToTwoChannelOutput = False
    args.removeCalcFromLabels = False

    # Save training args to the post_train results folder
    with open(os.path.join(pre_train_results_dir, 'post_training_args.pkl'), 'wb') as f:
        args = pickle.load(f)

    # Start a training run
    post_training_run(post_train_images_dir, post_train_labels_dir, pre_train_results_dir, post_train_results_dir, args)
    