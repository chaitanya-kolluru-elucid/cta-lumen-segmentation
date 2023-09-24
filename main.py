'''
Master script to run pre-training, training and inference 
all in one wandb run. Datasets are tracked as artifact
objects. 

Pre-trains on single level annotations, trained on 
complete coronary tree cases, test on held-out datasets.

Save model to .onnx
'''
import random
import numpy as np

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from monai.utils import first, set_determinism
from monai.transforms import *
from monai.networks.nets import *
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, get_confusion_matrix
from monai.losses import DiceLoss, DiceCELoss, GeneralizedDiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, PersistentDataset, ThreadDataLoader
from monai.config import print_config
from monai.metrics import compute_hausdorff_distance

import sys
import argparse
import wandb
import os
import glob
import math
import pickle
import yaml
import SimpleITK as sitk
import matplotlib.pyplot as plt
from datetime import datetime
from pytz import timezone
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

import metadata_calculator
from losses import TI_Loss, cldice, dsv

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
set_determinism(seed=0)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Train the model
def train(model, train_loader, val_loader, loss_function, metric, optimizer,  config, stage):

    # Tell wandb to log model gradients
    wandb.watch(model, None, log="all", log_freq=10)

    # Variables to hold loss and metrics    
    train_loss_values = []
    val_loss_values = []

    best_metric = -np.Inf
    best_metric_epoch = -1
    metric_values = []

    # Transforms for predictions and labels
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=config['generic_params']['num_classes'])]) 
    post_label = Compose([AsDiscrete(to_onehot=config['generic_params']['num_classes'])])

    # Train the model and validate for the set number of epochs
    for epoch in range(config[stage]['epochs']):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{config[stage]['epochs']}")
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
            print(f"{step}/{len(train_loader)}, " f"train_loss: {loss.item():.4f}")
            torch.cuda.empty_cache()

        train_loss /= step
        train_loss_values.append(train_loss)
        print(f"epoch {epoch + 1} average loss: {train_loss:.4f}")
        wandb.log({'train/loss':train_loss}, step=epoch)

        if (epoch + 1) % config[stage]['val_interval'] == 0:
            model.eval()
            val_loss = 0
            step = 0
            with torch.no_grad():
                for val_data in val_loader:
                    step += 1
                    val_inputs, val_labels = (val_data["image"].to(device), val_data["label"].to(device))
                    roi_size = config['generic_params']['roi_size']
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
                    torch.save(model.state_dict(), os.path.join(config['results_dir'], 'best_val_metric_model_' + stage + '.pt'))
                    print("saved new best metric model")

                print(f"current epoch: {epoch + 1} \ncurrent val mean loss: {val_loss:.4f} \ncurrent val mean dice: {val_metric:.4f} \nbest val mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")

# Load pre-training weights
def load_pretraining_weights(model, config):

    pretrain_weights_path = os.path.join(config['results_dir'], 'best_val_metric_model_pre_train.pt')

    if not os.path.exist(pretrain_weights_path):
        print('Could not find pretrained weights file in the results folder. Exiting.')
        sys.exit(1)

    # Load the weights into the model
    model.load_state_dict(torch.load(pretrain_weights_path))

    return model

# Load the model and weights for inference
def load_trained_model(config):

    # Get the GPU device
    device = torch.device("cuda:0")

    # Load model architecture
    model = torch.load(os.path.join(config['results_dir'], 'model_architecture.pt'))
    model = model.to(device)

    # Load the weights checkpoint
    model.load_state_dict(torch.load(os.path.join(config['results_dir'], 'best_val_metric_model_train.pt')))

    return model

# Run inference
def inference(model, test_loader, config, stage):

    # Create a folder to save prediction images
    os.makedirs(os.path.join(config['results_dir'], 'preds'))

    # Convert model to evaluation
    model.eval()

    post_transforms = Compose([Invertd(keys="pred", transform=test_transforms, orig_keys="image", meta_keys="pred_meta_dict",
                               orig_meta_keys="image_meta_dict", meta_key_postfix="meta_dict", nearest_interp=True, to_tensor=True)])

    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_inputs = test_data["image"].to(device)
            roi_size = config[stage]['roi_size']
            sw_batch_size = 4
            test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]

            volume = torch.argmax(test_data[0]["pred"], dim=0).detach().cpu()

            image = sitk.ReadImage(test_inputs.meta["filename_or_obj"][0])

            nrrd_label_volume = sitk.GetImageFromArray(np.transpose(volume, axes = [2,1,0]).astype(np.uint8))
            nrrd_label_volume.CopyInformation(image)
            
            filename = test_inputs.meta["filename_or_obj"][0].split('/')[-1]
            pred_filename = filename.split('.')[0] + '_seg_out.nrrd'

            sitk.WriteImage(nrrd_label_volume, os.path.join(config['results_dir'], 'preds', pred_filename))

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, torch.randn(1, config[stage]['roi_size'][0], config[stage]['roi_size'][1], config[stage]['roi_size'][2], device="cuda"), "model.onnx")
    wandb.save("model.onnx")

# Get model architecture, loss, metric and optimizer
def create_model_architecture(config, stage):

    # Get network architecture
    num_classes = config['generic_params']['num_classes'] 

    if config['generic_params']['architecture'] == 'UNet':
        model = UNet(spatial_dims=3, in_channels=1, out_channels=num_classes, channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2, norm=Norm.BATCH, dropout=0.2)

    elif config['generic_params']['architecture'] == 'SegResNet':
        model = SegResNet(spatial_dims=3, in_channels=1, out_channels=num_classes, norm=Norm.BATCH)

    elif config['generic_params']['architecture'] == 'AttentionUnet':
        model = AttentionUnet(spatial_dims=3, in_channels=1, out_channels=num_classes, channels=[16, 32, 64, 128, 256], strides=(2, 2, 2, 2, 2), dropout=0.0)

    elif config['generic_params']['architecture'] == 'FlexibleUNet':
        model = FlexibleUNet(spatial_dims=3, in_channels=1, out_channels=num_classes, backbone='efficientnet-b0', pretrained=False, decoder_channels=(256, 128, 64, 32, 16))

    else:
        print('Model architecture not found, ensure that the architecture is either UNet, SegResNet, AttentionUnet or FlexibleUNet. Exiting.')
        sys.exit(1)
    
    # Get the GPU device
    device = torch.device("cuda:0")

    # Move model to GPU
    model = model.to(device)

    # Save the model architecture to the results folder
    torch.save(model, os.path.join(config['results_dir'], 'model_architecture.pt'))

    # Get loss function
    loss_function = config[stage]['loss']

    if loss_function == 'DiceCE':
        loss_function = DiceCELoss(include_background=config[stage]['include_bg_in_loss'], to_onehot_y=True, softmax=True, batch=config[stage]['dice_batch_reduction'], ce_weight=torch.Tensor(config[stage]['ce_weights']).to(device)) 

    elif loss_function == 'Topological':
        dice_ce_loss = DiceCELoss(include_background=config[stage]['include_bg_in_loss'], to_onehot_y=True, softmax=True, batch=config[stage]['dice_batch_reduction'], ce_weight=torch.Tensor(config[stage]['ce_weights']).to(device)) 
        ti_loss_function = TI_Loss.TI_Loss(dim=3, connectivity=26, inclusion=[[2,1]], exclusion=[], min_thick=1)
        loss_function = lambda outputs, labels: dice_ce_loss(outputs, labels) + (1e-6) * ti_loss_function(outputs, labels)
    
    elif loss_function == 'clDice':
        loss_function = cldice.soft_dice_cldice_ce(iter_=20, num_classes = num_classes, lumen_class=1, include_bg=config[stage]['include_bg_in_loss'])

    else:
        print('Loss function not found, ensure that the loss is one of DiceCE, Topological or clDice. Exiting.')
        sys.exit(1)        

    # Create the metric
    if config[stage]['metrics'] == 'Dice':
        metric = DiceMetric(include_background=False, reduction="mean") 
    else:
        print('Alternate metrics not yet implemented.Exiting')
        sys.exit(1)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), config[stage]['learning_rate']) 
    
    return model, loss_function, metric, optimizer 
    
# Create data loaders
def create_data_loaders(config, stage):

    # Get list of training images and labels
    imagesList = sorted(glob.glob(os.path.join(config['data_dir'], config[stage]['images_data_dir'], "*.nrrd")))
    labelsList =  sorted(glob.glob(os.path.join(config['data_dir'], config[stage]['labels_data_dir'], "*.nrrd")))

    # Shuffle image and label lists the same way
    random.seed(3)
    random.shuffle(imagesList)

    random.seed(3)
    random.shuffle(labelsList)

    # Check whether same number of images and labels exist
    if not len(imagesList) == len(imagesList):
        print('Please verify that the number of images and labels are the same. Exiting.')
        sys.exit(1)
    
    # Get number of validation cases
    num_validation_cases = math.floor(config[stage]['val_ratio'] * len(imagesList))

    # Create data dictionaries, training and validation
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(imagesList, labelsList)]

    if num_validation_cases != 0:
        train_files, val_files = data_dicts[:-num_validation_cases], data_dicts[-num_validation_cases:]
    else:
        train_files = data_dicts

    # Get the target pixel spacing used for transforms
    with open(os.path.join(config['metadata_dir'], config['generic_params']['target_pixel_spacing']), 'rb') as f:
        target_pixel_spacing = pickle.load(f)

    # Create data transforms for training and validation
    train_transforms = Compose([
                                LoadImaged(keys=["image", "label"], image_only=False),
                                EnsureChannelFirstd(keys=["image", "label"]),
                                NormalizeIntensityd(keys=["image"]),
                                Orientationd(keys=["image", "label"], axcodes="RAS"),
                                Spacingd(keys=["image", "label"], pixdim=target_pixel_spacing, mode=("bilinear", "nearest")),
                                CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8)),
                                RandCropByLabelClassesd(keys=["image", "label"], label_key="label", spatial_size=config['generic_params']['roi_size'],
                                                        num_classes=config['generic_params']['num_classes'], ratios=config[stage]['crop_ratios'],
                                                        num_samples=int(np.sum(config[stage]['crop_ratios'])))
                                ])

    val_transforms = Compose([
                            LoadImaged(keys=["image", "label"], image_only=False),
                            EnsureChannelFirstd(keys=["image", "label"]),
                            NormalizeIntensityd(keys=["image"]),
                            Orientationd(keys=["image", "label"], axcodes="RAS"),
                            Spacingd(keys=["image", "label"], pixdim=target_pixel_spacing, mode=("bilinear", "nearest")),
                            CastToTyped(keys=["image", "label"], dtype=(np.float32, np.uint8))
                            ])

    # Create training and validation data loaders
    train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir = './cache')
    train_loader = DataLoader(train_ds, batch_size=config[stage]['batch_size'], shuffle=True, num_workers=4)

    if num_validation_cases != 0:
        val_ds = CacheDataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
    else:
        val_loader = None

    # Return loaders
    return train_loader, val_loader

# Get the current date time for ET timezone
def get_current_datetime(zone):

    tz = timezone(zone)
    current_datetime = datetime.now(tz).strftime("%D%m%y_%H%M%S")
    return current_datetime

# Create references to the dataset as wandb artifacts
def create_and_log_dataset_artifacts(config):

    # Create the artifacts
    pre_train_images_artifact = wandb.Artifact(config['pre_train']['images_data_dir'], type="dataset")
    pre_train_labels_artifact = wandb.Artifact(config['pre_train']['labels_data_dir'], type="dataset")

    train_images_artifact = wandb.Artifact(config['train']['images_data_dir'], type="dataset")
    train_labels_artifact = wandb.Artifact(config['train']['labels_data_dir'], type="dataset")
    
    test_images_artifact = wandb.Artifact(config['test']['images_data_dir'], type="dataset")
    test_labels_artifact = wandb.Artifact(config['test']['labels_data_dir'], type="dataset")

    # Add references to the artifacts    
    pre_train_images_artifact.add_reference('file:' + os.path.join(config['data_dir'], config['pre_train']['images_data_dir']))
    pre_train_labels_artifact.add_reference('file:' + os.path.join(config['data_dir'], config['pre_train']['images_data_dir']))

    train_images_artifact.add_reference('file:' + os.path.join(config['data_dir'], config['train']['images_data_dir']))
    train_labels_artifact.add_reference('file:' + os.path.join(config['data_dir'], config['train']['images_data_dir']))

    test_images_artifact.add_reference('file:' + os.path.join(config['data_dir'], config['test']['images_data_dir']))
    test_labels_artifact.add_reference('file:' + os.path.join(config['data_dir'], config['test']['images_data_dir']))

    # Log the artifacts
    wandb.log_artifact(pre_train_images_artifact)
    wandb.log_artifact(pre_train_labels_artifact)
    wandb.log_artifact(train_images_artifact)
    wandb.log_artifact(train_labels_artifact)
    wandb.log_artifact(test_images_artifact)
    wandb.log_artifact(test_labels_artifact)

# Execute model pipeline
def model_pipeline(hyperparameters):

    # Tell wandb to get started
    with wandb.init(project=hyperparameters['wandb_project'], config=hyperparameters):
      
        # Access all hyperparameters through wandb.config, so logging matches execution!
        config = wandb.config

        # Create a results directory with current date and time
        current_datetime = get_current_datetime()
        config['results_dir'] = os.path.join(config['results_dir'], current_datetime)
        os.makedirs(config['results_dir'])

        # Create and log references to the datasets as artifacts
        create_and_log_dataset_artifacts(config)

        # Model pre-training
        stage = 'pre_train'
        if config[stage]['run']:

            train_loader, val_loader = create_data_loaders(config, stage)
            model, loss, metric, optimizer = create_model_architecture(config, stage)
            train(model, train_loader, val_loader, loss, metric, optimizer,  config, stage)

        # Model training
        stage = 'train'
        if config[stage]['run']:

            train_loader, val_loader = create_data_loaders(config, stage)
            model, loss, metric, optimizer = create_model_architecture(config, stage)

            if config[stage]['use_pre_training_weights']:
                model = load_pretraining_weights(model, config)

            train(model,train_loader, val_loader, loss, metric, optimizer,  config, stage)

        # Model inference
        if config['inference']['run']:
            stage = 'inference'

            test_loader, val_loader = create_data_loaders(config, stage)

            model = load_trained_model(config)
            model, loss, optimizer = create_model_architecture(config, stage)

            inference(model, test_loader, config, stage)

    return model    

# Entry point for the script
if __name__ == '__main__':

    # Read the hyperparameters yaml file
    if os.path.exists('./hyperparameters.yaml'):

        with open('./hyperparameters.yaml', 'r') as f:
            hyperparameters = yaml.safe_load(f)
   
    else:
        print('Did not find a hyperparameters.yaml file in current folder. Exiting.')
        sys.exit(1)

    # Build, train and test the model with the pipeline
    model = model_pipeline(hyperparameters)