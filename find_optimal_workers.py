from time import time
import multiprocessing as mp
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import random, glob, os, sys, math
from tqdm import tqdm
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
    RandFlipd
)
import numpy as np

data_dir = './data'
images_dir = 'crop_imagesTr'
labels_dir = 'crop_labelsTr'
mask_suffix = '-lumen-wall-mask'

train_roi_size = (96, 96, 96)
crop_ratios = [0, 4, 4]

# Get list of training images
images = sorted(glob.glob(os.path.join(data_dir, images_dir, "*.nii.gz")))

# Get list of training labels
labels = sorted(glob.glob(os.path.join(data_dir, labels_dir, '*' + mask_suffix + '.nii.gz')))

# Set random seed and shuffle image list
random.seed(3)
random.shuffle(images)

# Set random seed and shuffle label list
random.seed(3)
random.shuffle(labels)

if not len(images) == len(labels):
    print('Please verify that the number of images and labels are the same. Exiting.')
    sys.exit(0)

num_cases = len(images)
num_validation_cases = math.floor(0.9 * num_cases)

# Create data dictionaries, training for 10 cases, validation on 2 cases
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]
train_files, val_files = data_dicts[:-num_validation_cases], data_dicts[-num_validation_cases:]

median_pixel_spacing = (0.4, 0.4, 0.4)
fg_intensity_metrics = [18, 388, 760, -923]

# Transforms for the training dataset
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], image_only=False),
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
            spatial_size=train_roi_size,
            num_classes=3,
            ratios=crop_ratios,
            num_samples=int(np.sum(crop_ratios)), 
        ),
    ]
)

train_ds = Dataset(data=train_files, transform=train_transforms)

# for batch_size in tqdm(range(1, 24, 2)):
#     for num_workers in tqdm(range(2, mp.cpu_count(), 2)):  
#         train_loader = DataLoader(train_ds,shuffle=True,num_workers=num_workers,batch_size=batch_size,pin_memory=True)
#         start = time()
#         for i, data in enumerate(train_loader, 0):
#             print(i)
#         end = time()
#         print("Finish with:{} second, num_workers={}, batch_size={}".format(end - start, num_workers, batch_size))

num_workers = 8
batch_size = 2
train_loader = DataLoader(train_ds,shuffle=True,num_workers=num_workers,batch_size=batch_size,pin_memory=True)
start = time()
for i, data in enumerate(train_loader, 0):
    print(i)
end = time()
print("Finish with:{} second, num_workers={}, batch_size={}".format(end - start, num_workers, batch_size))