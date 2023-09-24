import glob
import SimpleITK as sitk
from tqdm import tqdm
import os

input_dir = './data/crop_labelsTs_asoca'
input_filelist = glob.glob(os.path.join(input_dir, '*.nii.gz'))

output_dir = './data/asoca_labels'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
for k in tqdm(range(len(input_filelist))):

    image = sitk.ReadImage(input_filelist[k])

    filename = input_filelist[k].split('/')[-1] 

    sitk.WriteImage(image, os.path.join(output_dir, filename[:-6] + 'nrrd'))