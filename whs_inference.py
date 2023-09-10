import torch 
import os
import glob

import monai.bundle
import subprocess
import shutil
    
def run_pred_monai(model_folder: str, input_dir: str, output_dir: str):

    # Get list of images
    input_filelist = glob.glob(os.path.join(input_dir, '*.nii.gz'))

    # Change folder to run the monai bundle
    cwd = os.getcwd()
    os.chdir(model_folder)

    # Run the MONAI prediction function
    subprocess.run(["python", "-m", "monai.bundle", "run", "--config_file", "./configs/inference.json", "--datalist", str(input_filelist), "--output_dir", output_dir])

    # Change folder back to previous
    os.chdir(cwd)

    # Arrange predictions into one folder, and delete individual folders
    predictions_filelist = glob.glob(os.path.join(output_dir, '*', '*.nii.gz'))

    for k in range(len(predictions_filelist)):

        os.rename(predictions_filelist[k], os.path.join(output_dir, predictions_filelist[k].split('/')[-1]))

    prediction_folders = glob.glob(os.path.join(output_dir, '*'))
    prediction_folders = [folder for folder in prediction_folders if os.path.isdir(folder)]

    for k in range(len(prediction_folders)):
        shutil.rmtree(prediction_folders[k])