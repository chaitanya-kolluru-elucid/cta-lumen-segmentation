# Lumen segmentation in CTA images
Lumen segmentation methods using Elucid in-house CTA datasets.

# Dataset preparation
Using datasets from MUSC, OLVZ and HolyCross institutions for model training and validation.  
Using datasets from MIE University for held-out testing.  
Both are single level branching datasets.

To test on multi level branching dataset, we use the ASOCA dataset.  
Datasets are at /inst/staging/AppData/Institutions/

MUSC:
FFR Pilot 2.0\
- CAP Exam Data_Group001
- CAP Exam Data_Group002
- CAP Exam Data_Group003

OLVZ:
FFR Pilot 2.0\
- CAP Exam Data_Group001A
- CAP Exam Data_Group001B
- CAP Exam Data_Group004A
- CAP Exam Data_Group005
 
Holy Cross:
FFR Pilot 2.0\
 - CAP Exam Data_Group001
 - CAP Exam Data_Group002

MIE University
FFR Pilot 2.0\
- CAP Exam Data_Group000

# Running the scripts
- Run create_training_datasets_latest.py on schoenberg machine. This will ask for an institution and CAP exam data group to parse, and save binary masks and image stacks in the specified save paths.
- Run remove_images_without_labels.py on schoenberg. If matching segmentation files are not found, this will remove the image stacks.  
- Images and labels are now ready to be copied over to paperspace.
- Clone this repository on paperspace. Set up data directory and copy over the images and labels.
- Run crop_with_whs.py with appropriate paths. This will run the MONAI whole body CT segmentation model, save those predictions in ```whsPreds``` folder, and then crop the images and labels and save them in folders with the suffix ```crop_```
- Run the train_model.py with appropriate parameters, training using the cropped images
- Run the run_inference.py script