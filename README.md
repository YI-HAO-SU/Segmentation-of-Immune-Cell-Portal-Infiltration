# Segmentation-of-Immune-Cell-Portal-Infiltration

The primary objective is to accurately delineate and identify areas of immune cell portal infiltration within a given context. 
Through advanced segmentation techniques, the project aims to provide a precise and detailed map of immune cell distribution around portals.
This segmentation process facilitates a deeper understanding of immune cell behavior and spatial relationships, contributing valuable insights to immunological studies and related research endeavors.

![圖片](https://github.com/YeeHaoSu/Segmentation-of-Immune-Cell-Portal-Infiltration/assets/90921571/584e64a8-312c-4d09-ab52-a65fd6801974)


# File & Directory Structure

$username == username in local

$model_version == different type of model (ResUnet++, AttenUnet)

$version == training version

$case_name == file name of WSI 

    
# Into. to each .py File 
## Main Execution File

0. run.sh

    0-1 Remember to change username、version、model_version
   
    0-2 Run the following .py files in sequence

1. save_training_pkl.py

   1-1 Read the case name in case_list.txt, only the patches with the specified case name will be put into the training.

   1-2 Put the original patch in the specified data_dir_path.

   1-3 Save the path as train.pkl, val.pkl.

2. show_pkl.py

   2-1 Printing out the first contents of the specified pkl

3. train.py

   3-1 Optional connection to wandb project is $model_version, run is $version.

   3-2 weight is saved for each epoch, validation loss is saved as best_model.pt.
