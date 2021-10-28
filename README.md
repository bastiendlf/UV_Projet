# UV_Projet
This repo is meant to store code for UV Project IMT Nord Europe.

Shiting PAN 
Bastien DELFORGE

# Folders organisation

### /data
Contains .tiff non-compressed images to convert.
### /demo
Contains notebooks to see how to use some of our functions.
### /models
Contains trained machine learning models (pickle format). Please load them with load_model() from utils.training.py.
### /notebooks
Contains notebooks used to make our study, some results are still available. Note : if your folder /output/datasets/ is empty do not run them.
### /output
Contains the computed datasets with compressed images. Inside, there are the folders /datasets with images saved as list of DCT and /CompressedImg with images saved as .png with multiple compressions.
If you don't have these folders with content, ask to get .zip version of them. (long to generate)
### /utils
Contains utilities functions to make our study
### The Python scripts dataset_XXX.py
They are meant to compute the offline datasets. Default, they will compute the datasets for all parameters writen in the script.
For example dataset_d1_d2_d3.py, if you only want d2 run this -> python3 dataset_d1_d2_d3.py d2


### If you want to customize the settings of the dataset generation
1) Place the .tif images in /data
2) Make sure the folder /output/datasets exists
3) Open the jupyter notebook notebooks/MakeDataSet.ipynb
4) In the block below "Define settings values" set as many jpeg compression parameters as you wish.
5) Run each cell of the notebook (the last one can be long)
6) The datasets will be generated in the /output/datasets folder

### Note: 
Each dataset will be placed in a folder with a number. In a folder with a number, you will find: 
1 - numpy arrays representing jpeg compressed images
2 - a file called settings.pickle

This file contains the compression settings of the current dataset.

To load a dataset, please use the functions in the utils.makeDataset file

