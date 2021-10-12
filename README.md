# UV_Projet
This repo is meant to store code for UV Project IMT Nord Europe

Shiting PAN 
Bastien DELFORGE

# Dataset generation

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

