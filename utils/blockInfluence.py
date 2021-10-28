import os
import pandas as pd
import numpy as np
import utils.makeDataset as DatasetMaker
import cv2
import matplotlib.pyplot as plt


def read_dct_blocks(datasets):
    """ Read all DCT blocks of all images from all groups in datasets
    param datasets : list of datasets numbers to train and test
    return : dataframe where one row refers to one DCT block in one image"""
    my_list = []
    for current_dataset in datasets:
        _, X_current = DatasetMaker.load_dataset(current_dataset)
        for i in range(len(X_current)):
            for block in X_current[i]:
                my_list.append([i+1, block.flatten(), current_dataset])
    df = pd.DataFrame(my_list, columns=['image_number','dct', 'class'])
    df = df.reset_index(drop=True)
    return df

def train_test_block(input_folder, datasets, train_size=0.7):
    """Create a dataframe with all DCT blocks and split it into train and test set,
    the same image with two different compression cannot be both in training and test set
    param input_folder : folders where are stored the datasets
    param datsets : list of numbers of datasets
    param train_size : training set size (0.01 to 1)
    return : 2 dataframes : training and test"""
    n = len(os.listdir(input_folder+str(datasets[0])))-1
    index = list(np.arange(n))
    np.random.shuffle(index)
    train_index = index[:int((n + 1) * train_size)]
    test_index = index[int((n + 1) * train_size):]

    all_data = read_dct_blocks(datasets)
    df_train = all_data[all_data['image_number'].isin(train_index)]
    df_test = all_data[all_data['image_number'].isin(test_index)]
    return df_train, df_test


def training(clf, input_folder, datasets, train_size=0.7):
    """ Train the model with the given datasets
    param clf : model of classification
    param input_folder : folders where are stored the datasets
    param datasets : list of numbers of datasets
    param train_size : training set size (0.01 to 1)
    return : the trained classifier, dataframes of train and test set """
    train, test = train_test_block(input_folder, datasets, train_size)
    X_train, X_test, y_train, y_test = np.array([e for e in train['dct']]), np.array([e for e in test['dct']]), train['class'], test['class']
    clf.fit(X_train, y_train)
    return clf, train, test


def prediction(clf, test, datasets):
    """ Complete the dataframe of test with prediction results
    param clf : model of classification
    param test : dataframe of test set
    param datasets : list of numbers of datasets used for training
    return : dataframe of test set with prediction results in columns 'pred_class' and 'pred_proba'"""
    x_test = np.array([e for e in test['dct']])
    res = clf.predict_proba(x_test)
    test['pred_class']=[datasets[x] for x in np.argmax(res, axis=1)]
    test['pred_proba']=np.ndarray.max(res, axis=1)
    return test

def localization_on_image(data_folder, image_number, dataset_number, test, score=0.90):
    """ Localize   100*(1-score)% blocks with the best prediction scores
    param data_folder : folder where are stored the images
    param image_number : number of image for which you want to localize the blocs with best prediction
    param dataset_number : list of numbers of datasets
    param test : dataframe of test set with prediction results
    param score : 100*(1-score)% blocs will be showed on the result image
    return : image with rectangles, dataframe containing all blocks of the given images, with positions in the image,
    list of positions of rectangles in the image"""
    img = cv2.imread(data_folder+'ucid'+str(0)*(5-len(str(image_number)))+str(image_number)+'.tif', 0) # image name
    height, width = img.shape
    block_pred = test[(test['class']==dataset_number)&(test['image_number']==image_number)]
    block_pred = block_pred.reset_index(drop=True)
    color = (255, 0, 0)
    thickness = 1
    n_bloc_w = width/8
    block_pred['block_start_point'] = [[int(x%n_bloc_w)*8, int(x/n_bloc_w)*8] for x in block_pred.index.values]
    block_pred['block_end_point'] = [[x[0]+8, x[1]+8] for x in block_pred['block_start_point']]
    block_best_pred = block_pred[block_pred['pred_proba']>=score]
    vis_blocks = []
    for i in block_best_pred.index.values:
        img = cv2.rectangle(img, block_best_pred.loc[i,'block_start_point'], block_best_pred.loc[i,'block_end_point'], color, thickness)
        vis_blocks.append(block_best_pred.loc[i,'block_start_point'])
    return img, block_pred, vis_blocks


def visualize_bloc(data_folder, datasets, test_res, image_range, dataset_range, score_quantile):
    """ Return images with (1-score_quantile)% best predicted blocs surrounded by rectangle
    param data_folder : folder where are stored the images
    param datasets : list of numbers of datasets
    param test_res : dataframe of test set with prediction results
    param image_range : range of image (in the shuffled test set, unique number) that you want to visualize
    param dataset_range : dataset range for this image
    param score_quantile : (1-score_quantile)% best predicted blocs
    return : image with rectangles, dataframe containing all blocks of the given images, with positions in the image,
    list of positions of rectangles in the image"""
    img_numbers = test_res['image_number'].unique()
    img_number = img_numbers[image_range]
    dataset_nb = datasets[dataset_range]
    s = test_res[test_res['class'] == dataset_nb].pred_proba.quantile([score_quantile]).values[0]

    img_res, bl_pred, vis = localization_on_image(data_folder, img_number, dataset_nb, test_res, s)
    img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
    plt.imshow(img_res)
    return img_res, bl_pred, vis