import pandas as pd
import numpy as np
import os


def coef_dct(folder, order='zigzag'):
    """ compute average DCT of all images (with DC coefficient=0) in the given folder
    param folder : folder containing the images
    param order : order for vectorization of the DCT matrix ('zigzag' or 'simple')
    return avg_dct : list of average DCT"""
    img_blocks = os.listdir(folder)
    dct_blocks = [np.load(folder+img_block) for img_block in img_blocks if img_block.endswith('.npy')]
    avg_dct = np.mean(np.array(dct_blocks), axis=1)
    for x in avg_dct:
        x[0][0]=0
    if order=='zigzag':
        avg_dct = [np.concatenate([np.diagonal(x[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-x.shape[0], x.shape[0])]) for x in avg_dct]
    elif order=='simple':
        avg_dct = [x.flatten() for x in avg_dct] # dans l'ordre
    else:
        raise Exception("order should be 'zigzag' or 'simple'")

    return avg_dct


def train_test(input_folder, nb_dataset, train_size = 0.7, order='zigzag'):
    """ split images in the folders given by nb_dataset into train and test sets
    param input_folder : directory containing all the datasets
    param nb_dataset : list of numbers of datasets to perform on
    param train_size : size of training set
    param order : order for vectorization of the DCT matrix ('zigzag' or 'simple')
    return df_train, df_test : dataframes of train and test sets"""
    n = len(os.listdir(input_folder+str(nb_dataset[0])))-1
    index = list(np.arange(n))
    np.random.shuffle(index)
    train_index = index[:int((n + 1) * train_size)]
    test_index = index[int((n + 1) * train_size):]

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for i in nb_dataset:
        dct_ = coef_dct(input_folder+str(i)+"/", order=order )
        df_train = df_train.append([[list(x),i] for ind,x in enumerate(dct_) if ind in train_index ])
        df_test = df_test.append([[list(x), i] for ind, x in enumerate(dct_) if ind in test_index])

    df_train = df_train.reset_index(drop=True)
    df_train = df_train.rename(columns={0:'average_dct', 1:'class'})

    df_test = df_test.reset_index(drop=True)
    df_test = df_test.rename(columns={0: 'average_dct', 1: 'class'})
    # shuffle
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    return df_train, df_test


def cross_validation(input_folder, nb_dataset, n_splits, clf, order='zigzag'):
    """ Do cross validation with same splitting method
    param input_folder : directory containing all the datasets
    param nb_dataset : list of numbers of datasets to perform on
    param n_splits : number of folds
    param clf : model to train and test
    param order : order for vectorization of the DCT matrix ('zigzag' or 'simple')
    return : mean of scores"""
    n = len(os.listdir(input_folder+str(nb_dataset[0])))-1
    index = list(np.arange(n))
    np.random.shuffle(index)
    split_size = int(n/n_splits)
    scores = []
    for i in range(n_splits):
        test_index = index[i*split_size:(i+1)*split_size]
        train_index = [x for x in index if x not in test_index]
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        for i in nb_dataset:
            dct_ = coef_dct(input_folder+str(i)+"/", order=order)
            df_train = df_train.append([[list(x),i] for ind,x in enumerate(dct_) if ind in train_index ])
            df_test = df_test.append([[list(x), i] for ind, x in enumerate(dct_) if ind in test_index])

        df_train = df_train.reset_index(drop=True)
        df_train = df_train.rename(columns={0:'average_dct', 1:'class'})
        df_test = df_test.reset_index(drop=True)
        df_test = df_test.rename(columns={0: 'average_dct', 1: 'class'})
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        df_test = df_test.sample(frac=1).reset_index(drop=True)

        X_train, X_test, y_train, y_test = np.array([e for e in df_train['average_dct']]), np.array([e for e in df_test['average_dct']]), df_train['class'], df_test['class']
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))

    return np.mean(np.array(scores))