import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import utils.makeDataset as DatasetMaker
import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def show_confusion_matrix(y_test, y_pred, class_name, title="", show=True):
    """
    Plot a nice looking confusion matrix
    :param y_test: list
    :param y_pred: list
    :param class_name: list of labels
    :param title: str
    :param show: Bool
    :return: None
    """
    cm = confusion_matrix(y_test, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cmn, annot=True, fmt='.5f', xticklabels=class_name, yticklabels=class_name, cmap='Blues')
    if title != "":
        plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    if show:
        plt.show(block=False)


def save_model_and_label_encoder(model, le, path_to_model_folder, name):
    """
    Saves model and label encoder used to train model
    :param model: machine learning trained model
    :param le: label_encoder
    :param path_to_model_folder: str
    :param name: str
    :return: none
    """
    save_model(model, path_to_model_folder, name)
    with open(os.path.join(path_to_model_folder, f"LABEL_ENCODER_{name}.pickle"), 'wb') as handle:
        pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_and_label_encoder(path_to_model_folder, name):
    """
    Loads model and label encoder used to train model
    :param path_to_model_folder: str
    :param name: str
    :return: none
    """
    model = load_model(path_to_model_folder, name)
    with open(os.path.join(path_to_model_folder, f"LABEL_ENCODER_{name}.pickle"), 'rb') as handle:
        label_encoder = pickle.load(handle)
    return model, label_encoder


def save_model(model, path_to_model_folder, name):
    """
    Saves model
    :param model: machine learning trained model
    :param path_to_model_folder: str
    :param name: str
    :return: none
    """
    with open(os.path.join(path_to_model_folder, f"{name}.pickle"), 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path_to_model_folder, name):
    """
    Loads trained model
    :param path_to_model_folder: str
    :param name: str
    :return: none
    """
    with open(os.path.join(path_to_model_folder, f"{name}.pickle"), 'rb') as handle:
        model = pickle.load(handle)
    return model


def split_dct_img(X, y):
    """
    Converts a list of image (one image is a list of DCT) into a list of DCT with the corresponding labels
    :param X: list of list of 8x8 DCT blocks
    :param y: labels
    :return: list of DCT blocks and their labels
    """
    res_X = []
    res_y = []
    for curr_X, curr_y in zip(X, y):
        for block in curr_X:
            res_X.append(block.reshape(64))
            res_y.append(curr_y)
    return res_X, res_y


def majority_vote(model, X_test):
    """
    Predicts an image class based on the majority vote of its DCT blocks
    :param model: machine learning trained model
    :param X_test: list images (on images is a list of DCT blocks)
    :return: list of prediction per image based on the majority vote of its dct blocks
    """
    y_pred = []
    for curr_X_test in X_test:
        curr_y_pred = model.predict(
            curr_X_test.reshape(curr_X_test.shape[0], curr_X_test.shape[1] * curr_X_test.shape[2]))
        unique, counts = np.unique(curr_y_pred, return_counts=True)
        y_pred.append(unique[np.argmax(counts)])
    return y_pred


def get_average_dct(dct_images):
    """
    Computes the average DCT block for a list of compressed images
    :param dct_images: list of images, an image is a list of DCT blocks
    :return: list of DCT average
    """
    list_average_dct = np.mean(np.array(dct_images), axis=1)
    return [avg_2D.flatten() for avg_2D in list_average_dct]


def unique_images(X, y):
    """
    See report -> its method 1 to split data (not the train the model with the same images with different compressions
    :param X: list of compressed images (average dct)
    :param y: list of correponding labels
    :return: list of X and list of y
    """
    nb_class = len(np.unique(y))
    values, counts = np.unique(y, return_counts=True)

    img_per_class = int(len(X) / nb_class)
    extract = int(img_per_class / nb_class)

    new_X = []
    new_Y = []

    for i in range(nb_class):
        new_X += X[i * counts[i] + i * extract: i * counts[i] + i * extract + extract]
        new_Y += y[i * counts[i] + i * extract: i * counts[i] + i * extract + extract]

    return new_X, new_Y


def get_trained_model(model, PATH_MODELS, MODEL_NAME, X_train_block=None, y_train_block=None):
    """
    Get a trained model (if not available offline, will be trained and saved)
    :param model: machine learning model
    :param PATH_MODELS: path (str)
    :param MODEL_NAME: str
    :param X_train_block: list of DCT blocks
    :param y_train_block: list of corresponding labels
    :return: a trained model
    """
    if not (os.path.exists(os.path.join(PATH_MODELS, MODEL_NAME + '.pickle'))):
        print(f"Training {MODEL_NAME}, please wait...")
        model.fit(X_train_block, y_train_block)
        print(f"Saving {MODEL_NAME} at {os.path.join(PATH_MODELS, MODEL_NAME)}.pickle")
        save_model(model, PATH_MODELS, MODEL_NAME)
    else:
        print(f"Loading {MODEL_NAME}, please wait...")
        model = load_model(PATH_MODELS, MODEL_NAME)
    return model


def pipeline_dct_blocks(MODEL_TYPE, DATASETS):
    """
    Loads model, trains it and evaluates it for DCT blocks and majority vote
    :param MODEL_TYPE: must be "rf", "adaboost" or 'svm"
    :param DATASETS: list of dataset folder names (["1", "2", "3"]
    :return: None
    """
    MODELS = {
        "rf": RandomForestClassifier(max_depth=2, random_state=0),
        "adaboost": AdaBoostClassifier(),
        "svm": make_pipeline(StandardScaler(), SVC(gamma='auto', probability=False))
    }

    PATH_MODELS = f"../models/train_on_blocks/{MODEL_TYPE}/"
    MODEL_NAME = ''.join([str(MODEL_TYPE)] + ['_c' + str(curr) for curr in DATASETS])

    X = []
    y = []

    for current_dataset in DATASETS:
        _, X_current = DatasetMaker.load_dataset(current_dataset)
        y += [current_dataset] * len(X_current)
        X += X_current

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    X_train_block, y_train_block = split_dct_img(X_train, y_train)
    X_test_block, y_test_block = split_dct_img(X_test, y_test)

    model = get_trained_model(MODELS[MODEL_TYPE], PATH_MODELS, MODEL_NAME, X_train_block, y_train_block)
    y_pred_block = model.predict(X_test_block)
    print(classification_report(y_test_block, y_pred_block))
    y_pred = majority_vote(model, X_test)
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(12, 5))
    plt.suptitle(MODEL_NAME)
    plt.subplot(121)
    show_confusion_matrix(y_test_block, y_pred_block, DATASETS, "DCT blocks", False)
    plt.subplot(122)
    show_confusion_matrix(y_test, y_pred, DATASETS, "Majority vote", False)
    plt.show()


def pipeline_avg_dct(MODEL_TYPE, DATASETS):
    """
    Load model, train it and evaluates it for average DCT
    :param MODEL_TYPE: must be "rf", "adaboost" or 'svm"
    :param DATASETS: list of dataset folder names (["1", "2", "3"]
    :return: None
    """
    MODELS = {
        "rf": RandomForestClassifier(max_depth=10, random_state=0),
        "adaboost": AdaBoostClassifier(),
        "svm": make_pipeline(StandardScaler(), SVC(gamma='auto', probability=False))
    }

    PATH_MODELS = f"../models/averageDCT/{MODEL_TYPE}/"
    MODEL_NAME = ''.join([str(MODEL_TYPE)] + ['_c' + str(curr) for curr in DATASETS])

    X = []
    y_label = []

    for current_dataset in DATASETS:
        _, X_current = DatasetMaker.load_dataset(current_dataset)
        y_label += [current_dataset] * len(X_current)
        X += X_current

    le = LabelEncoder()
    y = le.fit_transform(y_label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = get_trained_model(MODELS[MODEL_TYPE], PATH_MODELS, MODEL_NAME, X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    show_confusion_matrix(y_test, y_pred, DATASETS, f"Average DCT {MODEL_NAME}")


def compare_pipeline_avg_dct(DATASETS, MODELS):
    """
    Load models, trains them and evaluates them for average DCT
    :param DATASETS: list of dataset folder names (["1", "2", "3"]
    :param MODELS: list of model type (["rf", "svm", "adaboost"]
    :return: none
    """
    X = []
    y_label = []

    for current_dataset in DATASETS:
        _, X_current = DatasetMaker.load_dataset(current_dataset)
        y_label += [current_dataset] * len(X_current)
        X += X_current

    le = LabelEncoder()
    y = le.fit_transform(y_label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    for model_type in MODELS.keys():
        PATH_MODELS = f"../models/averageDCT/{model_type}/"
        MODEL_NAME = ''.join([str(model_type)] + ['_c' + str(curr) for curr in DATASETS])

        model = get_trained_model(MODELS[model_type], PATH_MODELS, MODEL_NAME, X_train, y_train)
        y_pred = model.predict(X_test)
        print(model_type)
        print(classification_report(y_test, y_pred))
        show_confusion_matrix(y_test, y_pred, DATASETS, f"Average DCT {MODEL_NAME}")
