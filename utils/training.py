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
    cm = confusion_matrix(y_test, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cmn, annot=True, fmt='.5f', xticklabels=class_name, yticklabels=class_name, cmap='Blues')
    if title != "":
        plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    if show:
        plt.show(block=False)


def save_model(model, path_to_model_folder, name):
    with open(os.path.join(path_to_model_folder, f"{name}.pickle"), 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path_to_model_folder, name):
    with open(os.path.join(path_to_model_folder, f"{name}.pickle"), 'rb') as handle:
        model = pickle.load(handle)
    return model


def split_dct_img(X, y):
    res_X = []
    res_y = []
    for curr_X, curr_y in zip(X, y):
        for block in curr_X:
            res_X.append(block.reshape(64))
            res_y.append(curr_y)
    return res_X, res_y


def majority_vote(model, X_test):
    y_pred = []
    for curr_X_test in X_test:
        curr_y_pred = model.predict(
            curr_X_test.reshape(curr_X_test.shape[0], curr_X_test.shape[1] * curr_X_test.shape[2]))
        unique, counts = np.unique(curr_y_pred, return_counts=True)
        y_pred.append(unique[np.argmax(counts)])
    return y_pred


def get_average_dct(dct_images):
    list_average_dct = np.mean(np.array(dct_images), axis=1)
    return [avg_2D.flatten() for avg_2D in list_average_dct]


def unique_images(X, y):
    nb_class = len(np.unique(y))
    values, counts = np.unique(y, return_counts=True)

    img_per_class = int(len(X) / nb_class)
    extract = int(img_per_class / nb_class)

    new_X = []
    new_Y = []

    for i in range(nb_class):
        # print(i * counts[i] + i * extract)
        # print(i * counts[i] + i * extract + extract)
        new_X += X[i * counts[i] + i * extract: i * counts[i] + i * extract + extract]
        new_Y += y[i * counts[i] + i * extract: i * counts[i] + i * extract + extract]

    return new_X, new_Y


def get_trained_model(model, PATH_MODELS, MODEL_NAME, X_train_block=None, y_train_block=None):
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
