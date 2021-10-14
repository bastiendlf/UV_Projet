import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def show_confusion_matrix(y_test, y_pred, class_name):
    cm = confusion_matrix(y_test, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cmn, annot=True, fmt='.5f', xticklabels=class_name, yticklabels=class_name, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)


def save_model(model, path_to_model_folder, name):
    with open(os.path.join(path_to_model_folder, f"{name}.pickle"), 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path_to_model_folder, name):
    with open(os.path.join(path_to_model_folder, f"{name}.pickle"), 'rb') as handle:
        model = pickle.load(handle)
    return model
