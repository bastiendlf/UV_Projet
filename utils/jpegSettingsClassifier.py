import numpy as np
from utils.training import load_model_and_label_encoder, load_model
import pickle
import os


class JpegSettingsClassifier:
    """
    This object is meant to help loading data, making predictions and computing metrics for Qf and d for our project
    """

    def __init__(self, model_qf="rf", model_d="rf"):
        self.MODEL_QF = model_qf
        self.PATH_MODELS_QF = f"../models/final_models/Q/{self.MODEL_QF}/"

        self.MODEL_QF_NAME = f"{self.MODEL_QF}_Q10_Q50_Q90"
        self.model_Q, self.label_encoder_Q = load_model_and_label_encoder(self.PATH_MODELS_QF, self.MODEL_QF_NAME)

        self.MODEL_D = model_d
        self.PATH_MODELS_D = "../models/final_models/"

        self.MODEL_NAME_Q10 = f"{self.MODEL_D}_d1_Q10_d2_Q10_d3_Q10"
        self.model_d_Q10 = load_model(os.path.join(self.PATH_MODELS_D, "d_Q10/"), self.MODEL_NAME_Q10)

        self.MODEL_NAME_Q50 = f"{self.MODEL_D}_d1_Q50_d2_Q50_d3_Q50"
        self.model_d_Q50 = load_model(os.path.join(self.PATH_MODELS_D, "d_Q50/"), self.MODEL_NAME_Q50)

        self.MODEL_NAME_Q90 = f"{self.MODEL_D}_d1_Q90_d2_Q90_d3_Q90"
        self.model_d_Q90 = load_model(os.path.join(self.PATH_MODELS_D, "d_Q90/"), self.MODEL_NAME_Q90)

        with open(os.path.join(self.PATH_MODELS_D, f"LABEL_ENCODER_rf_d1_d2_d3.pickle"), 'rb') as handle:
            self.label_encoder_d = pickle.load(handle)

    def get_y_true_q(self, list_dict_ground_truth):
        """
        Get a list of encoded labels for Qf from a list of dictionary of ground truth
        :param list_dict_ground_truth: must be the loaded file settings.pickle from the /validation dataset
        :return: list of labels
        """
        return self.label_encoder_Q.transform([f"Q{dict_settings['Q']}" for dict_settings in list_dict_ground_truth])

    def get_y_q_labels(self, y_q):
        """
        Transforms encoded labels into labels (string)
        :param y_q: list of encoded labels
        :return: list
        """
        return self.label_encoder_Q.inverse_transform(y_q)

    def predict_q(self, X):
        """
        Predicts Qf for a list of compressed images
        :param X: X must be a list of 64 x 1 vectors, each vector is the average DCT of a compressed image
        :return: list of predictions
        """
        return self.model_Q.predict(X)

    def get_y_true_d(self, list_dict_ground_truth):
        """
        Get a list of encoded labels for d from a list of dictionary of ground truth
        :param list_dict_ground_truth: must be the loaded file settings.pickle from the /validation dataset
        :return: list of labels
        """
        y_d_true = []
        for element in list_dict_ground_truth:
            if "function d1" in str(element["d"]):
                y_d_true.append(self.label_encoder_d.transform(['d1'])[0])
            elif "function d2" in str(element["d"]):
                y_d_true.append(self.label_encoder_d.transform(['d2'])[0])
            elif "function d3" in str(element["d"]):
                y_d_true.append(self.label_encoder_d.transform(['d3'])[0])
        return y_d_true

    def predict_d(self, X, y_q_pred_labels=None):
        """
        Predicts d for a list of compressed images
        :param X: X must be a list of 64 x 1 vectors, each vector is the average DCT of a compressed image
        :param y_q_pred_labels: previously predicted Qf for each image if not provided, the will be predicted inside
        :return: a list of predictions
        """

        if y_q_pred_labels is None:
            y_q_pred = self.predict_q(X)
            y_q_pred_labels = self.get_y_q_labels(y_q_pred)

        y_d_pred = []
        for current_X, Q_pred in zip(X, y_q_pred_labels):
            if Q_pred == "Q10":
                y_d_pred.append(self.model_d_Q10.predict(current_X.reshape(1, -1))[0])
            elif Q_pred == "Q50":
                y_d_pred.append(self.model_d_Q50.predict(current_X.reshape(1, -1))[0])
            elif Q_pred == "Q90":
                y_d_pred.append(self.model_d_Q90.predict(current_X.reshape(1, -1))[0])
        return y_d_pred

    def get_y_d_labels(self, y_d):
        """
        Transforms encoded labels into labels (string)
        :param y_q: list of encoded labels
        :return: list
        """
        return self.label_encoder_d.inverse_transform(y_d)

    def get_y_multiclass(self, y_q, y_d):
        """
        Transforms list of Qf (groundtruth of prediction) and q (groundtruth of prediction) into a matrix of prediction
        (2 columns and X lines)
        :param y_q: Qf labels
        :param y_d: d labels
        :return:
        """
        y_multiclass = []
        for q, d in zip(y_q, y_d):
            y_multiclass.append([q, d])
        return np.array(y_multiclass)

    def get_well_predicted_Q(self, y_true_multiclass, y_pred_multiclass):
        """
        Computes the amount of well predicted Q
        :param y_true_multiclass: 2D array
        :param y_pred_multiclass: 2D array
        :return: float (%)
        """
        return np.sum(np.equal(y_true_multiclass[:, 0], y_pred_multiclass[:, 0])) / float(
            y_true_multiclass.shape[0])

    def get_well_predicted_d(self, y_true_multiclass, y_pred_multiclass):
        """
        Computes the amount of well predicted d
        :param y_true_multiclass: 2D array
        :param y_pred_multiclass: 2D array
        :return: float (%)
        """
        return np.sum(np.equal(y_true_multiclass[:, 1], y_pred_multiclass[:, 1])) / float(
            y_true_multiclass.shape[0])

    def get_well_predicted_Q_and_d(self, y_true_multiclass, y_pred_multiclass):
        """
        Computes the amount of well predicted Qf and d at the same time
        :param y_true_multiclass: 2D array
        :param y_pred_multiclass: 2D array
        :return: float (%)
        """
        compare = y_true_multiclass == y_pred_multiclass
        good_Q_and_d_number = len([element for element in compare if (element[0] and element[1])])
        return good_Q_and_d_number / float(y_true_multiclass.shape[0])

    def convert_multi_label_to_single(self, y_multilabel):
        """
        Converts multi label, multiclass to multiclass
        :param y_multilabel: 2D array
        :return: np array of encoded labels
        """
        new_y = []
        for q, d in y_multilabel:
            new_class = None
            if q == 0 and d == 0:
                new_class = 0
            if q == 0 and d == 1:
                new_class = 1
            if q == 0 and d == 2:
                new_class = 2

            if q == 1 and d == 0:
                new_class = 3
            if q == 1 and d == 1:
                new_class = 4
            if q == 1 and d == 2:
                new_class = 5

            if q == 2 and d == 0:
                new_class = 6
            if q == 2 and d == 1:
                new_class = 7
            if q == 2 and d == 2:
                new_class = 8

            new_y.append(new_class)
        return np.array(new_y)
