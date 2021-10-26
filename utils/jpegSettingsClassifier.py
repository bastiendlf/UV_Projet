import numpy as np
from utils.training import load_model_and_label_encoder, load_model
import pickle

import os


class JpegSettingsClassifier:

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
        return self.label_encoder_Q.transform([f"Q{dict_settings['Q']}" for dict_settings in list_dict_ground_truth])

    def get_y_q_labels(self, y_q):
        return self.label_encoder_Q.inverse_transform(y_q)

    def predict_q(self, X):
        return self.model_Q.predict(X)

    def get_y_true_d(self, list_dict_ground_truth):
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
        return self.label_encoder_d.inverse_transform(y_d)

    def get_y_multiclass(self, y_q, y_d):
        y_multiclass = []
        for q, d in zip(y_q, y_d):
            y_multiclass.append([q, d])
        return np.array(y_multiclass)

    def get_well_predicted_Q(self, y_true_multiclass, y_pred_multiclass):
        return np.sum(np.equal(y_true_multiclass[:, 0], y_pred_multiclass[:, 0])) / float(
            y_true_multiclass.shape[0])

    def get_well_predicted_d(self, y_true_multiclass, y_pred_multiclass):
        return np.sum(np.equal(y_true_multiclass[:, 1], y_pred_multiclass[:, 1])) / float(
            y_true_multiclass.shape[0])

    def get_well_predicted_Q_and_d(self, y_true_multiclass, y_pred_multiclass):
        compare = y_true_multiclass == y_pred_multiclass
        good_Q_and_d_number = len([element for element in compare if (element[0] and element[1])])
        return good_Q_and_d_number / float(y_true_multiclass.shape[0])
