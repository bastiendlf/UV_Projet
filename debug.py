import utils.makeDataset as DatasetMaker
import os

from utils.jpegSettingsClassifier import JpegSettingsClassifier

if __name__ == '__main__':
    if os.getcwd().endswith("/Code"):
        os.chdir(os.path.join(os.getcwd(), "utils"))

    jpegSettingsClassifier = JpegSettingsClassifier()

    ground_truth, X_validation = DatasetMaker.load_dataset("validation")
    y_true_q = jpegSettingsClassifier.get_y_true_q(ground_truth)
    y_pred_q = jpegSettingsClassifier.predict_q(X_validation)

    y_true_d = jpegSettingsClassifier.get_y_true_d(ground_truth)
    y_pred_d = jpegSettingsClassifier.predict_d(X_validation, jpegSettingsClassifier.get_y_q_labels(y_pred_q))

    y_true_multiclass = jpegSettingsClassifier.get_y_multiclass(y_true_q, y_true_d)
    y_pred_multiclass = jpegSettingsClassifier.get_y_multiclass(y_pred_q, y_pred_d)
    print(f"Well predicted Qf : {jpegSettingsClassifier.get_well_predicted_Q(y_true_multiclass, y_pred_multiclass)}%")
    print(f"Well predicted d : {jpegSettingsClassifier.get_well_predicted_d(y_true_multiclass, y_pred_multiclass)}%")
    print(f"Well predicted Qf and d : {jpegSettingsClassifier.get_well_predicted_Q_and_d(y_true_multiclass, y_pred_multiclass)}%")

    y_true_multiclass_labels = jpegSettingsClassifier.get_y_multiclass(jpegSettingsClassifier.get_y_q_labels(y_true_q), jpegSettingsClassifier.get_y_d_labels(y_true_d))
    y_pred_multiclass_labels = jpegSettingsClassifier.get_y_multiclass(jpegSettingsClassifier.get_y_q_labels(y_pred_q), jpegSettingsClassifier.get_y_d_labels(y_pred_d))
