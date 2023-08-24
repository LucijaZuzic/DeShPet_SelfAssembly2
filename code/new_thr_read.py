import numpy as np
from merge_data import merge_data
from sklearn.model_selection import StratifiedKFold
from load_data import load_data_SA
from automate_training import data_and_labels_from_indices
from utils import (
    set_seed,
    predictions_thr_name,
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
    DATA_PATH,
)
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    auc,
    f1_score,
)
from custom_plots import my_accuracy_calculate, weird_division, convert_to_binary


def returnGMEAN(actual, pred):
    tn = 0
    tp = 0
    apo = 0
    ane = 0
    for i in range(len(pred)):
        a = actual[i]
        p = pred[i]
        if a == 1:
            apo += 1
        else:
            ane += 1
        if p == a:
            if a == 1:
                tp += 1
            else:
                tn += 1

    return np.sqrt(tp / apo * tn / ane)


def read_ROC(test_labels, model_predictions, lines_dict):
    # Get false positive rate and true positive rate.
    fpr, tpr, thresholds = roc_curve(test_labels, model_predictions)

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ix = np.argmax(gmeans)

    # Get recall and precision.
    precision, recall, thresholdsPR = precision_recall_curve(
        test_labels, model_predictions
    )

    # Calculate the F1 score for each threshold
    fscore = []
    for i in range(len(precision)):
        fscore.append(
            weird_division(2 * precision[i] * recall[i], precision[i] + recall[i])
        )

    # Locate the index of the largest F1 score
    ixPR = np.argmax(fscore)

    model_predictions_binary_thrPR_new = convert_to_binary(
        model_predictions, thresholdsPR[ixPR]
    )
    model_predictions_binary_thrROC_new = convert_to_binary(
        model_predictions, thresholds[ix]
    )

    model_predictions_binary = convert_to_binary(model_predictions, 0.5)

    lines_dict["ROC thr new = "].append(thresholds[ix])
    lines_dict["ROC AUC = "].append(roc_auc_score(test_labels, model_predictions))
    lines_dict["gmean (0.5) = "].append(
        returnGMEAN(test_labels, model_predictions_binary)
    )
    lines_dict["gmean (PR thr new) = "].append(
        returnGMEAN(test_labels, model_predictions_binary_thrPR_new)
    )
    lines_dict["gmean (ROC thr new) = "].append(
        returnGMEAN(test_labels, model_predictions_binary_thrROC_new)
    )
    lines_dict["Accuracy (ROC thr new) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, thresholds[ix])
    )


def read_PR(test_labels, model_predictions, lines_dict):
    # Get recall and precision.
    precision, recall, thresholds = precision_recall_curve(
        test_labels, model_predictions
    )

    # Calculate the F1 score for each threshold
    fscore = []
    for i in range(len(precision)):
        fscore.append(
            weird_division(2 * precision[i] * recall[i], precision[i] + recall[i])
        )

    # Locate the index of the largest F1 score
    ix = np.argmax(fscore)

    # Get false positive rate and true positive rate.
    fpr, tpr, thresholdsROC = roc_curve(test_labels, model_predictions)

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ixROC = np.argmax(gmeans)

    model_predictions_binary_thrPR_new = convert_to_binary(
        model_predictions, thresholds[ix]
    )
    model_predictions_binary_thrROC_new = convert_to_binary(
        model_predictions, thresholdsROC[ixROC]
    )
    model_predictions_binary = convert_to_binary(model_predictions, 0.5)

    lines_dict["PR thr new = "].append(thresholds[ix])
    lines_dict["PR AUC = "].append(auc(recall, precision))
    lines_dict["F1 (0.5) = "].append(f1_score(test_labels, model_predictions_binary))
    lines_dict["F1 (PR thr new) = "].append(
        f1_score(test_labels, model_predictions_binary_thrPR_new)
    )
    lines_dict["F1 (ROC thr new) = "].append(
        f1_score(test_labels, model_predictions_binary_thrROC_new)
    )
    lines_dict["Accuracy (PR thr new) = "].append(
        my_accuracy_calculate(
            test_labels, model_predictions_binary_thrPR_new, thresholds[ix]
        )
    )
    lines_dict["Accuracy (0.5) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, 0.5)
    )


seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
path_list = [
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
]

N_FOLDS_FIRST = 5
N_FOLDS_SECOND = 5
offset = 1
properties = np.ones(95)
properties[0] = 0
masking_value = 2

vals_in_lines = [
    "ROC AUC = ",
    "PR AUC = ",
    "gmean (0.5) = ",
    "F1 (0.5) = ",
    "Accuracy (0.5) = ",
    "ROC thr new = ",
    "PR thr new = ",
    "gmean (ROC thr new) = ",
    "F1 (ROC thr new) = ",
    "Accuracy (ROC thr new) = ",
    "gmean (PR thr new) = ",
    "F1 (PR thr new) = ",
    "Accuracy (PR thr new) = ",
]

for some_path in path_list:
    print(some_path)
    lines_dict_avg_avg = dict()
    for val in vals_in_lines:
        lines_dict_avg_avg[val] = []
    for some_seed in seed_list:
        set_seed(some_seed)
        if some_path == AP_DATA_PATH:
            params_nr = 1
        if some_path == SP_DATA_PATH:
            params_nr = 7
        if some_path == AP_SP_DATA_PATH:
            params_nr = 1
        if some_path == TSNE_SP_DATA_PATH:
            params_nr = 5
        if some_path == TSNE_AP_SP_DATA_PATH:
            params_nr = 8

        # Algorithm settings
        names = ["AP"]
        if some_path == SP_DATA_PATH or some_path == TSNE_SP_DATA_PATH:
            names = []

        SA_data = np.load(DATA_PATH + "data_SA_updated.npy", allow_pickle=True).item()

        SA, NSA = load_data_SA(
            some_path, SA_data, names, offset, properties, masking_value
        )

        # Merge SA nad NSA data the train and validation subsets.
        all_data, all_labels = merge_data(some_path, SA, NSA)

        # Define N-fold cross validation test harness for splitting the test data from the train and validation data
        kfold_first = StratifiedKFold(
            n_splits=N_FOLDS_FIRST, shuffle=True, random_state=some_seed
        )
        # Define N-fold cross validation test harness for splitting the validation from the train data
        kfold_second = StratifiedKFold(
            n_splits=N_FOLDS_SECOND, shuffle=True, random_state=some_seed
        )

        test_number = 0

        lines_dict_avg = dict()
        for val in vals_in_lines:
            lines_dict_avg[val] = []

        for train_and_validation_data_indices, test_data_indices in kfold_first.split(
            all_data, all_labels
        ):
            test_number += 1

            allpreds = []
            alllabels = []

            lines_dict = dict()
            for val in vals_in_lines:
                lines_dict[val] = []

                # Convert train and validation indices to train and validation data and train and validation labels
                (
                    train_and_validation_data,
                    train_and_validation_labels,
                ) = data_and_labels_from_indices(
                    all_data, all_labels, train_and_validation_data_indices
                )

                fold_nr = 0

                for train_data_indices, validation_data_indices in kfold_second.split(
                    train_and_validation_data, train_and_validation_labels
                ):
                    fold_nr += 1

                    # Convert validation indices to validation data and validation labels
                    validation_data, validation_labels = data_and_labels_from_indices(
                        train_and_validation_data,
                        train_and_validation_labels,
                        validation_data_indices,
                    )

                    predictions_file_all = open(
                        predictions_thr_name(
                            some_path, test_number, params_nr, fold_nr
                        ),
                        "r",
                    )
                    predictions_file_lines_all = eval(
                        predictions_file_all.readlines()[0].replace("\n", "")
                    )
                    predictions_file_all.close()
                    # print(len(predictions_file_lines_all))

                    for i in range(len(predictions_file_lines_all)):
                        alllabels.append(validation_labels[i])
                        allpreds.append(predictions_file_lines_all[i])

                read_ROC(alllabels, allpreds, lines_dict)
                read_PR(alllabels, allpreds, lines_dict)

                for x in lines_dict:
                    lines_dict_avg[x].append(lines_dict[x][0])

            for x in lines_dict_avg:
                avgval = 0
                for y in lines_dict_avg[x]:
                    avgval += y
                avgval /= len(lines_dict_avg[x])
                lines_dict_avg[x] = [avgval]

            for x in lines_dict_avg:
                lines_dict_avg_avg[x].append(lines_dict_avg[x][0])

    for x in lines_dict_avg_avg:
        avgval = 0
        for y in lines_dict_avg_avg[x]:
            avgval += y
        avgval /= len(lines_dict_avg_avg[x])
        lines_dict_avg_avg[x] = [avgval]

    str1 = ""
    str2 = ""
    for x in lines_dict_avg_avg:
        str1 += x + " "
        if x != "PR thr new = " and x != "ROC thr new = ":
            str2 += str(np.round(lines_dict_avg_avg[x][0], 3)) + " "
        else:
            str2 += str(lines_dict_avg_avg[x][0]) + " "
    print(str1.replace(" thr new)", ")").replace(" = ", ""))
    print(str2)
