import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    auc,
    f1_score,
)
from custom_plots import (
    my_accuracy_calculate,
    weird_division,
    convert_to_binary,
)
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

def read_ROC(test_labels, model_predictions, lines_dict, prthr, rocthr):
    # Get false positive rate and true positive rate.
    fpr, tpr, thresholds = roc_curve(test_labels, model_predictions)

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ix = np.argmax(gmeans)

    model_predictions_binary_thrROC = convert_to_binary(
        model_predictions, rocthr
    )

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

    model_predictions_binary_thrPR = convert_to_binary(
        model_predictions, prthr
    )

    model_predictions_binary = convert_to_binary(model_predictions, 0.5)

    lines_dict["ROC thr = "].append(rocthr)
    lines_dict["gmean (0.5) = "].append(
        returnGMEAN(test_labels, model_predictions_binary)
    )
    lines_dict["gmean (PR thr) = "].append(
        returnGMEAN(test_labels, model_predictions_binary_thrPR)
    )
    lines_dict["gmean (ROC thr) = "].append(
        returnGMEAN(test_labels, model_predictions_binary_thrROC)
    )
    lines_dict["ROC AUC = "].append(roc_auc_score(test_labels, model_predictions))
    lines_dict["Accuracy (ROC thr) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, rocthr)
    )


def read_PR(test_labels, model_predictions, lines_dict, prthr, rocthr):
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
    model_predictions_binary_thrPR = convert_to_binary(
        model_predictions, prthr
    )
    model_predictions_binary = convert_to_binary(model_predictions, 0.5)

    # Get false positive rate and true positive rate.
    fpr, tpr, thresholdsROC = roc_curve(test_labels, model_predictions)

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ixROC = np.argmax(gmeans)

    model_predictions_binary_thrROC = convert_to_binary(
        model_predictions, rocthr
    )

    lines_dict["PR thr = "].append(prthr)
    lines_dict["PR AUC = "].append(auc(recall, precision))
    lines_dict["F1 (0.5) = "].append(f1_score(test_labels, model_predictions_binary))
    lines_dict["F1 (PR thr) = "].append(
        f1_score(test_labels, model_predictions_binary_thrPR)
    )
    lines_dict["F1 (ROC thr) = "].append(
        f1_score(test_labels, model_predictions_binary_thrROC)
    )
    lines_dict["Accuracy (PR thr) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, prthr)
    )
    lines_dict["Accuracy (0.5) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, 0.5)
    )
params_for_model = {"AP": 1, "SP": 7, "AP_SP": 1, "TSNE_SP": 5, "TSNE_AP_SP": 8}
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
PRthr = {
    "AP": 0.2581173828,
    "SP": 0.31960518800000004,
    "AP_SP": 0.3839948796,
    "TSNE_SP": 0.30602415759999996,
    "TSNE_AP_SP": 0.34321978799999997,
}
ROCthr = {
    "AP": 0.5024869916000001,
    "SP": 0.5674178616000001,
    "AP_SP": 0.5708274524,
    "TSNE_SP": 0.566148754,
    "TSNE_AP_SP": 0.5588111376,
}
vals_in_lines = [
    "ROC thr = ",
    "PR thr = ",
    "ROC AUC = ",
    "gmean (ROC thr) = ",
    "F1 (ROC thr) = ",
    "Accuracy (ROC thr) = ",
    "PR AUC = ",
    "gmean (PR thr) = ",
    "F1 (PR thr) = ",
    "Accuracy (PR thr) = ",
    "gmean (0.5) = ",
    "F1 (0.5) = ",
    "Accuracy (0.5) = ",
]
def filter_dict(minlen, maxlen):
    models_line_dicts = dict()
    for model_name in os.listdir("../seeds/seed_369953070"):
        if "similarity" in model_name:
            continue
        #print(model_name.replace("_model_data", "").replace("_data", ""))
        num_par = 3
        if "SP" in model_name:
            num_par *= 3
        #for params_num in range(1, num_par):
        params_num = params_for_model[model_name.replace("_model_data", "").replace("_data", "")]
        lines_dict = dict()
        for v in vals_in_lines:
            lines_dict[v] = [] 
        for seed_val_ix in range(len(seed_list)):
            seed_val = seed_list[seed_val_ix]
            #print(seed_val)
            pred_arr1_filter = []
            pred_arr2_filter = []
            seqs_filter = []
            labs_filter = []
            for test_num in range(1, 6):
                dir_pred = "../seeds/seed_" + str(seed_val) + "/" + model_name + "/" + model_name.replace("model_data", "test_" + str(test_num)).replace("data", "test_" + str(test_num))
                csv_seed_test_fold = pd.read_csv("../seeds/seed_" + str(seed_val) + "/similarity/test_fold_" + str(test_num) + ".csv", index_col = False)
                seqs = list(csv_seed_test_fold["sequence"])
                labs = list(csv_seed_test_fold["label"])
                pred_file = open(dir_pred + "/" + model_name.replace("model_data", "test_" + str(test_num)).replace("data", "test_" + str(test_num)) + "_predictions.txt", "r")
                pred_arrs = pred_file.readlines()
                pred_arr1 = eval(pred_arrs[0])
                pred_arr2 = eval(pred_arrs[1])
                for seq_ix in range(len(seqs)):
                    if len(seqs[seq_ix]) > maxlen or len(seqs[seq_ix]) < minlen:
                        continue
                    pred_arr1_filter.append(pred_arr1[seq_ix])
                    pred_arr2_filter.append(pred_arr2[seq_ix])
                    seqs_filter.append(seqs[seq_ix])
                    labs_filter.append(labs[seq_ix])
            read_PR(labs_filter, pred_arr1_filter, lines_dict, PRthr[model_name.replace("_model_data", "").replace("_data", "")], ROCthr[model_name.replace("_model_data", "").replace("_data", "")])
            read_ROC(labs_filter, pred_arr1_filter, lines_dict, PRthr[model_name.replace("_model_data", "").replace("_data", "")], ROCthr[model_name.replace("_model_data", "").replace("_data", "")])
        #print(lines_dict)
        models_line_dicts[model_name.replace("_model_data", "").replace("_data", "")] = lines_dict
    return models_line_dicts
models_line_dicts1 = filter_dict(3, 9)
print(models_line_dicts1["AP_SP"]["Accuracy (PR thr) = "])
print(models_line_dicts1["AP"]["Accuracy (PR thr) = "])
models_line_dicts1 = filter_dict(2, 15)
print(models_line_dicts1["AP_SP"]["Accuracy (PR thr) = "])
print(models_line_dicts1["AP"]["Accuracy (PR thr) = "])