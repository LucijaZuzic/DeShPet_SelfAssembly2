import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
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

    return np.sqrt(weird_division(tp, apo) * weird_division(tn, ane))

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

    model_predictions_binary_thrPR = convert_to_binary(
        model_predictions, prthr
    )
    # Get false positive rate and true positive rate.
    fpr, tpr, thresholdsROC = roc_curve(test_labels, model_predictions)

    lines_dict["F1 (PR thr) = "].append(
        f1_score(test_labels, model_predictions_binary_thrPR)
    )
    lines_dict["Accuracy (PR thr) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, prthr)
    )
    lines_dict["gmean (PR thr) = "].append(
        returnGMEAN(test_labels, model_predictions_binary_thrPR)
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
    "gmean (PR thr) = ",
    "F1 (PR thr) = ",
    "Accuracy (PR thr) = "
]
def filter_dict(minlen, maxlen):
    models_line_dicts = dict()
    for model_name in os.listdir("../seeds/seed_369953070"):
        if "similarity" in model_name:
            continue
        #print(model_name.replace("_model_data", "").replace("_data", ""))
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
            if not os.path.isdir("review/long/preds/" + str(minlen) + "_" + str(maxlen)+ "/" + model_name):
                os.makedirs("review/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name)
            model_predictions_binary_thrPR = convert_to_binary(pred_arr1_filter, PRthr[model_name.replace("_model_data", "").replace("_data", "")])
            model_predictions_binary_thrROC = convert_to_binary(pred_arr1_filter, ROCthr[model_name.replace("_model_data", "").replace("_data", "")])
            model_predictions_binary_50 = convert_to_binary(pred_arr1_filter, 0.5)
            df_newPR = pd.DataFrame({"preds": model_predictions_binary_thrPR, "labels": labs_filter, "feature": seqs_filter})
            df_newROC = pd.DataFrame({"preds": model_predictions_binary_thrROC, "labels": labs_filter, "feature": seqs_filter})
            df_new50 = pd.DataFrame({"preds": model_predictions_binary_50, "labels": labs_filter, "feature": seqs_filter})
            #df_newPR.to_csv("review/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_seed_" + str(seed_val) + "_PR_preds.csv")
            #df_newROC.to_csv("review/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_seed_" + str(seed_val) + "_ROC_preds.csv")
            #df_new50.to_csv("review/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_seed_" + str(seed_val) + "_50_preds.csv")
            read_PR(labs_filter, pred_arr1_filter, lines_dict, PRthr[model_name.replace("_model_data", "").replace("_data", "")], ROCthr[model_name.replace("_model_data", "").replace("_data", "")])
        #print(lines_dict)
        models_line_dicts[model_name.replace("_model_data", "").replace("_data", "")] = lines_dict
    return models_line_dicts

model_order = {"AP": "AP", "SP": "SP", "AP_SP": "AP-SP", "TSNE_SP": "t-SNE SP", "TSNE_AP_SP": "t-SNE AP-SP"}
def print_dict(mini, maxi):
    dicti = filter_dict(mini, maxi)
    linea = "Metric"
    colnames = ["Metric"]
    for model in model_order:
        linea += " & " + model_order[model]
        colnames.append(model_order[model])
    dict_csv_data = dict()
    for c in colnames:
        dict_csv_data[c] = []
    #print(linea + " \\\\ \\hline")
    for metric in dicti["AP"]:
        if "thr" in metric and ")" not in metric:
            continue
        linea = metric.replace(" = ", "")
        dict_csv_data["Metric"].append(metric.replace(" = ", ""))
        for model in model_order:
            rv = 1
            if "Acc" not in metric:
                rv = 3
            linea += " & " + str(np.round(np.average(dicti[model][metric]), rv))
            dict_csv_data[model_order[model]].append(np.average(dicti[model][metric]))
        #print(linea + " \\\\ \\hline")
    if not os.path.isdir("review/short"):
        os.makedirs("review/short")
    df_new = pd.DataFrame(dict_csv_data)
    #df_new.to_csv("review/short/" + str(mini) + "_" + str(maxi) + ".csv")
    return df_new

def return_lens():
    for model_name in os.listdir("../seeds/seed_369953070"):
        if "similarity" in model_name:
            continue
        for seed_val_ix in range(len(seed_list)):
            seed_val = seed_list[seed_val_ix]
            lens = dict()
            lens_positive = dict() 
            lens_negative = dict()
            for test_num in range(1, 6):
                csv_seed_test_fold = pd.read_csv("../seeds/seed_" + str(seed_val) + "/similarity/test_fold_" + str(test_num) + ".csv", index_col = False)
                seqs = list(csv_seed_test_fold["sequence"])
                labs = list(csv_seed_test_fold["label"])
                for seq_ix in range(len(seqs)):
                    if len(seqs[seq_ix]) not in lens:
                        lens[len(seqs[seq_ix])] = 0
                    lens[len(seqs[seq_ix])] += 1
                    if labs[seq_ix] == 1:
                        if len(seqs[seq_ix]) not in lens_positive:
                            lens_positive[len(seqs[seq_ix])] = 0
                        lens_positive[len(seqs[seq_ix])] += 1
                    else:
                        if len(seqs[seq_ix]) not in lens_negative:
                            lens_negative[len(seqs[seq_ix])] = 0
                        lens_negative[len(seqs[seq_ix])] += 1
            return(lens, lens_positive, lens_negative)
lens_all, lens_positive, lens_negative = return_lens()
print(lens_all)
print(lens_positive)
print(lens_negative)

dicti = {"Metric": ["Total", "Positive", "Negative"]}
for len1 in range(3, 24):
    if os.path.isfile("review/short/" + str(len1) + "_" + str(len1) + ".csv"):
        print("ok", len1)
        print(len1, lens_all[len1], lens_positive[len1], lens_negative[len1])
        filepd = pd.read_csv("review/short/" + str(len1) + "_" + str(len1) + ".csv")
        scores = filepd["AP-SP"]
        metrics = filepd["Metric"]
        dicti[str(len1)] = [lens_all[len1], lens_positive[len1], lens_negative[len1]]
        for ix in range(len(scores)):
            if "PR thr" in metrics[ix]:
                dicti[str(len1)].append(scores[ix])
                if metrics[ix] not in dicti["Metric"]:
                    dicti["Metric"].append(metrics[ix])
df_new = pd.DataFrame(dicti)
df_new.to_csv("review/all_new.csv")
print(dicti)

dicti = {"Metric": ["Total", "Positive", "Negative"]}
for len1 in range(3, 24):
    if len1 not in lens_all:
        continue
    vp = 0
    vn = 0
    if len1 in lens_positive:
        vp = lens_positive[len1]
    if len1 in lens_negative:
        vn = lens_negative[len1]
    print(len1, lens_all[len1], vp, vn)
    filepd = print_dict(len1, len1)
    scores = filepd["AP-SP"]
    metrics = filepd["Metric"]
    dicti[str(len1)] = [lens_all[len1], vp, vn]
    for ix in range(len(scores)):
        if "PR thr" in metrics[ix]:
            dicti[str(len1)].append(scores[ix])
            if metrics[ix] not in dicti["Metric"]:
                dicti["Metric"].append(metrics[ix])
df_new = pd.DataFrame(dicti)
df_new.to_csv("review/all_new_expand.csv")