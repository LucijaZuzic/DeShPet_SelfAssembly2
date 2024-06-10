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

    return np.sqrt(weird_division(tp, apo) * weird_division(tn, ane))

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

df = pd.read_csv("genetic_peptides.csv")

new = pd.read_csv("text_low_0.csv")["Feature"]
dict_hex = {}
ix_to_skip = []
peps_skipped = []
ix = 0
for i in df["Feature"]:
    find_err = False
    for n in new:
        if i.lower() in n.lower() or n.lower() in i.lower():
            find_err = True
    if find_err:
        ix_to_skip.append(ix)
        peps_skipped.append(i) 
    ix += 1
    dict_hex[i] = "1"
print(len(df["Feature"]), len(df["Feature"]) - len(ix_to_skip))

actual_AP = []
for i in df["Label"]:
    actual_AP.append(i)

test_labels = []
threshold = 0.5
for i in df["Label"]:
    if i < threshold:
        test_labels.append(0)
    else:
        test_labels.append(1)
seqs_new = list(dict_hex.keys())

def return_lens():
    seqs = seqs_new
    lens = dict()
    for seq_ix in range(len(seqs)):
        if seq_ix not in ix_to_skip:
            continue
        if len(seqs[seq_ix]) not in lens:
            lens[len(seqs[seq_ix])] = 0
        lens[len(seqs[seq_ix])] += 1
    return(lens)
         
def count_classes(minlen, maxlen):
    seqs = seqs_new
    labs = test_labels
    classes = dict()
    for seq_ix in range(len(seqs)):
        if len(seqs[seq_ix]) > maxlen or len(seqs[seq_ix]) < minlen or seq_ix not in ix_to_skip:
            continue
        if labs[seq_ix] not in classes:
            classes[labs[seq_ix]] = 0
        classes[labs[seq_ix]] += 1
    return(classes)

def filter_dict(minlen, maxlen):
    models_line_dicts = dict()
    seqs = seqs_new
    labs = test_labels
    for model_name in os.listdir("../final"):
        lines_dict = dict()
        for v in vals_in_lines:
            lines_dict[v] = []
        pred_arr1_filter = []
        seqs_filter = []
        labs_filter = []
        short_model = model_name.replace("_model_data", "").replace("_data", "")
        pred_file = open("../final/" + model_name + "/" + short_model + "_predictions_genetic.txt", "r")
        pred_arrs = pred_file.readlines()
        pred_arr1 = eval(pred_arrs[0])
        for seq_ix in range(len(seqs)):
            if len(seqs[seq_ix]) > maxlen or len(seqs[seq_ix]) < minlen or seq_ix not in ix_to_skip:
                continue
            pred_arr1_filter.append(pred_arr1[seq_ix])
            seqs_filter.append(seqs[seq_ix])
            labs_filter.append(labs[seq_ix])
        if not os.path.isdir("review_genetic_low_0/long/preds/" + str(minlen) + "_" + str(maxlen)+ "/" + model_name):
            os.makedirs("review_genetic_low_0/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name)
        model_predictions_binary_thrPR = convert_to_binary(pred_arr1_filter, PRthr[model_name.replace("_model_data", "").replace("_data", "")])
        model_predictions_binary_thrROC = convert_to_binary(pred_arr1_filter, ROCthr[model_name.replace("_model_data", "").replace("_data", "")])
        model_predictions_binary_50 = convert_to_binary(pred_arr1_filter, 0.5)
        df_newPR = pd.DataFrame({"preds": model_predictions_binary_thrPR, "labels": labs_filter, "feature": seqs_filter})
        df_newROC = pd.DataFrame({"preds": model_predictions_binary_thrROC, "labels": labs_filter, "feature": seqs_filter})
        df_new50 = pd.DataFrame({"preds": model_predictions_binary_50, "labels": labs_filter, "feature": seqs_filter})
        df_newPR.to_csv("review_genetic_low_0/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_PR_preds.csv")
        df_newROC.to_csv("review_genetic_low_0/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_ROC_preds.csv")
        df_new50.to_csv("review_genetic_low_0/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_50_preds.csv")
        read_PR(labs_filter, pred_arr1_filter, lines_dict, PRthr[model_name.replace("_model_data", "").replace("_data", "")], ROCthr[model_name.replace("_model_data", "").replace("_data", "")])
        read_ROC(labs_filter, pred_arr1_filter, lines_dict, PRthr[model_name.replace("_model_data", "").replace("_data", "")], ROCthr[model_name.replace("_model_data", "").replace("_data", "")])
        #print(lines_dict)
        models_line_dicts[model_name.replace("_model_data", "").replace("_data", "")] = lines_dict
    return models_line_dicts

def filter_dict_no_lines(minlen, maxlen):
    seqs = seqs_new
    labs = test_labels
    for model_name in os.listdir("../final"):
        pred_arr1_filter = []
        seqs_filter = []
        labs_filter = []
        short_model = model_name.replace("_model_data", "").replace("_data", "")
        pred_file = open("../final/" + model_name + "/" + short_model + "_predictions_genetic.txt", "r")
        pred_arrs = pred_file.readlines()
        pred_arr1 = eval(pred_arrs[0])
        for seq_ix in range(len(seqs)):
            if len(seqs[seq_ix]) > maxlen or len(seqs[seq_ix]) < minlen or seq_ix not in ix_to_skip:
                continue
            pred_arr1_filter.append(pred_arr1[seq_ix])
            seqs_filter.append(seqs[seq_ix])
            labs_filter.append(labs[seq_ix])
        if not os.path.isdir("review_genetic_low_0/long/preds/" + str(minlen) + "_" + str(maxlen)+ "/" + model_name):
            os.makedirs("review_genetic_low_0/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name)
        model_predictions_binary_thrPR = convert_to_binary(pred_arr1_filter, PRthr[model_name.replace("_model_data", "").replace("_data", "")])
        model_predictions_binary_thrROC = convert_to_binary(pred_arr1_filter, ROCthr[model_name.replace("_model_data", "").replace("_data", "")])
        model_predictions_binary_50 = convert_to_binary(pred_arr1_filter, 0.5)
        df_newPR = pd.DataFrame({"preds": model_predictions_binary_thrPR, "labels": labs_filter, "feature": seqs_filter})
        df_newROC = pd.DataFrame({"preds": model_predictions_binary_thrROC, "labels": labs_filter, "feature": seqs_filter})
        df_new50 = pd.DataFrame({"preds": model_predictions_binary_50, "labels": labs_filter, "feature": seqs_filter})
        df_newPR.to_csv("review_genetic_low_0/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_PR_preds.csv")
        df_newROC.to_csv("review_genetic_low_0/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_ROC_preds.csv")
        df_new50.to_csv("review_genetic_low_0/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_50_preds.csv")

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
    if not os.path.isdir("review_genetic_low_0/short"):
        os.makedirs("review_genetic_low_0/short")
    df_new = pd.DataFrame(dict_csv_data)
    df_new.to_csv("review_genetic_low_0/short/" + str(mini) + "_" + str(maxi) + ".csv")

lens = return_lens()
for lena in sorted(lens.keys()):
    print(lena, lens[lena])
larger = []
for lena in sorted(lens.keys()):
    print(lena, count_classes(lena, lena), lens[lena])
    if len(count_classes(lena, lena)) > 1:
        larger.append(lena)
print(larger)

for a in sorted(lens.keys()):
    for b in sorted(lens.keys()):
        if b < a:
            continue
        filter_dict_no_lines(a, b)
        rnge = list(range(a, b + 1))
        is_range_ok = False
        for r in rnge:
            if r in larger:
                is_range_ok = True
                break
        if not is_range_ok:
            continue
        print_dict(a, b)
        print(min(lens), a - 1, count_classes(min(lens), a - 1), a, b, count_classes(a, b), b + 1, max(lens), count_classes(b + 1, max(lens)))