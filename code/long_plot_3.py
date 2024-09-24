import pandas as pd
import numpy as np
from scipy import stats
import sklearn
import matplotlib.pyplot as plt
from load_data import MAX_LEN
from utils import (
    predictions_longest_name,
    scatter_name_long,
    PATH_TO_NAME,
    DATA_PATH,
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
)
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    auc,
    f1_score,
)
from custom_plots import my_accuracy_calculate, weird_division, convert_to_binary

plt.rcParams.update({"font.size": 22})
PRthr = {
    AP_DATA_PATH: 0.2581173828,
    SP_DATA_PATH: 0.31960518800000004,
    AP_SP_DATA_PATH: 0.3839948796,
    TSNE_SP_DATA_PATH: 0.30602415759999996,
    TSNE_AP_SP_DATA_PATH: 0.34321978799999997,
}
ROCthr = {
    AP_DATA_PATH: 0.5024869916000001,
    SP_DATA_PATH: 0.5674178616000001,
    AP_SP_DATA_PATH: 0.5708274524,
    TSNE_SP_DATA_PATH: 0.566148754,
    TSNE_AP_SP_DATA_PATH: 0.5588111376,
}


def myfunc(x):
    return slope * x + intercept


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


def read_ROC(test_labels, model_predictions, lines_dict, oldPR, oldROC):
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

    model_predictions_binary_thrPR_new = convert_to_binary(model_predictions, oldPR)
    model_predictions_binary_thrROC_new = convert_to_binary(model_predictions, oldROC)

    model_predictions_binary = convert_to_binary(model_predictions, 0.5)

    lines_dict["ROC thr = "].append(oldROC)
    lines_dict["ROC AUC = "].append(roc_auc_score(test_labels, model_predictions))
    lines_dict["gmean (0.5) = "].append(
        returnGMEAN(test_labels, model_predictions_binary)
    )
    lines_dict["gmean (PR thr) = "].append(
        returnGMEAN(test_labels, model_predictions_binary_thrPR_new)
    )
    lines_dict["gmean (ROC thr) = "].append(
        returnGMEAN(test_labels, model_predictions_binary_thrROC_new)
    )
    lines_dict["Accuracy (ROC thr) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, oldROC)
    )


def read_PR(test_labels, model_predictions, lines_dict, oldPR, oldROC):
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

    model_predictions_binary_thrPR_new = convert_to_binary(model_predictions, oldPR)
    model_predictions_binary_thrROC_new = convert_to_binary(model_predictions, oldROC)
    model_predictions_binary = convert_to_binary(model_predictions, 0.5)

    lines_dict["PR thr = "].append(oldPR)
    lines_dict["PR AUC = "].append(auc(recall, precision))
    lines_dict["F1 (0.5) = "].append(f1_score(test_labels, model_predictions_binary))
    lines_dict["F1 (PR thr) = "].append(
        f1_score(test_labels, model_predictions_binary_thrPR_new)
    )
    lines_dict["F1 (ROC thr) = "].append(
        f1_score(test_labels, model_predictions_binary_thrROC_new)
    )
    lines_dict["Accuracy (PR thr) = "].append(
        my_accuracy_calculate(test_labels, model_predictions_binary_thrPR_new, oldPR)
    )
    lines_dict["Accuracy (0.5) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, 0.5)
    )


offset = 1
properties = np.ones(95)
properties[0] = 0
masking_value = 2
df = pd.read_csv(
    DATA_PATH + "collection_of_peptide_data.csv"
)

dict_hex = {}
actual_AP = []
duplicate_list = []
duplicate_AP_list = []
threshold = 1.75
for ix in range(len(df["Feature"])):
    if df["Feature"][ix] not in dict_hex:
        dict_hex[df["Feature"][ix]] = "1"
        actual_AP.append(df["Label"][ix])
    else:
        duplicate_list.append(df["Feature"][ix])
        duplicate_AP_list.append(df["Label"][ix])
for val in duplicate_list:
    ix = list(dict_hex.keys()).index(val)
    ix_dup = duplicate_list.index(list(dict_hex.keys())[ix])
    error_status = (actual_AP[ix] < threshold) != (duplicate_AP_list[ix_dup] < threshold)
    print(error_status, list(dict_hex.keys())[ix], actual_AP[ix], duplicate_AP_list[ix_dup])
    actual_AP[ix] = np.average([actual_AP[ix], duplicate_AP_list[ix_dup]])

seq_example = ""
for i in range(MAX_LEN):
    seq_example += "A"
dict_hex[seq_example] = "1"

test_labels = []
for i in actual_AP:
    if i < threshold:
        test_labels.append(0)
    else:
        test_labels.append(1)

path_list = [
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
]

vals_in_lines = [
    "ROC AUC = ",
    "PR AUC = ",
    "gmean (0.5) = ",
    "F1 (0.5) = ",
    "Accuracy (0.5) = ",
    "ROC thr = ",
    "PR thr = ",
    "gmean (ROC thr) = ",
    "F1 (ROC thr) = ",
    "Accuracy (ROC thr) = ",
    "gmean (PR thr) = ",
    "F1 (PR thr) = ",
    "Accuracy (PR thr) = ",
]

dict_csv_data = dict()
colnames = ["Metric"]
model_order = {"AP": "AP", "SP": "SP", "AP_SP": "AP-SP", "TSNE_SP": "t-SNE SP", "TSNE_AP_SP": "t-SNE AP-SP"}
for model in model_order:
    colnames.append(model_order[model])
for c in colnames:
    dict_csv_data[c] = []
dict_csv_data["Metric"].append("Pearson")
dict_csv_data["Metric"].append("Spearman")

lines_dict = dict()
for val in vals_in_lines:
    lines_dict[val] = []

for some_path in path_list:
    model = some_path.replace("../", "").replace("/", "").replace("_model_data", "").replace("_data", "")
    fileopen = open(predictions_longest_name(some_path), "r")
    predictions = eval(fileopen.readlines()[0].replace("\n", ""))
    predictions = predictions[:-1]
    fileopen.close()
    slope, intercept, r, p, std_err = stats.linregress(predictions, actual_AP)
    print(some_path)
    print("R: " + str(r))
    print("corrcoef: " + str(np.corrcoef(predictions, actual_AP)[0][1]))
    dict_csv_data[model_order[model]].append(np.corrcoef(predictions, actual_AP)[0][1])
    print("spearmanr: " + str(stats.spearmanr(predictions, actual_AP)[0]))
    dict_csv_data[model_order[model]].append(stats.spearmanr(predictions, actual_AP)[0])
    print("R2: " + str(sklearn.metrics.r2_score(predictions, actual_AP)))

    read_ROC(test_labels, predictions, lines_dict, PRthr[some_path], ROCthr[some_path])
    read_PR(test_labels, predictions, lines_dict, PRthr[some_path], ROCthr[some_path])

    mymodel = list(map(myfunc, predictions))
    plt.title(PATH_TO_NAME[some_path] + " model")
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("AP")
    plt.plot(predictions, mymodel, color="#ff120a")
    plt.scatter(predictions, actual_AP, color="#2e85ff")
    plt.savefig(scatter_name_long(some_path).replace("long", "longavg"), bbox_inches="tight")
    plt.savefig(scatter_name_long(some_path).replace("long", "longavg").replace(".png", "") + ".svg", bbox_inches="tight")
    plt.savefig(scatter_name_long(some_path).replace("long", "longavg").replace(".png", "") + ".pdf", bbox_inches="tight")
    plt.close()

for x in lines_dict:
    if "thr" in x and ")" not in x:
        continue
    print(x)
    print(np.round(lines_dict[x], 3))
    dict_csv_data["Metric"].append(x.replace(" = ", ""))
    ixpth = 0
    for v in lines_dict[x]:
        model = path_list[ixpth].replace("../", "").replace("/", "").replace("_model_data", "").replace("_data", "")
        dict_csv_data[model_order[model]].append(v)
        ixpth += 1

print(dict_csv_data)
df_new = pd.DataFrame(dict_csv_data)
df_new.to_csv("review/newest_data_avg.csv")