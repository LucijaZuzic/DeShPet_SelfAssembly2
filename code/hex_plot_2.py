import pandas as pd
import numpy as np
from scipy import stats
import sklearn
import matplotlib.pyplot as plt
from load_data import MAX_LEN
from utils import (
    predictions_hex_name,
    scatter_name,
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

from matplotlib import rc
rc('font',**{'family':'Arial'})
cm = 1/2.54  # centimeters in inches
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

    lines_dict["ROC thr new = "].append(oldROC)
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

    lines_dict["PR thr new = "].append(oldPR)
    lines_dict["PR AUC = "].append(auc(recall, precision))
    lines_dict["F1 (0.5) = "].append(f1_score(test_labels, model_predictions_binary))
    lines_dict["F1 (PR thr new) = "].append(
        f1_score(test_labels, model_predictions_binary_thrPR_new)
    )
    lines_dict["F1 (ROC thr new) = "].append(
        f1_score(test_labels, model_predictions_binary_thrROC_new)
    )
    lines_dict["Accuracy (PR thr new) = "].append(
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
    DATA_PATH + "41557_2022_1055_MOESM3_ESM_Figure3a_5mer_score_shortMD.csv"
)


new = [ 
    "PTPCY",
    "PPPHY",
    "SYCGY",
    "KWMDF",
    "FFEKF",
    "KWEFY",
    "FKFEF",
    "RWLDY",
    "WKPYY",
    "VVVVV",
    "FKIDF",
    "VKVFF",
    "KFFFE",
    "KFAFD",
    "VKVEV",
    "RVSVD", 
    "KKFDD",
    "VKVKV",
    "KVKVK",
    "DPDPD",
]

dict_hex = {}
ix_to_skip = []
peps_skipped = []
ix = 0
for i in df["pep"]:
    find_err = False
    for n in new:
        if i.lower() in n.lower() or n.lower() in i.lower():
            find_err = True
    if find_err:
        ix_to_skip.append(ix)
        peps_skipped.append(i) 
    ix += 1
    dict_hex[i] = "1"
print(len(df["pep"]), len(df["pep"]) - len(ix_to_skip))

actual_AP = []
for i in df["AP"]:
    actual_AP.append(i)

seq_example = ""
for i in range(MAX_LEN):
    seq_example += "A"
dict_hex[seq_example] = "1"

test_labels = []
threshold = 1.75
for i in df["AP"]:
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
    "ROC thr new = ",
    "PR thr new = ",
    "gmean (ROC thr new) = ",
    "F1 (ROC thr new) = ",
    "Accuracy (ROC thr new) = ",
    "gmean (PR thr new) = ",
    "F1 (PR thr new) = ",
    "Accuracy (PR thr new) = ",
]

lines_dict = dict()
for val in vals_in_lines:
    lines_dict[val] = []

print(ix_to_skip, peps_skipped)

for some_path in path_list:
    fileopen = open(predictions_hex_name(some_path), "r")
    predictions = eval(fileopen.readlines()[0].replace("\n", ""))
    predictions = predictions[:-1]
    fileopen.close()

    predictions_NEW = []
    test_labels_NEW = []
    actual_AP_NEW = []
    for ix in range(len(predictions)):
        if ix not in ix_to_skip:
            predictions_NEW.append(predictions[ix])
            test_labels_NEW.append(test_labels[ix])
            actual_AP_NEW.append(actual_AP[ix])

    slope, intercept, r, p, std_err = stats.linregress(predictions_NEW, actual_AP_NEW)
    print(some_path)
    print("R: " + str(r))
    print("corrcoef: " + str(np.round(np.corrcoef(predictions_NEW, actual_AP_NEW)[0][1], 2)))
    print("spearmanr: " + str(np.round(stats.spearmanr(predictions_NEW, actual_AP_NEW)[0], 2)))
    print("R2: " + str(np.round(sklearn.metrics.r2_score(predictions_NEW, actual_AP_NEW), 2)))

    read_ROC(test_labels_NEW, predictions_NEW, lines_dict, PRthr[some_path], ROCthr[some_path])
    read_PR(test_labels_NEW, predictions_NEW, lines_dict, PRthr[some_path], ROCthr[some_path])

    mymodel = list(map(myfunc, predictions_NEW))
    rc('font',**{'family':'Arial'})
    #plt.rcParams.update({"font.size": 5})
    SMALL_SIZE = 5
    MEDIUM_SIZE = 5
    BIGGER_SIZE = 5

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.figure(figsize=(4.1*cm, 2.6*cm), dpi = 300)
    plt.title(PATH_TO_NAME[some_path] + " model")
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("AP")
    plt.plot(predictions_NEW, mymodel, linewidth = 1, color="#ff120a")
    plt.scatter(predictions_NEW, actual_AP_NEW, s = 2, color="#2e85ff")
    plt.savefig(scatter_name(some_path).replace(".png", "") + "_fixed.png", bbox_inches="tight")
    plt.savefig(scatter_name(some_path).replace(".png", "") + "_fixed.svg", bbox_inches="tight")
    plt.savefig(scatter_name(some_path).replace(".png", "") + "_fixed.pdf", bbox_inches="tight")
    plt.close()

for x in lines_dict:
    print(x)
    if "Acc" in x:
        print(np.round(lines_dict[x], 1))
    else:
        print(np.round(lines_dict[x], 3))