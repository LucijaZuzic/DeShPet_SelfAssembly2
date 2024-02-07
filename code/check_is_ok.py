
from sklearn.model_selection import StratifiedKFold
N_FOLDS_FIRST = 5
import numpy as np
from utils import DATA_PATH, predictions_name, set_seed, AP_DATA_PATH, SP_DATA_PATH, AP_SP_DATA_PATH, TSNE_SP_DATA_PATH, TSNE_AP_SP_DATA_PATH
import pandas as pd
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

    #lines_dict["ROC thr = "].append(rocthr)
   # lines_dict["gmean (0.5) = "].append(
    #    returnGMEAN(test_labels, model_predictions_binary)
  #  )
    lines_dict["gmean (PR thr) = "].append(
        returnGMEAN(test_labels, model_predictions_binary_thrPR)
    )
  #  lines_dict["gmean (ROC thr) = "].append(
  #      returnGMEAN(test_labels, model_predictions_binary_thrROC)
  #  )
  #  lines_dict["ROC AUC = "].append(roc_auc_score(test_labels, model_predictions))
   # lines_dict["Accuracy (ROC thr) = "].append(
   #     my_accuracy_calculate(test_labels, model_predictions, rocthr)
   # ) 

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

    #lines_dict["PR thr = "].append(prthr)
    #lines_dict["PR AUC = "].append(auc(recall, precision))
    #lines_dict["F1 (0.5) = "].append(f1_score(test_labels, model_predictions_binary))
    lines_dict["F1 (PR thr) = "].append(
        f1_score(test_labels, model_predictions_binary_thrPR)
    )
   # lines_dict["F1 (ROC thr) = "].append(
   #     f1_score(test_labels, model_predictions_binary_thrROC)
   # )
    lines_dict["Accuracy (PR thr) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, prthr)
    )
    #lines_dict["Accuracy (0.5) = "].append(
     #   my_accuracy_calculate(test_labels, model_predictions, 0.5)
    #)

# Algorithm settings
N_FOLDS_FIRST = 5
SA_data = np.load(DATA_PATH + "data_SA_updated.npy", allow_pickle=True).item()
sequences = []
labels = []
MAX_LEN = 24

for peptide in SA_data:
    if SA_data[peptide] != "1":
        continue
    if len(peptide) > MAX_LEN or SA_data[peptide] == "-1":
        continue
    sequences.append(peptide)
    labels.append(SA_data[peptide])

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276] 

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

is_in_test = dict()

for n in new:
    is_in_test[n] = dict()

def read_one_prediction(some_path, test_number):
    file = open(predictions_name(some_path, test_number), "r")
    lines = file.readlines()
    predictions = eval(lines[0])
    labels = eval(lines[1])
    file.close()
    return predictions, labels

for some_seed in seed_list: 

    test_number = 0
    for test_number in range(N_FOLDS_FIRST):
        train_name = (
            "../seeds/seed_"
            + str(some_seed)
            + "/similarity/"
            + "train_fold_"
            + str(test_number + 1) + ".csv"
        )
        test_name = (
            "../seeds/seed_"
            + str(some_seed)
            + "/similarity/"
            + "test_fold_"
            + str(test_number + 1) + ".csv"
        )
        file_train = pd.read_csv(train_name)
        file_test = pd.read_csv(test_name)
        test_seqs = list(file_test["sequence"])
        train_seqs = list(file_train["sequence"]) 
        train_test_label = []
        for sq in new:
            if sq in test_seqs:
                train_test_label.append("test")
                is_in_test[sq][some_seed] = (test_number + 1, test_seqs.index(sq))
            if sq in train_seqs:
                train_test_label.append("train")
        print(some_seed, test_number + 1, train_test_label.count("train"), train_test_label.count("test"), len(train_seqs), len(test_seqs), train_test_label.count("train") / len(train_seqs), train_test_label.count("test") / len(test_seqs))
 
paths = [
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
]

predict_pep_path = dict()
for pep in is_in_test:
    predict_pep_path[pep] = dict()
    for some_path in paths:
        predict_pep_path[pep][some_path] = dict()
        for some_seed in seed_list:
            predict_pep_path[pep][some_path][some_seed] = dict()
for pep in is_in_test:
    for some_seed in seed_list:
        test_number, ix = is_in_test[pep][some_seed]
        set_seed(some_seed)
        for some_path in paths:
            all_predictions, all_labels = read_one_prediction(some_path, test_number) 
            predict_pep_path[pep][some_path][some_seed] = (all_predictions[ix], all_labels[ix])
 
vals_in_lines = [   
    "gmean (PR thr) = ",
    "F1 (PR thr) = ",
    "Accuracy (PR thr) = ", 
]
       
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
lines_dict_all = dict()
for some_path in paths:  
    lines_dict_all[some_path] = dict()  
 
for some_path in paths: 
    lines_dict = dict()
        
    for val in vals_in_lines:
        lines_dict[val] = []

    for some_seed in seed_list:

        preds = []
        labs = []

        for pep in new:
            
            pred, lab = predict_pep_path[pep][some_path][some_seed] 
            preds.append(pred)
            labs.append(lab)

        read_PR(labs, preds, lines_dict, PRthr[some_path], ROCthr[some_path])
        read_ROC(labs, preds, lines_dict, PRthr[some_path], ROCthr[some_path])
 

    for val in lines_dict: 
        lines_dict_all[some_path][val] = np.round(np.average(lines_dict[val]), 3)

for p in lines_dict_all:  
    print(p, lines_dict_all[p])
