from utils import (
    set_seed,
    history_name,
    final_history_name,
    scatter_name,
    predictions_thr_name,
    DATA_PATH,
    PATH_TO_NAME,
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
)
from sklearn.model_selection import StratifiedKFold
from automate_training import data_and_labels_from_indices
import pandas as pd
import os
path_list = [
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
]
new_dict = dict()
sheet_name_and_data = dict()
for some_path in path_list:
    hex_file = scatter_name(some_path).replace(".png", "") + "_fixed.csv"
    df = pd.read_csv(hex_file)
    new_dict["AP"] = df["AP"]
    new_dict["Predicted self-assembly probability " + PATH_TO_NAME[some_path]] = df["Predicted self-assembly probability"]
    new_dict["Regression " + PATH_TO_NAME[some_path]] = df["Regression"]
    new_df = pd.DataFrame()
    new_df["AP"] = df["AP"]
    new_df["Predicted self-assembly probability " + PATH_TO_NAME[some_path]] = df["Predicted self-assembly probability"]
    new_df["Regression " + PATH_TO_NAME[some_path]] = df["Regression"]
    sheet_name_and_data[PATH_TO_NAME[some_path]] = new_df
for some_path in path_list:
    mode_determine = "a" 
    if some_path == path_list[0]:
        mode_determine = "w"
    writer = pd.ExcelWriter("my_merged_hex.xlsx", engine = 'openpyxl', mode = mode_determine)
    sheet_name_and_data[PATH_TO_NAME[some_path]].to_excel(writer, sheet_name = "4c - " + PATH_TO_NAME[some_path], index = False)
    writer.close()
df_new_alt_6000 = pd.DataFrame(new_dict)
df_new_alt_6000.to_csv("my_merged_hex_alt.csv", index = False)

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

new_files_dict = dict()
minlen = 3
maxlen = 24
for model_name in os.listdir("review_20/long/preds/" + str(minlen) + "_" + str(maxlen) + "/"):
    if not os.path.isdir("review_20/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/"):
        continue
    new_files_dict_model = dict()
    for seed_val in seed_list:
        pred_arr1_filter = []
        pred_arr2_filter = []
        ix_filter = []
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
                if seqs[seq_ix] not in new:
                    continue
                pred_arr1_filter.append(pred_arr1[seq_ix])
                pred_arr2_filter.append(pred_arr2[seq_ix])
                seqs_filter.append(seqs[seq_ix])
                labs_filter.append(labs[seq_ix])
                ix_filter.append(test_num)
        thr_vals = [0.5 for v in pred_arr1_filter]
        thr_PR_vals = [PRthr[model_name.replace("_model_data", "").replace("_data", "")] for v in pred_arr1_filter]
        thr_ROC_vals = [ROCthr[model_name.replace("_model_data", "").replace("_data", "")] for v in pred_arr1_filter]
        df2 = pd.read_csv("review_20/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_seed_" + str(seed_val) + "_PR_preds.csv")
        new_files_dict["labels"] = df2["labels"]
        new_files_dict["sequence"] = df2["feature"]
        #new_files_dict["0.5 thr"] = thr_vals
        new_files_dict["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
        new_files_dict["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
        new_files_dict["test " + str(seed_val)] = ix_filter
        #new_files_dict["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + " seed " + str(seed_val)] = df2["preds"]
        new_files_dict["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + " seed " + str(seed_val)] = pred_arr1_filter
        new_files_dict_model["labels"] = df2["labels"]
        new_files_dict_model["sequence"] = df2["feature"]
        #new_files_dict_model["0.5 thr"] = thr_vals
        new_files_dict_model["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
        new_files_dict_model["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
        new_files_dict_model["test " + str(seed_val)] = ix_filter
        #new_files_dict_model["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + " seed " + str(seed_val)] = df2["preds"]
        new_files_dict_model["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + " seed " + str(seed_val)] = pred_arr1_filter
    df_new = pd.DataFrame(new_files_dict_model)
    df_new.to_csv("my_merged_hex_alt_20_" + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + ".csv", index = False)
df_new_20 = pd.DataFrame(new_files_dict)
df_new_20.to_csv("my_merged_hex_alt_20.csv", index = False)

df = pd.read_csv("../data/41557_2022_1055_MOESM3_ESM_Figure3a_5mer_score_shortMD.csv")

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

actual_AP = []
for i in df["AP"]:
    actual_AP.append(i)

test_labels = []
threshold = 1.75
for i in df["AP"]:
    if i < threshold:
        test_labels.append(0)
    else:
        test_labels.append(1)
seqs_new = list(dict_hex.keys())

new_files_dict = dict()
minlen = 5
maxlen = 5
for model_name in os.listdir("review_6000/long/preds/" + str(minlen) + "_" + str(maxlen) + "/"):
    if not os.path.isdir("review_6000/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/"):
        continue
    new_files_dict_model = dict()
    pred_file = open("../final/" + model_name + "/" + model_name.replace("_model_data", "").replace("_data", "") + "_predictions_hex.txt", "r")
    pred_arrs = pred_file.readlines()
    pred_arr1 = eval(pred_arrs[0])
    pred_arr1_filter = []
    seqs_filter = []
    labs_filter = []
    for seq_ix in range(len(seqs_new)):
        if len(seqs_new[seq_ix]) > maxlen or len(seqs_new[seq_ix]) < minlen or seq_ix in ix_to_skip:
            continue
        pred_arr1_filter.append(pred_arr1[seq_ix])
        seqs_filter.append(seqs_new[seq_ix])
        labs_filter.append(test_labels[seq_ix])
    thr_vals = [0.5 for v in pred_arr1_filter]
    thr_PR_vals = [PRthr[model_name.replace("_model_data", "").replace("_data", "")] for v in pred_arr1_filter]
    thr_ROC_vals = [ROCthr[model_name.replace("_model_data", "").replace("_data", "")] for v in pred_arr1_filter]
    df3 = pd.read_csv("review_6000/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_PR_preds.csv")
    new_files_dict["labels"] = df3["labels"]
    new_files_dict["sequence"] = df3["feature"]
    #new_files_dict["0.5 thr"] = thr_vals
    new_files_dict["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
    new_files_dict["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
    #new_files_dict["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = df3["preds"]
    new_files_dict["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = pred_arr1_filter
    new_files_dict_model["labels"] = df3["labels"]
    new_files_dict_model["sequence"] = df3["feature"]
    #new_files_dict_model["0.5 thr"] = thr_vals
    new_files_dict_model["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
    new_files_dict_model["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
    #new_files_dict_model["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = df3["preds"]
    new_files_dict_model["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = pred_arr1_filter
    df_new = pd.DataFrame(new_files_dict_model)
    df_new.to_csv("my_merged_hex_alt_6000_" + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + ".csv", index = False)
df_new_6000 = pd.DataFrame(new_files_dict)
df_new_6000.to_csv("my_merged_hex_alt_6000.csv", index = False)

import numpy as np
dflong = pd.read_csv("../data/collection_of_peptide_data.csv")
dict_hexlong = {}
actual_avgAP = []
actual_minAP = []
actual_maxAP = []
actual_APlong = []
duplicate_list = []
duplicate_AP_list = []
threshold = 1.75
for ix in range(len(dflong["Feature"])):
    if dflong["Feature"][ix] not in dict_hexlong:
        dict_hexlong[dflong["Feature"][ix]] = "1"
        actual_APlong.append(dflong["Label"][ix])
        actual_avgAP.append(dflong["Label"][ix])
        actual_minAP.append(dflong["Label"][ix])
        actual_maxAP.append(dflong["Label"][ix])
    else:
        duplicate_list.append(dflong["Feature"][ix])
        duplicate_AP_list.append(dflong["Label"][ix])
for val in duplicate_list:
    ix = list(dict_hexlong.keys()).index(val)
    ix_dup = duplicate_list.index(list(dict_hexlong.keys())[ix])
    error_status = (actual_APlong[ix] < threshold) != (duplicate_AP_list[ix_dup] < threshold)
    #print(error_status, list(dict_hexlong.keys())[ix], actual_APlong[ix], duplicate_AP_list[ix_dup])
    actual_avgAP[ix] = np.average([actual_APlong[ix], duplicate_AP_list[ix_dup]])
    actual_minAP[ix] = np.min([actual_APlong[ix], duplicate_AP_list[ix_dup]])
    actual_maxAP[ix] = np.max([actual_APlong[ix], duplicate_AP_list[ix_dup]])

labslongvavg = []
labslongvmin = []
labslongvmax = []
labsaplongvavg = []
labsaplongvmin = []
labsaplongvmax = []
for i in actual_avgAP:
    labsaplongvavg.append(i)
    if i < threshold:
        labslongvavg.append(0)
    else:
        labslongvavg.append(1)
for i in actual_minAP:
    labsaplongvmin.append(i)
    if i < threshold:
        labslongvmin.append(0)
    else:
        labslongvmin.append(1)
for i in actual_maxAP:
    labsaplongvmax.append(i)
    if i < threshold:
        labslongvmax.append(0)
    else:
        labslongvmax.append(1)
seqslong = list(dict_hexlong.keys())
new_files_dict = dict()
minlen = 5
maxlen = 10
for model_name in os.listdir("review_62000/long/preds/" + str(minlen) + "_" + str(maxlen) + "/"):
    if not os.path.isdir("review_62000/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/"):
        continue
    new_files_dict_model = dict()
    pred_file = open("../final/" + model_name + "/" + model_name.replace("_model_data", "").replace("_data", "") + "_predictions_longest.txt", "r")
    pred_arrs = pred_file.readlines()
    pred_arr1 = eval(pred_arrs[0])
    seqs_filter = []
    labs_filtermin = []
    labs_filtermax = []
    labs_filteravg = []
    labsap_filtermin = []
    labsap_filtermax = []
    labsap_filteravg = []
    pred_arr1_filter = []
    for seq_ix in range(len(seqslong)):
        if len(seqslong[seq_ix]) > maxlen or len(seqslong[seq_ix]) < minlen:
            continue
        pred_arr1_filter.append(pred_arr1[seq_ix])
        seqs_filter.append(seqslong[seq_ix])
        labs_filtermin.append(labslongvmin[seq_ix])
        labs_filtermax.append(labslongvmax[seq_ix])
        labs_filteravg.append(labslongvavg[seq_ix])
        labsap_filtermin.append(labsaplongvmin[seq_ix])
        labsap_filtermax.append(labsaplongvmax[seq_ix])
        labsap_filteravg.append(labsaplongvavg[seq_ix])
    thr_PR_vals = [PRthr[model_name.replace("_model_data", "").replace("_data", "")] for v in pred_arr1_filter]
    thr_ROC_vals = [ROCthr[model_name.replace("_model_data", "").replace("_data", "")] for v in pred_arr1_filter]
    df4 = pd.read_csv("review_62000/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_PR_preds.csv")
    new_files_dict["labels min"] = labslongvmin
    new_files_dict["labels max"] = labslongvmax
    new_files_dict["labels avg"] = labslongvavg
    new_files_dict["AP min"] = labsaplongvmin
    new_files_dict["AP max"] = labsaplongvmax
    new_files_dict["AP avg"] = labsaplongvavg
    new_files_dict["sequence"] = df4["feature"]
    #new_files_dict["0.5 thr"] = thr_vals
    new_files_dict["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
    new_files_dict["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
    #new_files_dict["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = df4["preds"]
    new_files_dict["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = pred_arr1_filter
    new_files_dict_model["labels min"] = labslongvmin
    new_files_dict_model["labels max"] = labslongvmax
    new_files_dict_model["labels avg"] = labslongvavg
    new_files_dict_model["AP min"] = labsaplongvmin
    new_files_dict_model["AP max"] = labsaplongvmax
    new_files_dict_model["AP avg"] = labsaplongvavg
    new_files_dict_model["sequence"] = df4["feature"]
    #new_files_dict_model["0.5 thr"] = thr_vals
    new_files_dict_model["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
    new_files_dict_model["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
    #new_files_dict_model["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = df4["preds"]
    new_files_dict_model["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = pred_arr1_filter
    df_new = pd.DataFrame(new_files_dict_model)
    df_new.to_csv("my_merged_hex_alt_60000_" + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + ".csv", index = False)
df_new_60000 = pd.DataFrame(new_files_dict)
df_new_60000.to_csv("my_merged_hex_alt_60000.csv", index = False)

new_files_dict = dict()
minlen = 3
maxlen = 24
letters = {
    "AP": "d",
    "SP": "e",
    "AP_SP": "f",
    "TSNE_SP": "g",
    "TSNE_AP_SP": "h",
}
labels_original = []
sequences_original = []
for model_name in os.listdir("review/long/preds/" + str(minlen) + "_" + str(maxlen) + "/"):
    if not os.path.isdir("review/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/"):
        continue
    new_files_dict_model = dict()
    for seed_val in seed_list:
        pred_arr1_filter = []
        pred_arr2_filter = []
        ix_filter = []
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
                ix_filter.append(test_num)
        thr_vals = [0.5 for v in pred_arr1_filter]
        thr_PR_vals = [PRthr[model_name.replace("_model_data", "").replace("_data", "")] for v in pred_arr1_filter]
        thr_ROC_vals = [ROCthr[model_name.replace("_model_data", "").replace("_data", "")] for v in pred_arr1_filter]
        df2 = pd.read_csv("review/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_seed_" + str(seed_val) + "_PR_preds.csv")
        new_files_dict["labels"] = df2["labels"]
        new_files_dict["sequence"] = df2["feature"]
        #new_files_dict["0.5 thr"] = thr_vals
        new_files_dict["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
        new_files_dict["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
        new_files_dict["test " + str(seed_val)] = ix_filter
        #new_files_dict["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + " seed " + str(seed_val)] = df2["preds"]
        new_files_dict["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + " seed " + str(seed_val)] = pred_arr1_filter
        new_files_dict_model["labels"] = df2["labels"]
        new_files_dict_model["sequence"] = df2["feature"]
        labels_original = df2["labels"]
        sequences_original = df2["feature"]
        #new_files_dict_model["0.5 thr"] = thr_vals
        new_files_dict_model["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
        new_files_dict_model["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
        new_files_dict_model["test " + str(seed_val)] = ix_filter
        #new_files_dict_model["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + " seed " + str(seed_val)] = df2["preds"]
        new_files_dict_model["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + " seed " + str(seed_val)] = pred_arr1_filter
    df_new = pd.DataFrame(new_files_dict_model)
    df_new.to_csv("my_merged_hex_alt_original_" + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + ".csv", index = False)
    mode_determine = "a" 
    if model_name.replace("_model_data", "").replace("_data", "") == "AP":
        mode_determine = "w"
    writer = pd.ExcelWriter("Source_Data_Fig3.xlsx", engine = 'openpyxl', mode = mode_determine)
    df_new.to_excel(writer, sheet_name = "3" + letters[model_name.replace("_model_data", "").replace("_data", "")], index = False)
    writer.close()

df_new_original = pd.DataFrame(new_files_dict)
df_new_original.to_csv("my_merged_hex_alt_original.csv", index = False)
 
writer = pd.ExcelWriter("Source_Data_Fig4.xlsx", engine = 'openpyxl', mode = "w")
df_new_6000.to_excel(writer, sheet_name = "4b", index = False)
writer.close()

writer = pd.ExcelWriter("Source_Data_Fig4.xlsx", engine = 'openpyxl', mode = "a")
df_new_alt_6000.to_excel(writer, sheet_name = "4c", index = False)
writer.close()

writer = pd.ExcelWriter("Source_Data_Fig4.xlsx", engine = 'openpyxl', mode = "a")
df_new_20.to_excel(writer, sheet_name = "4d", index = False)
writer.close()

NUM_TESTS = 5

def read_one_final_history(some_path, test_number):
    acc_path, loss_path = final_history_name(some_path, test_number)
    acc_file = open(acc_path, "r")
    acc_lines = acc_file.readlines()
    acc = eval(acc_lines[0])
    acc_file.close()
    loss_file = open(loss_path, "r")
    loss_lines = loss_file.readlines()
    loss = eval(loss_lines[0])
    loss_file.close()
    return acc, loss

def read_one_history(some_path, test_number, params_nr, fold_nr):
    acc_name, val_acc_name, loss_name, val_loss_name = history_name(some_path, test_number, params_nr, fold_nr)
    acc_file = open(acc_name, "r")
    acc_lines = acc_file.readlines()
    acc = eval(acc_lines[0])
    acc_file.close()
    loss_file = open(loss_name, "r")
    loss_lines = loss_file.readlines()
    loss = eval(loss_lines[0])
    loss_file.close()
    val_acc_file = open(val_acc_name, "r")
    val_acc_lines = val_acc_file.readlines()
    val_acc = eval(val_acc_lines[0])
    val_acc_file.close()
    val_loss_file = open(val_loss_name, "r")
    val_loss_lines = val_loss_file.readlines()
    val_loss = eval(val_loss_lines[0])
    val_loss_file.close()
    return acc, loss, val_acc, val_loss

def read_all_final_history(some_path, new_dict):
    for test_number in range(1, NUM_TESTS + 1):
        acc, loss = read_one_final_history(some_path, test_number)
        new_dict["Acc " + some_path.replace("/", "").replace(".", "").replace("_model_data", "").replace("_data", "").replace("_", " ") + " Test " + str(test_number)] = acc
        new_dict["Loss " + some_path.replace("/", "").replace(".", "").replace("_model_data", "").replace("_data", "").replace("_", " ") + " Test " + str(test_number)] = loss
    return new_dict

hyperparameter_numcells = [32, 48, 64]
hyperparameter_kernel_size = [4, 6, 8]
long_list = []
short_list = []
for nc in hyperparameter_numcells:
    short_list.append([2 * nc])
    for ks in hyperparameter_kernel_size:
        long_list.append([nc, ks])

def read_all_history(some_path, new_dict):
    num_par = 3
    if "SP" in some_path:
        num_par *= 3
    for params_num in range(1, num_par + 1):
        for test_number in range(1, NUM_TESTS + 1):
            for val_number in range(1, NUM_TESTS + 1):
                acc, loss, val_acc, val_loss = read_one_history(some_path, test_number, params_num, val_number)
                long_title = some_path.replace("/", "").replace(".", "").replace("_model_data", "").replace("_data", "").replace("_", " ") 
                long_title += " Test " + str(test_number)
                long_title += " Val " + str(val_number)
                long_title += " Params " + str(params_num)
                if "AP" in some_path and "SP" not in some_path:
                    long_title += " (dense " + str(short_list[params_num - 1][0])
                if "AP" in some_path and "SP" in some_path:
                    long_title += " (num cells " + str(long_list[params_num - 1][0])
                    long_title += ", kernel size " + str(long_list[params_num - 1][1])
                    long_title += ", dense " + str(long_list[params_num - 1][0] * 2)
                if "AP" not in some_path and "SP" in some_path:
                    long_title += " (num cells " + str(long_list[params_num - 1][0])
                    long_title += ", kernel size " + str(long_list[params_num - 1][1])
                new_dict["Acc " + long_title + ")"] = acc
                new_dict["Loss " + long_title + ")"] = loss
                new_dict["Val Acc " + long_title + ")"] = val_acc
                new_dict["Val Loss " + long_title + ")"] = val_loss
    return new_dict

new_dict_final_acc = dict()
new_dict_final_vacc = dict()
for some_path in path_list:
    new_dict_final_acc = read_all_final_history(some_path, new_dict_final_acc)
    new_dict_final_vacc = read_all_history(some_path, new_dict_final_vacc)

df_anew = pd.DataFrame(new_dict_final_acc)
df_anew.to_csv("my_merged_acc_final.csv", index = False)

df_vanew = pd.DataFrame(new_dict_final_vacc)
df_vanew.to_csv("my_merged_vacc.csv", index = False)

nd1 = {"Labels": labels_original, "Sequences": sequences_original}
df1n = pd.DataFrame(nd1)

writer = pd.ExcelWriter("Source_Data_Fig2.xlsx", engine = 'openpyxl', mode = "w")
df1n.to_excel(writer, sheet_name = "2a", index = False)
writer.close()

writer = pd.ExcelWriter("Source_Data_Fig2.xlsx", engine = 'openpyxl', mode = "a")
df_vanew.to_excel(writer, sheet_name = "2e", index = False)
writer.close()

writer = pd.ExcelWriter("Source_Data_Fig2.xlsx", engine = 'openpyxl', mode = "a")
df_anew.to_excel(writer, sheet_name = "2f", index = False)
writer.close()

N_FOLDS_FIRST = 5
N_FOLDS_SECOND = 5
offset = 1
properties = np.ones(95)
properties[0] = 0
masking_value = 2
allseqs = dict()
allpreds = dict()
alllabels = dict()

def load_data_a(SA_data):
    sequences = []
    labels = []
    for peptide in SA_data:
        if len(peptide) > 24 or SA_data[peptide] == "-1":
            continue
        sequences.append(peptide)
        labels.append(SA_data[peptide])

    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    SA = []
    NSA = []
    for index in range(len(sequences)):
        if labels[index] == "1":
            SA.append(sequences[index])
        elif labels[index] == "0":
            NSA.append(sequences[index])

    return SA, NSA

def merge_data_a(SA, NSA):
    # Merge the bins and add labels
    merged_data = []
    for i in SA:
        merged_data.append(i)
    for i in NSA:
        merged_data.append(i)

    merged_labels = np.ones(len(SA) + len(NSA))
    merged_labels[len(SA) :] *= 0
    return merged_data, merged_labels

for some_path in path_list:
    allseqs[some_path] = dict()
    allpreds[some_path] = dict()
    alllabels[some_path] = dict()
    for some_seed in seed_list:
        allseqs[some_path][some_seed] = dict()
        allpreds[some_path][some_seed] = dict()
        alllabels[some_path][some_seed] = dict()
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

        SA, NSA = load_data_a(SA_data)

        # Merge SA nad NSA data the train and validation subsets.
        all_data, all_labels = merge_data_a(SA, NSA)

        # Define N-fold cross validation test harness for splitting the test data from the train and validation data
        kfold_first = StratifiedKFold(
            n_splits=N_FOLDS_FIRST, shuffle=True, random_state=some_seed
        )
        # Define N-fold cross validation test harness for splitting the validation from the train data
        kfold_second = StratifiedKFold(
            n_splits=N_FOLDS_SECOND, shuffle=True, random_state=some_seed
        )

        test_number = 0

        for train_and_validation_data_indices, test_data_indices in kfold_first.split(
            all_data, all_labels
        ):
            test_number += 1

            allseqs[some_path][some_seed][test_number] = dict()
            allpreds[some_path][some_seed][test_number] = dict()
            alllabels[some_path][some_seed][test_number] = dict()

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

                allseqs[some_path][some_seed][test_number][fold_nr] = []
                allpreds[some_path][some_seed][test_number][fold_nr] = []
                alllabels[some_path][some_seed][test_number][fold_nr] = []

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
                    allseqs[some_path][some_seed][test_number][fold_nr].append(validation_data[i])
                    alllabels[some_path][some_seed][test_number][fold_nr].append(validation_labels[i])
                    allpreds[some_path][some_seed][test_number][fold_nr].append(predictions_file_lines_all[i])
new_df_valn = {"seed": [], "test": [], "val": [], "seq": [], "lab": []}
for some_path in path_list:
    long_title2 = some_path.replace("/", "").replace(".", "").replace("_model_data", "").replace("_data", "").replace("_", " ")
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
    if "AP" in some_path and "SP" not in some_path:
        long_title2 += " (dense " + str(short_list[params_nr - 1][0])
    if "AP" in some_path and "SP" in some_path:
        long_title2 += " (num cells " + str(long_list[params_nr - 1][0])
        long_title2 += ", kernel size " + str(long_list[params_nr - 1][1])
        long_title2 += ", dense " + str(long_list[params_nr - 1][0] * 2)
    if "AP" not in some_path and "SP" in some_path:
        long_title2 += " (num cells " + str(long_list[params_nr - 1][0])
        long_title2 += ", kernel size " + str(long_list[params_nr - 1][1])
    new_df_valn[long_title2 + ")"] = []
for some_seed in seed_list:
    for test_number in range(1, NUM_TESTS + 1):
        for fold_nr in range(1, NUM_TESTS + 1):
            dict_seqs = dict()
            for some_path in path_list:
                long_title = some_path.replace("/", "").replace(".", "").replace("_model_data", "").replace("_data", "").replace("_", " ")
                for ix in range(len(allpreds[some_path][some_seed][test_number][fold_nr])):
                    sequse = allseqs[some_path][some_seed][test_number][fold_nr][ix]
                    labuse = alllabels[some_path][some_seed][test_number][fold_nr][ix]
                    if sequse not in dict_seqs:
                        dict_seqs[sequse] = {"labs": labuse}
                    dict_seqs[sequse][long_title] = allpreds[some_path][some_seed][test_number][fold_nr][ix]
            for sequse in dict_seqs:    
                new_df_valn["seed"].append(some_seed)
                new_df_valn["test"].append(test_number)
                new_df_valn["val"].append(fold_nr)
                new_df_valn["seq"].append(sequse)
                new_df_valn["lab"].append(dict_seqs[sequse]["labs"])
                for some_path in path_list:
                    long_title2 = some_path.replace("/", "").replace(".", "").replace("_model_data", "").replace("_data", "").replace("_", " ")
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
                    if "AP" in some_path and "SP" not in some_path:
                        long_title2 += " (dense " + str(short_list[params_nr - 1][0])
                    if "AP" in some_path and "SP" in some_path:
                        long_title2 += " (num cells " + str(long_list[params_nr - 1][0])
                        long_title2 += ", kernel size " + str(long_list[params_nr - 1][1])
                        long_title2 += ", dense " + str(long_list[params_nr - 1][0] * 2)
                    if "AP" not in some_path and "SP" in some_path:
                        long_title2 += " (num cells " + str(long_list[params_nr - 1][0])
                        long_title2 += ", kernel size " + str(long_list[params_nr - 1][1])
                    long_title = some_path.replace("/", "").replace(".", "").replace("_model_data", "").replace("_data", "").replace("_", " ")
                    new_df_valn[long_title2 + ")"].append(dict_seqs[sequse][long_title])
df_new_df_valn = pd.DataFrame(new_df_valn)
df_new_df_valn.to_csv("my_merged_valid_test_res.csv", index = False)