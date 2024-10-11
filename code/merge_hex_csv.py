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

def data_and_labels_from_indices(all_data, all_labels, indices):
    data = []
    labels = []

    for i in indices:
        data.append(all_data[i])
        labels.append(all_labels[i])

    return data, labels

import pandas as pd
import os
path_list = [
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
]

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

ap_to_seq = dict()
seq_to_ap = dict()
for seq_with_ap_ix in range(len(df["pep"])):
    ll = 0
    if df["AP"][seq_with_ap_ix] < threshold:
        ll = 0
    else:
        ll = 1

    if df["AP"][seq_with_ap_ix] not in ap_to_seq:
        ap_to_seq[df["AP"][seq_with_ap_ix]] = (df["pep"][seq_with_ap_ix], ll) 
    else:
        print("duplicate", df["AP"][seq_with_ap_ix])

    if df["pep"][seq_with_ap_ix] not in seq_to_ap:
        seq_to_ap[df["pep"][seq_with_ap_ix]] = (df["AP"][seq_with_ap_ix], ll) 
    else:
        print("duplicate second", df["pep"][seq_with_ap_ix])

new_dict = dict()
sheet_name_and_data = dict()
for some_path in path_list:
    hex_file = scatter_name(some_path).replace(".png", "") + "_fixed.csv"
    df = pd.read_csv(hex_file)
    labs = []
    seqs_u = []
    for ap in df["AP"]:
        labs.append(ap_to_seq[ap][1])
        seqs_u.append(ap_to_seq[ap][0])
    short_name = some_path.replace("_model_data", "").replace("_data", "").replace(".", "").replace("/", "")
    thr_vals = [0.5 for v in df["Predicted self-assembly probability"]]
    thr_PR_vals = [PRthr[short_name] for v in df["Predicted self-assembly probability"]]
    thr_ROC_vals = [ROCthr[short_name] for v in df["Predicted self-assembly probability"]]
    prpred = []
    rocpred = []
    halfpred = []
    for p in df["Predicted self-assembly probability"]:
        if p > PRthr[short_name]:
            prpred.append(1)
        else:
            prpred.append(0)
        if p > ROCthr[short_name]:
            rocpred.append(1)
        else:
            rocpred.append(0)
        if p > 0.5:
            halfpred.append(1)
        else:
            halfpred.append(0)
    new_dict["Sequence"] = seqs_u
    new_dict["Label"] = labs
    new_dict["Actual AP"] = df["AP"]
    new_dict["Predicted self-assembly probability " + PATH_TO_NAME[some_path]] = df["Predicted self-assembly probability"]
    new_dict["Thr PR " + PATH_TO_NAME[some_path]] = thr_PR_vals
    new_dict["Thr ROC " + PATH_TO_NAME[some_path]] = thr_ROC_vals
    new_dict["Label PR " + PATH_TO_NAME[some_path]] = prpred
    new_dict["Label ROC " + PATH_TO_NAME[some_path]] = rocpred
    new_dict["Label 0.5 " + PATH_TO_NAME[some_path]] = halfpred
    new_dict["Regression " + PATH_TO_NAME[some_path]] = df["Regression"]
    new_df = pd.DataFrame()
    new_df["Sequence"] = seqs_u
    new_df["Label"] = labs
    new_df["Actual AP"] = df["AP"]
    new_df["Predicted self-assembly probability " + PATH_TO_NAME[some_path]] = df["Predicted self-assembly probability"]
    new_df["Thr PR " + PATH_TO_NAME[some_path]] = thr_PR_vals
    new_df["Thr ROC " + PATH_TO_NAME[some_path]] = thr_ROC_vals
    new_df["Label PR " + PATH_TO_NAME[some_path]] = prpred
    new_df["Label ROC " + PATH_TO_NAME[some_path]] = rocpred
    new_df["Label 0.5 " + PATH_TO_NAME[some_path]] = halfpred
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

minlen = 3
maxlen = 24
dict_model = dict()
for model_name in os.listdir("review_20/long/preds/" + str(minlen) + "_" + str(maxlen) + "/"):
    if not os.path.isdir("review_20/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/"):
        continue
    long_title = model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")
    dict_model[long_title] = dict()
    for seed_val in seed_list:
        dict_model[long_title][seed_val] = dict()
        for test_num in range(1, 6):
            dict_model[long_title][seed_val][test_num] = dict()
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
                if seqs[seq_ix] not in dict_model[long_title][seed_val][test_num]:
                    dict_model[long_title][seed_val][test_num][seqs[seq_ix]] = {"label": labs[seq_ix]}
                dict_model[long_title][seed_val][test_num][seqs[seq_ix]]["Pred " + long_title] = pred_arr1[seq_ix]
                dict_model[long_title][seed_val][test_num][seqs[seq_ix]]["label " + long_title] = pred_arr2[seq_ix]

new_files_dict = {"Seed": [], "Test": [], "Sequence": [], "Label": []}
for long_title in dict_model:
    new_files_dict["PR thr " + long_title] = []
    new_files_dict["ROC thr " + long_title] = []
    new_files_dict["Pred " + long_title] = []
    new_files_dict["PR pred " + long_title] = []
    new_files_dict["ROC pred " + long_title] = []
    new_files_dict["0.5 pred " + long_title] = []
for seed_val in seed_list:
    for test_num in range(1, 6):
        seqs_sth = []
        for long_title in dict_model:
            seqs_sth = list(dict_model[long_title][seed_val][test_num].keys())
            break
        for sequse in seqs_sth:
            new_files_dict["Seed"].append(seed_val)
            new_files_dict["Test"].append(test_num)
            new_files_dict["Sequence"].append(sequse)
            new_files_dict["Label"].append(dict_model[long_title][seed_val][test_num][sequse]["label"])
            thr_title = long_title.replace(" ", "_")
            for long_title in dict_model:
                new_files_dict["PR thr " + long_title].append(PRthr[thr_title])
                new_files_dict["ROC thr " + long_title].append(ROCthr[thr_title])
                new_files_dict["Pred " + long_title].append(dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title])
                if dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title] > PRthr[thr_title]:
                     new_files_dict["PR pred " + long_title].append(1)
                else:
                     new_files_dict["PR pred " + long_title].append(0)
                if dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title] > ROCthr[thr_title]:
                     new_files_dict["ROC pred " + long_title].append(1)
                else:
                     new_files_dict["ROC pred " + long_title].append(0)
                if dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title] > 0.5:
                     new_files_dict["0.5 pred " + long_title].append(1)
                else:
                     new_files_dict["0.5 pred " + long_title].append(0)

df_new_20 = pd.DataFrame(new_files_dict)
df_new_20.to_csv("my_merged_hex_alt_20.csv", index = False)

for long_title in dict_model:
    new_files_dict_model = {"Seed": [], "Test": [], "Sequence": [], "Label": []}
    new_files_dict_model["PR thr " + long_title] = []
    new_files_dict_model["ROC thr " + long_title] = []
    new_files_dict_model["Pred " + long_title] = []
    new_files_dict_model["PR pred " + long_title] = []
    new_files_dict_model["ROC pred " + long_title] = []
    new_files_dict_model["0.5 pred " + long_title] = []
    for seed_val in seed_list:
        for test_num in range(1, 6):
            seqs_sth = list(dict_model[long_title][seed_val][test_num].keys())
            for sequse in seqs_sth:
                new_files_dict_model["Seed"].append(seed_val)
                new_files_dict_model["Test"].append(test_num)
                new_files_dict_model["Sequence"].append(sequse)
                new_files_dict_model["Label"].append(dict_model[long_title][seed_val][test_num][sequse]["label"])
                thr_title = long_title.replace(" ", "_")
                new_files_dict_model["PR thr " + long_title].append(PRthr[thr_title])
                new_files_dict_model["ROC thr " + long_title].append(ROCthr[thr_title])
                new_files_dict_model["Pred " + long_title].append(dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title])
                if dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title] > PRthr[thr_title]:
                     new_files_dict_model["PR pred " + long_title].append(1)
                else:
                     new_files_dict_model["PR pred " + long_title].append(0)
                if dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title] > ROCthr[thr_title]:
                     new_files_dict_model["ROC pred " + long_title].append(1)
                else:
                     new_files_dict_model["ROC pred " + long_title].append(0)
                if dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title] > 0.5:
                     new_files_dict_model["0.5 pred " + long_title].append(1)
                else:
                     new_files_dict_model["0.5 pred " + long_title].append(0)
    df_new = pd.DataFrame(new_files_dict_model)
    df_new.to_csv("my_merged_hex_alt_20_" + long_title + ".csv", index = False)

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
    prpred = []
    rocpred = []
    halfpred = []
    for p in pred_arr1_filter:
        if p > PRthr[model_name.replace("_model_data", "").replace("_data", "")]:
            prpred.append(1)
        else:
            prpred.append(0)
        if p > ROCthr[model_name.replace("_model_data", "").replace("_data", "")]:
            rocpred.append(1)
        else:
            rocpred.append(0)
        if p > 0.5:
            halfpred.append(1)
        else:
            halfpred.append(0)
    aps = []
    for pep in seqs_filter:
        aps.append(seq_to_ap[pep][0])
    new_files_dict["Sequence"] = seqs_filter
    new_files_dict["Label"] = labs_filter
    new_files_dict["Actual AP"] = aps
    #new_files_dict["0.5 thr"] = thr_vals
    new_files_dict["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
    new_files_dict["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
    new_files_dict["Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = pred_arr1_filter
    new_files_dict["PR Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = prpred
    new_files_dict["ROC Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = rocpred
    new_files_dict["0.5 Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = halfpred
    new_files_dict_model["Sequence"] = seqs_filter
    new_files_dict_model["Label"] = labs_filter
    new_files_dict_model["Actual AP"] = aps
    #new_files_dict_model["0.5 thr"] = thr_vals
    new_files_dict_model["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
    new_files_dict_model["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
    new_files_dict_model["Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = pred_arr1_filter
    new_files_dict_model["PR Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = prpred
    new_files_dict_model["ROC Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = rocpred
    new_files_dict_model["0.5 Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = halfpred
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
    prpred = []
    rocpred = []
    halfpred = []
    for p in pred_arr1_filter:
        if p > PRthr[model_name.replace("_model_data", "").replace("_data", "")]:
            prpred.append(1)
        else:
            prpred.append(0)
        if p > ROCthr[model_name.replace("_model_data", "").replace("_data", "")]:
            rocpred.append(1)
        else:
            rocpred.append(0)
        if p > 0.5:
            halfpred.append(1)
        else:
            halfpred.append(0)
    new_files_dict["Sequence"] = seqslong
    new_files_dict["Label min"] = labslongvmin
    new_files_dict["Label max"] = labslongvmax
    new_files_dict["Label avg"] = labslongvavg
    new_files_dict["AP min"] = labsaplongvmin
    new_files_dict["AP max"] = labsaplongvmax
    new_files_dict["AP avg"] = labsaplongvavg
    #new_files_dict["0.5 thr"] = thr_vals
    new_files_dict["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
    new_files_dict["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
    #new_files_dict["Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = df4["preds"]
    new_files_dict["Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = pred_arr1_filter
    new_files_dict["PR Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = prpred
    new_files_dict["ROC Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = rocpred
    new_files_dict["0.5 Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = halfpred
    new_files_dict_model["Sequence"] = seqslong
    new_files_dict_model["Label min"] = labslongvmin
    new_files_dict_model["Label max"] = labslongvmax
    new_files_dict_model["Label avg"] = labslongvavg
    new_files_dict_model["AP min"] = labsaplongvmin
    new_files_dict_model["AP max"] = labsaplongvmax
    new_files_dict_model["AP avg"] = labsaplongvavg
    #new_files_dict_model["0.5 thr"] = thr_vals
    new_files_dict_model["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
    new_files_dict_model["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
    #new_files_dict_model["Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = df4["preds"]
    new_files_dict_model["Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = pred_arr1_filter
    new_files_dict_model["PR Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = prpred
    new_files_dict_model["ROC Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = rocpred
    new_files_dict_model["0.5 Pred " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = halfpred
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
supporting_numbers = {
    "AP": "3",
    "SP": "4",
    "AP_SP": "5",
    "TSNE_SP": "6",
    "TSNE_AP_SP": "7",
}
dict_model = dict()
for model_name in os.listdir("review/long/preds/" + str(minlen) + "_" + str(maxlen) + "/"):
    if not os.path.isdir("review/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/"):
        continue
    long_title = model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")
    dict_model[long_title] = dict()
    for seed_val in seed_list:
        dict_model[long_title][seed_val] = dict()
        for test_num in range(1, 6):
            dict_model[long_title][seed_val][test_num] = dict()
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
                if seqs[seq_ix] not in dict_model[long_title][seed_val][test_num]:
                    dict_model[long_title][seed_val][test_num][seqs[seq_ix]] = {"label": labs[seq_ix]}
                dict_model[long_title][seed_val][test_num][seqs[seq_ix]]["Pred " + long_title] = pred_arr1[seq_ix]
                dict_model[long_title][seed_val][test_num][seqs[seq_ix]]["label " + long_title] = pred_arr2[seq_ix]
            

new_files_dict = {"Seed": [], "Test": [], "Sequence": [], "Label": []}
for long_title in dict_model:
    new_files_dict["PR thr " + long_title] = []
    new_files_dict["ROC thr " + long_title] = []
    new_files_dict["Pred " + long_title] = []
    new_files_dict["PR pred " + long_title] = []
    new_files_dict["ROC pred " + long_title] = []
    new_files_dict["0.5 pred " + long_title] = []
for seed_val in seed_list:
    for test_num in range(1, 6):
        seqs_sth = []
        for long_title in dict_model:
            seqs_sth = list(dict_model[long_title][seed_val][test_num].keys())
            break
        for sequse in seqs_sth:
            new_files_dict["Seed"].append(seed_val)
            new_files_dict["Test"].append(test_num)
            new_files_dict["Sequence"].append(sequse)
            new_files_dict["Label"].append(dict_model[long_title][seed_val][test_num][sequse]["label"])
            thr_title = long_title.replace(" ", "_")
            for long_title in dict_model:
                new_files_dict["PR thr " + long_title].append(PRthr[thr_title])
                new_files_dict["ROC thr " + long_title].append(ROCthr[thr_title])
                new_files_dict["Pred " + long_title].append(dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title])
                if dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title] > PRthr[thr_title]:
                     new_files_dict["PR pred " + long_title].append(1)
                else:
                     new_files_dict["PR pred " + long_title].append(0)
                if dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title] > ROCthr[thr_title]:
                     new_files_dict["ROC pred " + long_title].append(1)
                else:
                     new_files_dict["ROC pred " + long_title].append(0)
                if dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title] > 0.5:
                     new_files_dict["0.5 pred " + long_title].append(1)
                else:
                     new_files_dict["0.5 pred " + long_title].append(0)

df_new_original = pd.DataFrame(new_files_dict)
df_new_original.to_csv("my_merged_hex_alt_original.csv", index = False)

writer = pd.ExcelWriter("Source_Data_Table1.xlsx", engine = 'openpyxl', mode = "w")
df_new_original.to_excel(writer, sheet_name = "Table1", index = False)
writer.close()

first = True

new_S105 = dict()
new_S112 = dict()
for long_title in dict_model:
    new_files_dict_model = {"Seed": [], "Test": [], "Sequence": [], "Label": []}
    new_files_dict_model["PR thr " + long_title] = []
    new_files_dict_model["ROC thr " + long_title] = []
    new_files_dict_model["Pred " + long_title] = []
    new_files_dict_model["PR pred " + long_title] = []
    new_files_dict_model["ROC pred " + long_title] = []
    new_files_dict_model["0.5 pred " + long_title] = []
    for seed_val in seed_list:
        sequences_original = []
        labels_original = []
        for test_num in range(1, 6):
            seqs_sth = list(dict_model[long_title][seed_val][test_num].keys())
            for sequse in seqs_sth:
                sequences_original.append(sequse)
                labels_original.append(dict_model[long_title][seed_val][test_num][sequse]["label"])
            for sequse in seqs_sth:
                new_files_dict_model["Seed"].append(seed_val)
                new_files_dict_model["Test"].append(test_num)
                new_files_dict_model["Sequence"].append(sequse)
                new_files_dict_model["Label"].append(dict_model[long_title][seed_val][test_num][sequse]["label"])
                thr_title = long_title.replace(" ", "_")
                new_files_dict_model["PR thr " + long_title].append(PRthr[thr_title])
                new_files_dict_model["ROC thr " + long_title].append(ROCthr[thr_title])
                new_files_dict_model["Pred " + long_title].append(dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title])
                if dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title] > PRthr[thr_title]:
                     new_files_dict_model["PR pred " + long_title].append(1)
                else:
                     new_files_dict_model["PR pred " + long_title].append(0)
                if dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title] > ROCthr[thr_title]:
                     new_files_dict_model["ROC pred " + long_title].append(1)
                else:
                     new_files_dict_model["ROC pred " + long_title].append(0)
                if dict_model[long_title][seed_val][test_num][sequse]["Pred " + long_title] > 0.5:
                     new_files_dict_model["0.5 pred " + long_title].append(1)
                else:
                     new_files_dict_model["0.5 pred " + long_title].append(0)
    df_new = pd.DataFrame(new_files_dict_model)
    df_new.to_csv("my_merged_hex_alt_original_" + long_title + ".csv", index = False)
    mode_determine = "a" 
    if first:
        mode_determine = "w"
    first = False
    writer = pd.ExcelWriter("Source_Data_Fig3.xlsx", engine = 'openpyxl', mode = mode_determine)
    df_new.to_excel(writer, sheet_name = "Fig3" + letters[long_title.replace(" ", "_")], index = False)
    writer.close()

    new_files_dict_model_filtered = dict()
    for cols in new_files_dict_model:
        if "thr" not in cols and " pred " not in cols:
            new_files_dict_model_filtered[cols] = new_files_dict_model[cols]
        if "ROC" not in cols and "0.5" not in cols and long_title.replace(" ", "_") == "AP_SP":
            new_S105[cols] = new_files_dict_model[cols]
        if "ROC" not in cols and "0.5" not in cols:
            new_S112[cols] = new_files_dict_model[cols]
    df_new_filtered = pd.DataFrame(new_files_dict_model_filtered)
    writer_filtered = pd.ExcelWriter("Source_Data_FigS1." + supporting_numbers[long_title.replace(" ", "_")] + ".xlsx", engine = 'openpyxl', mode = "w")
    df_new_filtered.to_excel(writer_filtered, sheet_name = "FigS1." + supporting_numbers[long_title.replace(" ", "_")], index = False)
    writer_filtered.close()

df_new_S105 = pd.DataFrame(new_S105)
writer = pd.ExcelWriter("Source_Data_TableS1.5.xlsx", engine = 'openpyxl', mode = "w")
df_new_S105.to_excel(writer, sheet_name = "TableS1.5", index = False)
writer.close()

df_new_S112 = pd.DataFrame(new_S112)
writer = pd.ExcelWriter("Source_Data_FigS1.12.xlsx", engine = 'openpyxl', mode = "w")
df_new_S112.to_excel(writer, sheet_name = "FigS1.12", index = False)
writer.close()

writer = pd.ExcelWriter("Source_Data_FigS2.1.xlsx", engine = 'openpyxl', mode = "w")
df_new_6000.to_excel(writer, sheet_name = "FigS2.1b", index = False)
writer.close()

writer = pd.ExcelWriter("Source_Data_FigS2.1.xlsx", engine = 'openpyxl', mode = "a")
df_new_alt_6000.to_excel(writer, sheet_name = "FigS2.1c", index = False)
writer.close()

writer = pd.ExcelWriter("Source_Data_TableS1.4.xlsx", engine = 'openpyxl', mode = "w")
df_new_20.to_excel(writer, sheet_name = "TableS1.4", index = False)
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

def read_all_history_param_one(some_path, params_num, new_dict):
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

new_dict_final_vacc_AP_SP_params1 = dict()
new_dict_final_vacc_AP_SP_params1 = read_all_history_param_one(AP_SP_DATA_PATH, 1, new_dict_final_vacc_AP_SP_params1)
df_new_dict_final_vacc_AP_SP_params1 = pd.DataFrame(new_dict_final_vacc_AP_SP_params1)
writer = pd.ExcelWriter("Source_Data_FigS1.13.xlsx", engine = 'openpyxl', mode = "w")
df_new_dict_final_vacc_AP_SP_params1.to_excel(writer, sheet_name = "FigS1.13", index = False)
writer.close()

df_anew = pd.DataFrame(new_dict_final_acc)
df_anew.to_csv("my_merged_acc_final.csv", index = False)

df_vanew = pd.DataFrame(new_dict_final_vacc)
df_vanew.to_csv("my_merged_vacc.csv", index = False)

nd1 = {"Sequence": sequences_original, "Label": labels_original}
df1n = pd.DataFrame(nd1)

writer = pd.ExcelWriter("Source_Data_Fig2.xlsx", engine = 'openpyxl', mode = "w")
df1n.to_excel(writer, sheet_name = "Fig2a", index = False)
writer.close()

writer = pd.ExcelWriter("Source_Data_Fig2.xlsx", engine = 'openpyxl', mode = "a")
df_vanew.to_excel(writer, sheet_name = "Fig2e", index = False)
writer.close()

writer = pd.ExcelWriter("Source_Data_Fig2.xlsx", engine = 'openpyxl', mode = "a")
df_anew.to_excel(writer, sheet_name = "Fig2f", index = False)
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

new_df_valn = {"Seed": [], "Test": [], "Val": [], "Sequence": [], "Label": []}
new_df_valn_short = {"Seed": [], "Test": [], "Val": [], "Sequence": [], "Label": []}
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
    new_df_valn_short[long_title2] = []
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
            list_seqs = []
            list_labs = []
            for some_path in path_list:
                list_seqs = allseqs[some_path][some_seed][test_number][fold_nr]
                list_labs = alllabels[some_path][some_seed][test_number][fold_nr]
                break
            for ix_sequse in range(len(list_seqs)):    
                new_df_valn["Seed"].append(some_seed)
                new_df_valn["Test"].append(test_number)
                new_df_valn["Val"].append(fold_nr)
                new_df_valn["Sequence"].append(list_seqs[ix_sequse])
                new_df_valn["Label"].append(list_labs[ix_sequse])
                new_df_valn_short["Seed"].append(some_seed)
                new_df_valn_short["Test"].append(test_number)
                new_df_valn_short["Val"].append(fold_nr)
                new_df_valn_short["Sequence"].append(list_seqs[ix_sequse])
                new_df_valn_short["Label"].append(list_labs[ix_sequse])
                for some_path in path_list:
                    long_title2 = some_path.replace("/", "").replace(".", "").replace("_model_data", "").replace("_data", "").replace("_", " ")
                    new_df_valn_short[long_title2].append(allpreds[some_path][some_seed][test_number][fold_nr][ix_sequse])
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
                    new_df_valn[long_title2 + ")"].append(allpreds[some_path][some_seed][test_number][fold_nr][ix_sequse])
df_new_df_valn = pd.DataFrame(new_df_valn)
df_new_df_valn.to_csv("my_merged_valid_test_res.csv", index = False)

df_new_df_valn_short = pd.DataFrame(new_df_valn_short)
df_new_df_valn_short.to_csv("my_merged_valid_test_res_short.csv", index = False)

writer = pd.ExcelWriter("Source_Data_TableS1.2_no_thr.xlsx", engine = 'openpyxl', mode = "w")
df_new_df_valn_short.to_excel(writer, sheet_name = "TableS1.2", index = False)
writer.close()

new_df_valn_short_with_thr = dict()

for cols in ["Seed", "Test", "Val", "Sequence", "Label"]:
    new_df_valn_short_with_thr[cols] = new_df_valn_short[cols]
for long_title in new_df_valn_short:
    if long_title not in ["Seed", "Test", "Val", "Sequence", "Label"]:
        new_df_valn_short_with_thr["PR thr " + long_title] = []
        new_df_valn_short_with_thr["ROC thr " + long_title] = []
        new_df_valn_short_with_thr["Pred " + long_title] = []
        new_df_valn_short_with_thr["PR pred " + long_title]  = []
        new_df_valn_short_with_thr["ROC pred " + long_title]  = []
        new_df_valn_short_with_thr["0.5 pred " + long_title]  = []

for long_title in new_df_valn_short:
    if long_title not in ["Seed", "Test", "Val", "Sequence", "Label"]:
        thr_title = long_title.replace(" ", "_")
        arr_original = new_df_valn_short[long_title]
        for pred in arr_original:
            new_df_valn_short_with_thr["PR thr " + long_title].append(PRthr[thr_title])
            new_df_valn_short_with_thr["ROC thr " + long_title].append(ROCthr[thr_title])
            new_df_valn_short_with_thr["Pred " + long_title].append(pred)
            if pred > PRthr[thr_title]:
                new_df_valn_short_with_thr["PR pred " + long_title].append(1)
            else:
                new_df_valn_short_with_thr["PR pred " + long_title].append(0)
            if pred > ROCthr[thr_title]:
                new_df_valn_short_with_thr["ROC pred " + long_title].append(1)
            else:
                new_df_valn_short_with_thr["ROC pred " + long_title].append(0)
            if pred > 0.5:
                new_df_valn_short_with_thr["0.5 pred " + long_title].append(1)
            else:
                new_df_valn_short_with_thr["0.5 pred " + long_title].append(0)

df_new_df_valn_short_with_thr = pd.DataFrame(new_df_valn_short_with_thr)
df_new_df_valn_short_with_thr.to_csv("my_merged_valid_test_res_short_thr.csv", index = False)

writer = pd.ExcelWriter("Source_Data_TableS1.2.xlsx", engine = 'openpyxl', mode = "w")
df_new_df_valn_short_with_thr.to_excel(writer, sheet_name = "TableS1.2", index = False)
writer.close()