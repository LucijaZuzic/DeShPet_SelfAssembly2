from utils import (
    history_name,
    final_history_name,
    scatter_name,
    PATH_TO_NAME,
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
)
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
        thr_vals = [0.5 for v in pred_arr1_filter]
        thr_PR_vals = [PRthr[model_name.replace("_model_data", "").replace("_data", "")] for v in pred_arr1_filter]
        thr_ROC_vals = [ROCthr[model_name.replace("_model_data", "").replace("_data", "")] for v in pred_arr1_filter]
        df2 = pd.read_csv("review_20/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_seed_" + str(seed_val) + "_PR_preds.csv")
        new_files_dict["labels"] = df2["labels"]
        new_files_dict["sequence"] = df2["feature"]
        #new_files_dict["0.5 thr"] = thr_vals
        new_files_dict["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
        new_files_dict["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
        #new_files_dict["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + " seed " + str(seed_val)] = df2["preds"]
        new_files_dict["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + " seed " + str(seed_val)] = pred_arr1_filter
        new_files_dict_model["labels"] = df2["labels"]
        new_files_dict_model["sequence"] = df2["feature"]
        #new_files_dict_model["0.5 thr"] = thr_vals
        new_files_dict_model["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
        new_files_dict_model["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
        #new_files_dict_model["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = df2["preds"]
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
        thr_vals = [0.5 for v in pred_arr1_filter]
        thr_PR_vals = [PRthr[model_name.replace("_model_data", "").replace("_data", "")] for v in pred_arr1_filter]
        thr_ROC_vals = [ROCthr[model_name.replace("_model_data", "").replace("_data", "")] for v in pred_arr1_filter]
        df2 = pd.read_csv("review/long/preds/" + str(minlen) + "_" + str(maxlen) + "/" + model_name + "/" + str(minlen) + "_" + str(maxlen) + "_" + model_name + "_seed_" + str(seed_val) + "_PR_preds.csv")
        new_files_dict["labels"] = df2["labels"]
        new_files_dict["sequence"] = df2["feature"]
        #new_files_dict["0.5 thr"] = thr_vals
        new_files_dict["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
        new_files_dict["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
        #new_files_dict["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + " seed " + str(seed_val)] = df2["preds"]
        new_files_dict["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ") + " seed " + str(seed_val)] = pred_arr1_filter
        new_files_dict_model["labels"] = df2["labels"]
        new_files_dict_model["sequence"] = df2["feature"]
        labels_original = df2["labels"]
        sequences_original = df2["feature"]
        #new_files_dict_model["0.5 thr"] = thr_vals
        new_files_dict_model["PR thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_PR_vals
        new_files_dict_model["ROC thr " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = thr_ROC_vals
        #new_files_dict_model["preds " + model_name.replace("_model_data", "").replace("_data", "").replace("_", " ")] = df2["preds"]
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