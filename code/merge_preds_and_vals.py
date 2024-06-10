import pandas as pd
import os

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]

def merge_format_long(dirname, mini, maxi):
    dfdictPR = dict()
    dfdictROC = dict()
    dfdict50 = dict()
    for model in os.listdir(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/"):
        if not os.path.isdir(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + model):
            continue
        for seed in seed_list:
            pd_filePR = pd.read_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(model) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model) + "_seed_" + str(seed) + "_PR_preds.csv")
            dfdictPR["preds_" + str(model) + "_" + str(seed)] = pd_filePR["preds"]
            dfdictPR["labels_" + str(model) + "_" + str(seed)] = pd_filePR["labels"]
            dfdictPR["feature_" + str(model) + "_" + str(seed)] = pd_filePR["feature"]
            pd_fileROC = pd.read_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(model) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model) + "_seed_" + str(seed) + "_ROC_preds.csv")
            dfdictROC["preds_" + str(model) + "_" + str(seed)] = pd_fileROC["preds"]
            dfdictROC["labels_" + str(model) + "_" + str(seed)] = pd_fileROC["labels"]
            dfdictROC["feature_" + str(model) + "_" + str(seed)] = pd_fileROC["feature"]
            pd_file50 = pd.read_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(model) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model) + "_seed_" + str(seed) + "_50_preds.csv")
            dfdict50["preds_" + str(model) + "_" + str(seed)] = pd_file50["preds"]
            dfdict50["labels_" + str(model) + "_" + str(seed)] = pd_file50["labels"]
            dfdict50["feature_" + str(model) + "_" + str(seed)] = pd_file50["feature"]
    df_newPR = pd.DataFrame(dfdictPR)
    df_newPR.to_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_PR_preds.csv")
    df_newROC = pd.DataFrame(dfdictROC)
    df_newROC.to_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_ROC_preds.csv")
    df_new50 = pd.DataFrame(dfdict50)
    df_new50.to_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_50_preds.csv")

def merge_format(dirname, mini, maxi):
    dfdictPR = dict()
    dfdictROC = dict()
    dfdict50 = dict()
    for model in os.listdir(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/"):
        if not os.path.isdir(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + model):
            continue
        pd_filePR = pd.read_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(model) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model) + "_PR_preds.csv")
        dfdictPR["preds_" + str(model)] = pd_filePR["preds"]
        dfdictPR["labels_" + str(model)] = pd_filePR["labels"]
        dfdictPR["feature_" + str(model)] = pd_filePR["feature"]
        pd_fileROC = pd.read_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(model) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model) + "_ROC_preds.csv")
        dfdictROC["preds_" + str(model)] = pd_fileROC["preds"]
        dfdictROC["labels_" + str(model)] = pd_fileROC["labels"]
        dfdictROC["feature_" + str(model)] = pd_fileROC["feature"]
        pd_file50 = pd.read_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(model) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model) + "_50_preds.csv")
        dfdict50["preds_" + str(model)] = pd_file50["preds"]
        dfdict50["labels_" + str(model)] = pd_file50["labels"]
        dfdict50["feature_" + str(model)] = pd_file50["feature"]
    df_newPR = pd.DataFrame(dfdictPR)
    df_newPR.to_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_PR_preds.csv")
    df_newROC = pd.DataFrame(dfdictROC)
    df_newROC.to_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_ROC_preds.csv")
    df_new50 = pd.DataFrame(dfdict50)
    df_new50.to_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_50_preds.csv")

merge_format_long("review", 3, 24)
merge_format_long("review_20", 5, 5)
merge_format("review_6000", 5, 5)
merge_format("review_62000", 5, 10)

merge_format("review_genetic", 6, 10)

merge_format("review_genetic_low", 6, 10)
merge_format("review_genetic_low_0", 6, 6)
merge_format("review_genetic_low_1", 10, 10)

merge_format("review_genetic_strong", 6, 10)
merge_format("review_genetic_strong_0", 6, 6)
merge_format("review_genetic_strong_1", 7, 10)

merge_format("review_genetic_experiments", 9, 10)
merge_format("review_genetic_experimentsA", 9, 10)
merge_format("review_genetic_experimentsB", 9, 10)
merge_format("review_genetic_experimentsC", 9, 10)