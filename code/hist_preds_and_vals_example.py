import pandas as pd
import matplotlib.pyplot as plt
from example import survey

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
model_order = {"AP": "AP", "SP": "SP", "AP_SP": "AP-SP", "TSNE_SP": "t-SNE SP", "TSNE_AP_SP": "t-SNE AP-SP"}

def merge_format_long(dirname, type_pred, mini, maxi):
    results = dict()
    for model in model_order:
        model_long = model
        if model != "AP_SP":
            model_long = model_long + "_model"
        model_long = model_long + "_data"
        tp = 0
        fn = 0
        tn = 0
        fp = 0
        for seed in seed_list:
            pd_file = pd.read_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(model_long) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model_long) + "_seed_" + str(seed) + "_" + type_pred + "_preds.csv")
            preds = [int(x) for x in pd_file["preds"]]
            labs = [int(x) for x in pd_file["labels"]]
            for ix in range(len(preds)):
                if labs[ix] == 1:
                    if preds[ix] == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if preds[ix] == 0:
                        tn += 1
                    else:
                        fp += 1
        print(tn, fp, fn, tp)
        results[model_order[model]] = [tn, fp, fn, tp]
    survey(results)
    plt.savefig(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_models_new" + type_pred + ".png", bbox_inches = "tight")
    plt.close()

def merge_format_long_seed(dirname, type_pred, mini, maxi):
    results = dict()
    for model in model_order:
        model_long = model
        if model != "AP_SP":
            model_long = model_long + "_model"
        model_long = model_long + "_data"
        for seed in seed_list:
            tp = 0
            fn = 0
            tn = 0
            fp = 0
            pd_file = pd.read_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(model_long) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model_long) + "_seed_" + str(seed) + "_" + type_pred + "_preds.csv")
            preds = [int(x) for x in pd_file["preds"]]
            labs = [int(x) for x in pd_file["labels"]]
            for ix in range(len(preds)):
                if labs[ix] == 1:
                    if preds[ix] == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if preds[ix] == 0:
                        tn += 1
                    else:
                        fp += 1
            print(tn, fp, fn, tp)
            results[model_order[model] + "\n(seed " + str(seed) + ")"] = [tn, fp, fn, tp]
    survey(results)
    plt.savefig(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_models_new_seeds" + type_pred + ".png", bbox_inches = "tight")
    plt.close()

def merge_format(dirname, type_pred, mini, maxi):
    results = dict()
    for model in model_order:
        model_long = model
        if model != "AP_SP":
            model_long = model_long + "_model"
        model_long = model_long + "_data"
        tp = 0
        fn = 0
        tn = 0
        fp = 0
        pd_file = pd.read_csv(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(model_long) + "/" + str(mini) + "_" + str(maxi) + "_" + str(model_long) + "_" + type_pred + "_preds.csv")
        preds = [int(x) for x in pd_file["preds"]]
        labs = [int(x) for x in pd_file["labels"]]
        for ix in range(len(preds)):
            if labs[ix] == 1:
                if preds[ix] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if preds[ix] == 0:
                    tn += 1
                else:
                    fp += 1
        print(tn, fp, fn, tp)
        results[model_order[model]] = [tn, fp, fn, tp]
    survey(results)
    plt.savefig(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_models_new" + type_pred + ".png", bbox_inches = "tight")
    plt.close()

for type_pred in ["PR", "ROC", "50"]:
    merge_format_long("review", type_pred, 3, 24)
    merge_format_long_seed("review", type_pred, 3, 24)
    merge_format_long("review_20", type_pred, 5, 5)
    merge_format_long_seed("review_20", type_pred, 5, 5)
    merge_format("review_6000", type_pred, 5, 5)
    merge_format("review_62000", type_pred, 5, 10)