import pandas as pd
import os
import matplotlib.pyplot as plt

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
light_color = {"TN": "#FFF2CC", "FP": "#F8CECC", "FN": "#DAE8FC", "TP": "#D5E8D4"}
dark_color = {"TN": "#D6B656", "FP": "#B85450", "FN": "#6C8EBF", "TP": "#82B366"}
model_order = {"AP": "AP", "SP": "SP", "AP_SP": "AP-SP", "TSNE_SP": "t-SNE SP", "TSNE_AP_SP": "t-SNE AP-SP"}

def merge_format_long(dirname, type_pred, mini, maxi):
    ix_model = 0
    tp_list = [0 for model in model_order]
    fn_list = [0 for model in model_order]
    tn_list = [0 for model in model_order]
    fp_list = [0 for model in model_order]
    xtick_labels = []
    fig = plt.figure(figsize = (5, 5))
    ax = fig.add_subplot(111)
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
        tp_list[ix_model] += tp
        fn_list[ix_model] += fn
        tn_list[ix_model] += tn
        fp_list[ix_model] += fp
        ix_model += 1
        xtick_labels.append(model_order[model])
    bar_items = {"TN": tn_list, "FP": fp_list, "FN": fn_list, "TP": tp_list}
    bottom = [0 for x in tn_list]
    for label_sample, sample_count in bar_items.items():
        plt.bar(range(len(sample_count)), sample_count, linewidth = 3, width = 1, label = label_sample, bottom = bottom, edgecolor = dark_color[label_sample], color = light_color[label_sample])
        mid = [bottom[x] + sample_count[x] / 4 for x in range(len(bottom))]
        for num_ix in range(len(sample_count)):
            if sample_count[num_ix] > 0:
                plt.text(num_ix - len(str(sample_count[num_ix])) / 16, mid[num_ix], str(sample_count[num_ix]))
        bottom = [bottom[x] + sample_count[x] for x in range(len(bottom))]
    plt.xticks(range(len(xtick_labels)), xtick_labels)
    plt.yticks([tn_list[-1] + fp_list[-1], tn_list[-1] + fp_list[-1] + fn_list[-1] + tp_list[-1]], [str(tn_list[-1] + fp_list[-1]), str(tn_list[-1] + fp_list[-1] + fn_list[-1] + tp_list[-1])])
    plt.xlim(-0.5, len(xtick_labels) - 0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylabel("Number of peptides")
    plt.xlabel("Model")
    plt.legend(loc="lower left", bbox_to_anchor = (0, -0.2), ncol = 4)
    plt.savefig(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_models" + type_pred + ".png", bbox_inches = "tight")
    plt.close()

def merge_format_long_seed(dirname, type_pred, mini, maxi):
    ix_model = 0
    tick_ix = 0
    xtick_labels_text = []
    xtick_labels_coord = []
    fig = plt.figure(figsize = (25, 5))
    ax = fig.add_subplot(111)
    for model in model_order:
        model_long = model
        if model != "AP_SP":
            model_long = model_long + "_model"
        model_long = model_long + "_data"
        ix_seed = 0
        tp_list = [0 for seed in seed_list]
        fn_list = [0 for seed in seed_list]
        tn_list = [0 for seed in seed_list]
        fp_list = [0 for seed in seed_list]
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
            tp_list[ix_seed] += tp
            fn_list[ix_seed] += fn
            tn_list[ix_seed] += tn
            fp_list[ix_seed] += fp
            if ix_seed == 0:
                plt.text(tick_ix + ix_seed, (tp + fn + tn + fp) * 1.025, model_order[model])
            xtick_labels_coord.append(tick_ix + ix_seed)
            ix_seed += 1
            xtick_labels_text.append(str(ix_seed))
        ix_model += 1
        bar_items = {"TN": tn_list, "FP": fp_list, "FN": fn_list, "TP": tp_list}
        bottom = [0 for x in tn_list]
        for label_sample, sample_count in bar_items.items():
            if tick_ix == 0:
                plt.bar([tick_ix + x for x in range(len(sample_count))], sample_count, linewidth = 3, width = 1, label = label_sample, bottom = bottom, edgecolor = dark_color[label_sample], color = light_color[label_sample])
            else:
                plt.bar([tick_ix + x for x in range(len(sample_count))], sample_count, linewidth = 3, width = 1, bottom = bottom, edgecolor = dark_color[label_sample], color = light_color[label_sample])
            mid = [bottom[x] + sample_count[x] / 4 for x in range(len(bottom))]
            for num_ix in range(len(sample_count)):
                if sample_count[num_ix] > 0:
                    plt.text(tick_ix + num_ix - len(str(sample_count[num_ix])) / 16, mid[num_ix], str(sample_count[num_ix]))
            bottom = [bottom[x] + sample_count[x] for x in range(len(bottom))]
        tick_ix += len(tn_list) + 0.5
    plt.xticks(xtick_labels_coord, xtick_labels_text)
    plt.yticks([tn_list[-1] + fp_list[-1], tn_list[-1] + fp_list[-1] + fn_list[-1] + tp_list[-1]], [str(tn_list[-1] + fp_list[-1]), str(tn_list[-1] + fp_list[-1] + fn_list[-1] + tp_list[-1])])
    plt.xlim(-0.5, xtick_labels_coord[-1] + 0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylabel("Number of peptides")
    plt.xlabel("Number of seed")
    plt.legend(loc="lower left", bbox_to_anchor = (0, -0.2), ncol = 4)
    plt.savefig(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_models_seeds" + type_pred + ".png", bbox_inches = "tight")
    plt.close()

def merge_format(dirname, type_pred, mini, maxi):
    ix_model = 0
    tp_list = [0 for model in model_order]
    fn_list = [0 for model in model_order]
    tn_list = [0 for model in model_order]
    fp_list = [0 for model in model_order]
    xtick_labels = []
    fig = plt.figure(figsize = (5, 5))
    ax = fig.add_subplot(111)
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
        tp_list[ix_model] += tp
        fn_list[ix_model] += fn
        tn_list[ix_model] += tn
        fp_list[ix_model] += fp
        ix_model += 1
        xtick_labels.append(model_order[model])
    bar_items = {"TN": tn_list, "FP": fp_list, "FN": fn_list, "TP": tp_list}
    bottom = [0 for x in tn_list]
    for label_sample, sample_count in bar_items.items():
        plt.bar(range(len(sample_count)), sample_count, linewidth = 3, width = 1, label = label_sample, bottom = bottom, edgecolor = dark_color[label_sample], color = light_color[label_sample])
        mid = [bottom[x] + sample_count[x] / 4 for x in range(len(bottom))]
        for num_ix in range(len(sample_count)):
            if sample_count[num_ix] > 0:
                plt.text(num_ix - len(str(sample_count[num_ix])) / 16, mid[num_ix], str(sample_count[num_ix]))
        bottom = [bottom[x] + sample_count[x] for x in range(len(bottom))]
    plt.xticks(range(len(xtick_labels)), xtick_labels)
    plt.yticks([tn_list[-1] + fp_list[-1], tn_list[-1] + fp_list[-1] + fn_list[-1] + tp_list[-1]], [str(tn_list[-1] + fp_list[-1]), str(tn_list[-1] + fp_list[-1] + fn_list[-1] + tp_list[-1])])
    plt.xlim(-0.5, len(xtick_labels) - 0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.ylabel("Number of peptides")
    plt.xlabel("Model")
    plt.legend(loc="lower left", bbox_to_anchor = (0, -0.2), ncol = 4)
    plt.savefig(dirname + "/long/preds/" + str(mini) + "_" + str(maxi) + "/" + str(mini) + "_" + str(maxi) + "_all_models" + type_pred + ".png", bbox_inches = "tight")
    plt.close()

for type_pred in ["PR", "ROC", "50"]:
    merge_format_long("review", type_pred, 3, 24)
    merge_format_long_seed("review", type_pred, 3, 24)
    merge_format_long("review_20", type_pred, 5, 5)
    merge_format_long_seed("review_20", type_pred, 5, 5)
    merge_format("review_6000", type_pred, 5, 5)
    merge_format("review_62000", type_pred, 5, 10)