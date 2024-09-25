from utils import (
    set_seed,
    predictions_name,
    final_history_name,
    PATH_TO_NAME,
    PATH_TO_EXTENSION,
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
)
from custom_plots import merge_type_test_number, results_name, weird_division
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.metrics import roc_curve, precision_recall_curve

total_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
asp = 6.1
yas = 1.12
xas = yas * asp
cm = 1/2.54  # centimeters in inches

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


def read_all_final_history(some_path, lines_dict, sd_dict):
    all_acc = []
    all_loss = []
    for test_number in range(1, NUM_TESTS + 1):
        acc, loss = read_one_final_history(some_path, test_number)
        for a in acc:
            all_acc.append(float(a))
        for l in loss:
            all_loss.append(float(l))

    lines_dict["Maximum accuracy = "].append(np.max(all_acc) * 100)
    lines_dict["Minimal loss = "].append(np.min(all_loss) * 100)
    lines_dict["Accuracy = "].append(np.mean(all_acc) * 100)
    lines_dict["Loss = "].append(np.mean(all_loss))
    sd_dict["Accuracy = "].append(np.std(all_acc) * 100)
    sd_dict["Loss = "].append(np.std(all_loss))


def read_one_prediction(some_path, test_number):
    file = open(predictions_name(some_path, test_number), "r")
    lines = file.readlines()
    predictions = eval(lines[0])
    labels = eval(lines[1])
    file.close()
    return predictions, labels


def read_all_model_predictions(some_path, min_test_number, max_test_number):
    all_predictions = []
    all_labels = []
    for test_number in range(min_test_number, max_test_number + 1):
        predictions, labels = read_one_prediction(some_path, test_number)
        for prediction in predictions:
            all_predictions.append(prediction)
        for label in labels:
            all_labels.append(label)
    return all_predictions, all_labels

def hist_predicted_merged_numbers(
    model_type, test_number, test_labels, model_predictions, save
):
    plt.rcParams.update({"font.size": 12})
    # Create a histogram of the predicted probabilities only for the peptides that show self-assembly
    model_predictions_true = []
    model_predictions_false = []
    for x in range(len(test_labels)):
        if test_labels[x] == 1.0:
            model_predictions_true.append(float(model_predictions[x]))
        else:
            model_predictions_false.append(float(model_predictions[x]))

    size_of_bin_positive_low = dict()
    size_of_bin_positive_high = dict()
    size_of_bin_positive_both = dict()
    size_of_bin_positive_none = dict()
    for bin_ix in range(len(total_bins) - 1):
        size_of_bin_positive_low[bin_ix] = 0
        size_of_bin_positive_high[bin_ix] = 0
        size_of_bin_positive_both[bin_ix] = 0
        size_of_bin_positive_none[bin_ix] = 0
    for val in model_predictions_true:
        for bin_ix in range(len(total_bins) - 1):
            bin_start = total_bins[bin_ix]
            bin_end = total_bins[bin_ix + 1]
            if val >= bin_start and val < bin_end:
                size_of_bin_positive_low[bin_ix] += 1
            if val > bin_start and val <= bin_end:
                size_of_bin_positive_high[bin_ix] += 1
            if val >= bin_start and val <= bin_end:
                size_of_bin_positive_both[bin_ix] += 1
            if val > bin_start and val < bin_end:
                size_of_bin_positive_none[bin_ix] += 1
    for bin_ix in range(len(total_bins) - 1):
        bin_start = total_bins[bin_ix]
        bin_end = total_bins[bin_ix + 1]
        if size_of_bin_positive_low[bin_ix] != size_of_bin_positive_high[bin_ix]:
            print("lh", bin_ix, bin_start, bin_end, size_of_bin_positive_low[bin_ix], size_of_bin_positive_high[bin_ix])
        if size_of_bin_positive_low[bin_ix] != size_of_bin_positive_both[bin_ix]:
            print("lb", bin_ix, bin_start, bin_end, size_of_bin_positive_low[bin_ix], size_of_bin_positive_both[bin_ix])
        if size_of_bin_positive_low[bin_ix] != size_of_bin_positive_none[bin_ix]:
            print("ln", bin_ix, bin_start, bin_end, size_of_bin_positive_low[bin_ix], size_of_bin_positive_none[bin_ix])
        if size_of_bin_positive_high[bin_ix] != size_of_bin_positive_both[bin_ix]:
            print("hb", bin_ix, bin_start, bin_end, size_of_bin_positive_high[bin_ix], size_of_bin_positive_both[bin_ix])
        if size_of_bin_positive_high[bin_ix] != size_of_bin_positive_none[bin_ix]:
            print("hn", bin_ix, bin_start, bin_end, size_of_bin_positive_high[bin_ix], size_of_bin_positive_none[bin_ix])
        if size_of_bin_positive_both[bin_ix] != size_of_bin_positive_none[bin_ix]:
            print("bn", bin_ix, bin_start, bin_end, size_of_bin_positive_both[bin_ix], size_of_bin_positive_none[bin_ix])
    print("positive", model_type, size_of_bin_positive_none)

    size_of_bin_negative_low = dict()
    size_of_bin_negative_high = dict()
    size_of_bin_negative_both = dict()
    size_of_bin_negative_none = dict()
    for bin_ix in range(len(total_bins) - 1):
        size_of_bin_negative_low[bin_ix] = 0
        size_of_bin_negative_high[bin_ix] = 0
        size_of_bin_negative_both[bin_ix] = 0
        size_of_bin_negative_none[bin_ix] = 0
    for val in model_predictions_false:
        for bin_ix in range(len(total_bins) - 1):
            bin_start = total_bins[bin_ix]
            bin_end = total_bins[bin_ix + 1]
            if val >= bin_start and val < bin_end:
                size_of_bin_negative_low[bin_ix] += 1
            if val > bin_start and val <= bin_end:
                size_of_bin_negative_high[bin_ix] += 1
            if val >= bin_start and val <= bin_end:
                size_of_bin_negative_both[bin_ix] += 1
            if val > bin_start and val < bin_end:
                size_of_bin_negative_none[bin_ix] += 1
    for bin_ix in range(len(total_bins) - 1):
        bin_start = total_bins[bin_ix]
        bin_end = total_bins[bin_ix + 1]
        if size_of_bin_negative_low[bin_ix] != size_of_bin_negative_high[bin_ix]:
            print("lh", bin_ix, bin_start, bin_end, size_of_bin_negative_low[bin_ix], size_of_bin_negative_high[bin_ix])
        if size_of_bin_negative_low[bin_ix] != size_of_bin_negative_both[bin_ix]:
            print("lb", bin_ix, bin_start, bin_end, size_of_bin_negative_low[bin_ix], size_of_bin_negative_both[bin_ix])
        if size_of_bin_negative_low[bin_ix] != size_of_bin_negative_none[bin_ix]:
            print("ln", bin_ix, bin_start, bin_end, size_of_bin_negative_low[bin_ix], size_of_bin_negative_none[bin_ix])
        if size_of_bin_negative_high[bin_ix] != size_of_bin_negative_both[bin_ix]:
            print("hb", bin_ix, bin_start, bin_end, size_of_bin_negative_high[bin_ix], size_of_bin_negative_both[bin_ix])
        if size_of_bin_negative_high[bin_ix] != size_of_bin_negative_none[bin_ix]:
            print("hn", bin_ix, bin_start, bin_end, size_of_bin_negative_high[bin_ix], size_of_bin_negative_none[bin_ix])
        if size_of_bin_negative_both[bin_ix] != size_of_bin_negative_none[bin_ix]:
            print("bn", bin_ix, bin_start, bin_end, size_of_bin_negative_both[bin_ix], size_of_bin_negative_none[bin_ix])
    print("negative", model_type, size_of_bin_negative_none)

    plt.figure()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # Draw the density plot
    gp = sns.displot(
        {"SA": model_predictions_true},
        kde=True,
        bins=total_bins,
        palette={"SA": "#2e85ff"},
        legend=False,
    )
    axp = gp.axes[0, 0]
    lp = axp.lines[0]
    xp, yp = lp.get_data()
    for bin_ix in range(len(total_bins) - 1):
        hix_start = bin_ix * len(xp) // (len(total_bins) - 1)
        hix_end = (bin_ix + 1) * len(xp) // (len(total_bins) - 1) + 1
        husep = max(max(yp[hix_start:hix_end]), size_of_bin_positive_none[bin_ix]) + 10
        ofs = 0
        if size_of_bin_positive_none[bin_ix] < 100:
            ofs = 1 / 60
        if size_of_bin_positive_none[bin_ix] > 0:
            plt.text(total_bins[bin_ix] + ofs, husep + 40, str(size_of_bin_positive_none[bin_ix]), color = "#2e85ff")
    plt.ylim(0, 750)
    plt.xlabel("Self assembly probability")
    plt.ylabel("Number of peptides")
    plt.title(
        merge_type_test_number(model_type, test_number).replace("Test 0 Weak 1", "").replace("Test 0", "")
    )
    plt.savefig(save + "_SA_numbers.png", bbox_inches="tight")
    plt.savefig(save + "_SA_numbers.svg", bbox_inches="tight")
    plt.savefig(save + "_SA_numbers.pdf", bbox_inches="tight")
    plt.close()

    # Create a histogram of the predicted probabilities only for the peptides that don't show self-assembly

    plt.figure()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    gn = sns.displot(
        {"NSA": model_predictions_false},
        kde=True,
        bins=total_bins,
        palette={"NSA": "#ff120a"},
        legend=False,
    )
    axn = gn.axes[0, 0]
    ln = axn.lines[0]
    xn, yn = ln.get_data()
    for bin_ix in range(len(total_bins) - 1):
        hix_start = bin_ix * len(xn) // (len(total_bins) - 1)
        hix_end = (bin_ix + 1) * len(xn) // (len(total_bins) - 1) + 1
        husen = max(max(yn[hix_start:hix_end]), size_of_bin_negative_none[bin_ix]) + 10
        ofs = 0
        if size_of_bin_negative_none[bin_ix] < 100:
            ofs = 1 / 60
        if size_of_bin_negative_none[bin_ix] > 0:
            plt.text(total_bins[bin_ix] + ofs, husen + 40, str(size_of_bin_negative_none[bin_ix]), color = "#ff120a")
    plt.ylim(0, 750)
    plt.xlabel("Self assembly probability")
    plt.ylabel("Number of peptides")
    plt.title(
        merge_type_test_number(model_type, test_number).replace("Test 0 Weak 1", "").replace("Test 0", "")
    )
    plt.savefig(save + "_NSA_numbers.png", bbox_inches="tight")
    plt.savefig(save + "_NSA_numbers.svg", bbox_inches="tight")
    plt.savefig(save + "_NSA_numbers.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    sns.displot(
        {"SA": model_predictions_true, "NSA": model_predictions_false},
        kde=True,
        bins=total_bins,
        palette={"SA": "#2e85ff", "NSA": "#ff120a"},
    )
    for bin_ix in range(len(total_bins) - 1):
        hix_start = bin_ix * len(xn) // (len(total_bins) - 1)
        hix_end = (bin_ix + 1) * len(xn) // (len(total_bins) - 1) + 1
        husen = max(max(yn[hix_start:hix_end]), size_of_bin_negative_none[bin_ix]) + 10
        husep = max(max(yp[hix_start:hix_end]), size_of_bin_positive_none[bin_ix]) + 10
        huse = max(husen, husep)
        ofs = 0
        if size_of_bin_positive_none[bin_ix] < 100 and size_of_bin_negative_none[bin_ix] < 100:
            ofs = 1 / 60
        if size_of_bin_positive_none[bin_ix] > 0 and not size_of_bin_negative_none[bin_ix] > 0:
            plt.text(total_bins[bin_ix] + ofs, huse + 40, str(size_of_bin_positive_none[bin_ix]), color = "#2e85ff")
        if not size_of_bin_positive_none[bin_ix] > 0 and size_of_bin_negative_none[bin_ix] > 0:
            plt.text(total_bins[bin_ix] + ofs, huse + 40, str(size_of_bin_negative_none[bin_ix]), color = "#ff120a")
        if size_of_bin_positive_none[bin_ix] > 0 and size_of_bin_negative_none[bin_ix] > 0:
            plt.text(total_bins[bin_ix] + ofs, huse + 80, str(size_of_bin_positive_none[bin_ix]), color = "#2e85ff")
            plt.text(total_bins[bin_ix] + ofs, huse + 40, str(size_of_bin_negative_none[bin_ix]), color = "#ff120a")
    plt.ylim(0, 750)
    plt.xlabel("Self assembly probability")
    plt.ylabel("Number of peptides")
    plt.title(
        merge_type_test_number(model_type, test_number).replace("Test 0 Weak 1", "").replace("Test 0", "")
    )
    plt.savefig(save + "_all_numbers.png", bbox_inches="tight")
    plt.savefig(save + "_all_numbers.svg", bbox_inches="tight")
    plt.savefig(save + "_all_numbers.pdf", bbox_inches="tight")
    plt.close()
    plt.rcParams.update({"font.size": 22})

def hist_predicted_merged(
    model_type, test_number, test_labels, model_predictions, save
):
    plt.rcParams.update({"font.size": 12})
    # Create a histogram of the predicted probabilities only for the peptides that show self-assembly
    model_predictions_true = []
    model_predictions_false = []
    for x in range(len(test_labels)):
        if test_labels[x] == 1.0:
            model_predictions_true.append(float(model_predictions[x]))
        else:
            model_predictions_false.append(float(model_predictions[x]))

    plt.figure()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # Draw the density plot
    sns.displot(
        {"SA": model_predictions_true},
        kde=True,
        bins=total_bins,
        palette={"SA": "#2e85ff"},
        legend=False,
    )
    plt.ylim(0, 750)
    plt.xlabel("Self assembly probability")
    plt.ylabel("Number of peptides")
    plt.title(
        merge_type_test_number(model_type, test_number).replace("Test 0 Weak 1", "").replace("Test 0", "")
    )
    plt.savefig(save + "_SA.png", bbox_inches="tight")
    plt.savefig(save + "_SA.svg", bbox_inches="tight")
    plt.savefig(save + "_SA.pdf", bbox_inches="tight")
    plt.close()

    # Create a histogram of the predicted probabilities only for the peptides that don't show self-assembly

    plt.figure()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    sns.displot(
        {"NSA": model_predictions_false},
        kde=True,
        bins=total_bins,
        palette={"NSA": "#ff120a"},
        legend=False,
    )
    plt.ylim(0, 750)
    plt.xlabel("Self assembly probability")
    plt.ylabel("Number of peptides")
    plt.title(
        merge_type_test_number(model_type, test_number).replace("Test 0 Weak 1", "").replace("Test 0", "")
    )
    plt.savefig(save + "_NSA.png", bbox_inches="tight")
    plt.savefig(save + "_NSA.svg", bbox_inches="tight")
    plt.savefig(save + "_NSA.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    sns.displot(
        {"SA": model_predictions_true, "NSA": model_predictions_false},
        kde=True,
        bins=total_bins,
        palette={"SA": "#2e85ff", "NSA": "#ff120a"},
    )
    plt.ylim(0, 750)
    plt.xlabel("Self assembly probability")
    plt.ylabel("Number of peptides")
    plt.title(
        merge_type_test_number(model_type, test_number).replace("Test 0 Weak 1", "").replace("Test 0", "")
    )
    plt.savefig(save + "_all.png", bbox_inches="tight")
    plt.savefig(save + "_all.svg", bbox_inches="tight")
    plt.savefig(save + "_all.pdf", bbox_inches="tight")
    plt.close()
    plt.rcParams.update({"font.size": 22})

def hist_predicted_merged_numbers_models(
    model_type_all, test_number_all, test_labels_all, model_predictions_all, labuse, start_space, space_model, text_use, scatter_use, line_use, kde_use
):
    xt = []
    xl = []
    for ix_model in range(len(model_type_all)):
        xt.append(ix_model * (1 + space_model) + space_model + start_space)
        xt.append(0.5 + ix_model * (1 + space_model) + space_model + start_space)
        xt.append(1 + ix_model * (1 + space_model) + space_model + start_space)
        xl.append(str(0.0))
        xl.append(str(0.5))
        xl.append(str(1.0))
    model_predictions_true = []
    model_predictions_false = []
    size_of_bin_positive_low = dict()
    size_of_bin_positive_high = dict()
    size_of_bin_positive_both = dict()
    size_of_bin_positive_none = dict()
    size_of_bin_negative_low = dict()
    size_of_bin_negative_high = dict()
    size_of_bin_negative_both = dict()
    size_of_bin_negative_none = dict()
    for ix_model in range(len(model_type_all)):
        model_type = model_type_all[ix_model]
        test_labels = test_labels_all[ix_model]
        model_predictions = model_predictions_all[ix_model]
        # Create a histogram of the predicted probabilities only for the peptides that show self-assembly
        model_predictions_true.append([])
        model_predictions_false.append([])
        for x in range(len(test_labels)):
            if test_labels[x] == 1.0:
                model_predictions_true[-1].append(float(model_predictions[x]))
            else:
                model_predictions_false[-1].append(float(model_predictions[x]))

        size_of_bin_positive_low[ix_model] = dict()
        size_of_bin_positive_high[ix_model] = dict()
        size_of_bin_positive_both[ix_model] = dict()
        size_of_bin_positive_none[ix_model] = dict()
        for bin_ix in range(len(total_bins) - 1):
            size_of_bin_positive_low[ix_model][bin_ix] = 0
            size_of_bin_positive_high[ix_model][bin_ix] = 0
            size_of_bin_positive_both[ix_model][bin_ix] = 0
            size_of_bin_positive_none[ix_model][bin_ix] = 0
        for val in model_predictions_true[-1]:
            for bin_ix in range(len(total_bins) - 1):
                bin_start = total_bins[bin_ix]
                bin_end = total_bins[bin_ix + 1]
                if val >= bin_start and val < bin_end:
                    size_of_bin_positive_low[ix_model][bin_ix] += 1
                if val > bin_start and val <= bin_end:
                    size_of_bin_positive_high[ix_model][bin_ix] += 1
                if val >= bin_start and val <= bin_end:
                    size_of_bin_positive_both[ix_model][bin_ix] += 1
                if val > bin_start and val < bin_end:
                    size_of_bin_positive_none[ix_model][bin_ix] += 1
        for bin_ix in range(len(total_bins) - 1):
            bin_start = total_bins[bin_ix]
            bin_end = total_bins[bin_ix + 1]
            if size_of_bin_positive_low[ix_model][bin_ix] != size_of_bin_positive_high[ix_model][bin_ix]:
                print("lh", bin_ix, bin_start, bin_end, size_of_bin_positive_low[ix_model][bin_ix], size_of_bin_positive_high[ix_model][bin_ix])
            if size_of_bin_positive_low[ix_model][bin_ix] != size_of_bin_positive_both[ix_model][bin_ix]:
                print("lb", bin_ix, bin_start, bin_end, size_of_bin_positive_low[ix_model][bin_ix], size_of_bin_positive_both[ix_model][bin_ix])
            if size_of_bin_positive_low[ix_model][bin_ix] != size_of_bin_positive_none[ix_model][bin_ix]:
                print("ln", bin_ix, bin_start, bin_end, size_of_bin_positive_low[ix_model][bin_ix], size_of_bin_positive_none[ix_model][bin_ix])
            if size_of_bin_positive_high[ix_model][bin_ix] != size_of_bin_positive_both[ix_model][bin_ix]:
                print("hb", bin_ix, bin_start, bin_end, size_of_bin_positive_high[ix_model][bin_ix], size_of_bin_positive_both[ix_model][bin_ix])
            if size_of_bin_positive_high[ix_model][bin_ix] != size_of_bin_positive_none[ix_model][bin_ix]:
                print("hn", bin_ix, bin_start, bin_end, size_of_bin_positive_high[ix_model][bin_ix], size_of_bin_positive_none[ix_model][bin_ix])
            if size_of_bin_positive_both[ix_model][bin_ix] != size_of_bin_positive_none[ix_model][bin_ix]:
                print("bn", bin_ix, bin_start, bin_end, size_of_bin_positive_both[ix_model][bin_ix], size_of_bin_positive_none[ix_model][bin_ix])
        print("positive", model_type, size_of_bin_positive_none)

        size_of_bin_negative_low[ix_model] = dict()
        size_of_bin_negative_high[ix_model] = dict()
        size_of_bin_negative_both[ix_model] = dict()
        size_of_bin_negative_none[ix_model] = dict()
        for bin_ix in range(len(total_bins) - 1):
            size_of_bin_negative_low[ix_model][bin_ix] = 0
            size_of_bin_negative_high[ix_model][bin_ix] = 0
            size_of_bin_negative_both[ix_model][bin_ix] = 0
            size_of_bin_negative_none[ix_model][bin_ix] = 0
        for val in model_predictions_false[-1]:
            for bin_ix in range(len(total_bins) - 1):
                bin_start = total_bins[bin_ix]
                bin_end = total_bins[bin_ix + 1]
                if val >= bin_start and val < bin_end:
                    size_of_bin_negative_low[ix_model][bin_ix] += 1
                if val > bin_start and val <= bin_end:
                    size_of_bin_negative_high[ix_model][bin_ix] += 1
                if val >= bin_start and val <= bin_end:
                    size_of_bin_negative_both[ix_model][bin_ix] += 1
                if val > bin_start and val < bin_end:
                    size_of_bin_negative_none[ix_model][bin_ix] += 1
        for bin_ix in range(len(total_bins) - 1):
            bin_start = total_bins[bin_ix]
            bin_end = total_bins[bin_ix + 1]
            if size_of_bin_negative_low[ix_model][bin_ix] != size_of_bin_negative_high[ix_model][bin_ix]:
                print("lh", bin_ix, bin_start, bin_end, size_of_bin_negative_low[ix_model][bin_ix], size_of_bin_negative_high[ix_model][bin_ix])
            if size_of_bin_negative_low[ix_model][bin_ix] != size_of_bin_negative_both[ix_model][bin_ix]:
                print("lb", bin_ix, bin_start, bin_end, size_of_bin_negative_low[ix_model][bin_ix], size_of_bin_negative_both[ix_model][bin_ix])
            if size_of_bin_negative_low[ix_model][bin_ix] != size_of_bin_negative_none[ix_model][bin_ix]:
                print("ln", bin_ix, bin_start, bin_end, size_of_bin_negative_low[ix_model][bin_ix], size_of_bin_negative_none[ix_model][bin_ix])
            if size_of_bin_negative_high[ix_model][bin_ix] != size_of_bin_negative_both[ix_model][bin_ix]:
                print("hb", bin_ix, bin_start, bin_end, size_of_bin_negative_high[ix_model][bin_ix], size_of_bin_negative_both[ix_model][bin_ix])
            if size_of_bin_negative_high[ix_model][bin_ix] != size_of_bin_negative_none[ix_model][bin_ix]:
                print("hn", bin_ix, bin_start, bin_end, size_of_bin_negative_high[ix_model][bin_ix], size_of_bin_negative_none[ix_model][bin_ix])
            if size_of_bin_negative_both[ix_model][bin_ix] != size_of_bin_negative_none[ix_model][bin_ix]:
                print("bn", bin_ix, bin_start, bin_end, size_of_bin_negative_both[ix_model][bin_ix], size_of_bin_negative_none[ix_model][bin_ix])
        print("negative", model_type, size_of_bin_negative_none)

    plt.rcParams["svg.fonttype"] = "none"
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

    plt.figure(figsize=(xas*cm, yas*cm), dpi = 300)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # Draw the density plot
    tmp_bins = []
    ksp = dict()
    pp = dict()
    for ix_model in range(len(model_type_all)):
        model_predictions_true_tmp = []
        for tmp_pred in model_predictions_true[ix_model]:
            model_predictions_true_tmp.append(tmp_pred + ix_model * (1 + space_model) + space_model + start_space)
        tmp_bins_one = [b + ix_model * (1 + space_model) + space_model + start_space for b in total_bins]
        for tb in tmp_bins_one:
            tmp_bins.append(tb)
        ksp["SA " + merge_type_test_number(model_type_all[ix_model], test_number_all[ix_model]).replace("Test 0 Weak 1", "").replace("Test 0", "").replace("Model ", "")] = model_predictions_true_tmp
        pp["SA " + merge_type_test_number(model_type_all[ix_model], test_number_all[ix_model]).replace("Test 0 Weak 1", "").replace("Test 0", "").replace("Model ", "")] = "#2e85ff"
    gp = sns.displot(
        ksp,
        kde=kde_use,
        bins=tmp_bins,
        height = yas, aspect = xas / yas, palette=pp,
        legend=False,
    )
    for ix_model in range(len(model_type_all)):
        bin_mids = []
        bin_hs = []
        for bin_ix in range(len(total_bins) * ix_model, len(total_bins) * (ix_model + 1) - 1):
            bin_mids.append(tmp_bins[bin_ix] + 0.05)
            bin_hs.append(size_of_bin_positive_none[ix_model][bin_ix - len(total_bins) * ix_model])
        if line_use:
            plt.plot(bin_mids, bin_hs, linewidth = 1, color = "#2e85ff")
        if scatter_use:
            plt.scatter(bin_mids, bin_hs, s = 2, color = "#2e85ff")
        for bin_ix in range(len(total_bins) * ix_model, len(total_bins) * (ix_model + 1) - 1):
            husep = size_of_bin_positive_none[ix_model][bin_ix - len(total_bins) * ix_model]
            if bin_ix > len(total_bins) * ix_model and bin_ix < len(total_bins) * (ix_model + 1) - 2:
                prevhusep = size_of_bin_positive_none[ix_model][bin_ix - 1 - len(total_bins) * ix_model]
                nexthusep = size_of_bin_positive_none[ix_model][bin_ix + 1 - len(total_bins) * ix_model]
                avghusep = (prevhusep + nexthusep) / 2
                husep = max(avghusep, husep)
            ofs = 1 / 80
            if size_of_bin_positive_none[ix_model][bin_ix - len(total_bins) * ix_model] < 100:
                ofs = 1 / 60
            if size_of_bin_positive_none[ix_model][bin_ix - len(total_bins) * ix_model] > 0 and text_use:
                plt.text(tmp_bins[bin_ix] + ofs, husep + 40, str(size_of_bin_positive_none[ix_model][bin_ix - len(total_bins) * ix_model]), color = "#2e85ff")
    plt.ylim(0, 750)
    plt.xlim(0, start_space + (space_model + 1) * len(model_type_all) + 0.1)
    plt.xticks(xt, xl)
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides")
    plt.savefig(labuse + "_SA.png", bbox_inches="tight")
    plt.savefig(labuse + "_SA.svg", bbox_inches="tight")
    plt.savefig(labuse + "_SA.pdf", bbox_inches="tight")
    plt.close()

    plt.rcParams["svg.fonttype"] = "none"
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

    plt.figure(figsize=(xas*cm, yas*cm), dpi = 300)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # Create a histogram of the predicted probabilities only for the peptides that don't show self-assembly
    ksn = dict()
    pn = dict()
    for ix_model in range(len(model_type_all)):
        model_predictions_false_tmp = []
        for tmp_pred in model_predictions_false[ix_model]:
            model_predictions_false_tmp.append(tmp_pred + ix_model * (1 + space_model) + space_model + start_space)
        ksn["NSA " + merge_type_test_number(model_type_all[ix_model], test_number_all[ix_model]).replace("Test 0 Weak 1", "").replace("Test 0", "").replace("Model ", "")] = model_predictions_false_tmp
        pn["NSA " + merge_type_test_number(model_type_all[ix_model], test_number_all[ix_model]).replace("Test 0 Weak 1", "").replace("Test 0", "").replace("Model ", "")] = "#ff120a"
    gn = sns.displot(
        ksn,
        kde=kde_use,
        bins=tmp_bins,
        height = yas, aspect = xas / yas, palette=pn,
        legend=False,
    )
    for ix_model in range(len(model_type_all)):
        bin_mids = []
        bin_hs = []
        for bin_ix in range(len(total_bins) * ix_model, len(total_bins) * (ix_model + 1) - 1):
            bin_mids.append(tmp_bins[bin_ix] + 0.05)
            bin_hs.append(size_of_bin_negative_none[ix_model][bin_ix - len(total_bins) * ix_model])
        if line_use:
            plt.plot(bin_mids, bin_hs, linewidth = 1, color = "#ff120a")
        if scatter_use:
            plt.scatter(bin_mids, bin_hs, s = 2, color = "#ff120a")
        for bin_ix in range(len(total_bins) * ix_model, len(total_bins) * (ix_model + 1) - 1):
            husen = size_of_bin_negative_none[ix_model][bin_ix - len(total_bins) * ix_model]
            if bin_ix > len(total_bins) * ix_model and bin_ix < len(total_bins) * (ix_model + 1) - 2:
                prevhusen = size_of_bin_negative_none[ix_model][bin_ix - 1 - len(total_bins) * ix_model]
                nexthusen = size_of_bin_negative_none[ix_model][bin_ix + 1 - len(total_bins) * ix_model]
                avghusen = (prevhusen + nexthusen) / 2
                husen = max(avghusen, husen)
            ofs = 1 / 80
            if size_of_bin_negative_none[ix_model][bin_ix - len(total_bins) * ix_model] < 100:
                ofs = 1 / 60
            if size_of_bin_negative_none[ix_model][bin_ix - len(total_bins) * ix_model] > 0 and text_use:
                plt.text(tmp_bins[bin_ix] + ofs, husen + 40, str(size_of_bin_negative_none[ix_model][bin_ix - len(total_bins) * ix_model]), color = "#ff120a")
    plt.ylim(0, 750)
    plt.xlim(0, start_space + (space_model + 1) * len(model_type_all) + 0.1)
    plt.xticks(xt, xl)
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides")
    plt.savefig(labuse + "_NSA.png", bbox_inches="tight")
    plt.savefig(labuse + "_NSA.svg", bbox_inches="tight")
    plt.savefig(labuse + "_NSA.pdf", bbox_inches="tight")
    plt.close()

    plt.rcParams["svg.fonttype"] = "none"
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

    plt.figure(figsize=(xas*cm, yas*cm), dpi = 300)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    kst = dict()
    pt = dict()
    for ks in ksp:
        kst[ks] = ksp[ks]
        pt[ks] = pp[ks]
    for ks in ksn:
        kst[ks] = ksn[ks]
        pt[ks] = pn[ks]
    sns.displot(
        kst,
        kde=kde_use,
        bins=tmp_bins,
        height = yas, aspect = xas / yas, palette=pt,
        legend = False
    )
    for ix_model in range(len(model_type_all)):
        bin_mids = []
        bin_hs_p = []
        bin_hs_n = []
        for bin_ix in range(len(total_bins) * ix_model, len(total_bins) * (ix_model + 1) - 1):
            bin_mids.append(tmp_bins[bin_ix] + 0.05)
            bin_hs_p.append(size_of_bin_positive_none[ix_model][bin_ix - len(total_bins) * ix_model])
            bin_hs_n.append(size_of_bin_negative_none[ix_model][bin_ix - len(total_bins) * ix_model])
        if line_use:
            plt.plot(bin_mids, bin_hs_p, linewidth = 1, color = "#2e85ff")
            plt.plot(bin_mids, bin_hs_n, linewidth = 1, color = "#ff120a")
        if scatter_use:
            plt.scatter(bin_mids, bin_hs_p, s = 2, color = "#2e85ff")
            plt.scatter(bin_mids, bin_hs_n, s = 2, color = "#ff120a")
        for bin_ix in range(len(total_bins) * ix_model, len(total_bins) * (ix_model + 1) - 1):
            husen = size_of_bin_negative_none[ix_model][bin_ix - len(total_bins) * ix_model]
            if bin_ix > len(total_bins) * ix_model and bin_ix < len(total_bins) * (ix_model + 1) - 2:
                prevhusen = size_of_bin_negative_none[ix_model][bin_ix - 1 - len(total_bins) * ix_model]
                nexthusen = size_of_bin_negative_none[ix_model][bin_ix + 1 - len(total_bins) * ix_model]
                avghusen = (prevhusen + nexthusen) / 2
                husen = max(avghusen, husen)
            husep = size_of_bin_positive_none[ix_model][bin_ix - len(total_bins) * ix_model]
            if bin_ix > len(total_bins) * ix_model and bin_ix < len(total_bins) * (ix_model + 1) - 2:
                prevhusep = size_of_bin_positive_none[ix_model][bin_ix - 1 - len(total_bins) * ix_model]
                nexthusep = size_of_bin_positive_none[ix_model][bin_ix + 1 - len(total_bins) * ix_model]
                avghusep = (prevhusep + nexthusep) / 2
                husep = max(avghusep, husep)
            huse = max(husen, husep)
            ofs = 1 / 80
            if size_of_bin_positive_none[ix_model][bin_ix - len(total_bins) * ix_model] < 100 and size_of_bin_negative_none[ix_model][bin_ix - len(total_bins) * ix_model] < 100:
                ofs = 1 / 60
            if text_use:
                if size_of_bin_positive_none[ix_model][bin_ix - len(total_bins) * ix_model] > 0 and not size_of_bin_negative_none[ix_model][bin_ix - len(total_bins) * ix_model] > 0:
                    plt.text(tmp_bins[bin_ix] + ofs, huse + 40, str(size_of_bin_positive_none[ix_model][bin_ix - len(total_bins) * ix_model]), color = "#2e85ff")
                if not size_of_bin_positive_none[ix_model][bin_ix - len(total_bins) * ix_model] > 0 and size_of_bin_negative_none[ix_model][bin_ix - len(total_bins) * ix_model] > 0:
                    plt.text(tmp_bins[bin_ix] + ofs, huse + 40, str(size_of_bin_negative_none[ix_model][bin_ix - len(total_bins) * ix_model]), color = "#ff120a")
                if size_of_bin_positive_none[ix_model][bin_ix - len(total_bins) * ix_model] > 0 and size_of_bin_negative_none[ix_model][bin_ix - len(total_bins) * ix_model] > 0:
                    plt.text(tmp_bins[bin_ix] + ofs, huse + 90, str(size_of_bin_positive_none[ix_model][bin_ix - len(total_bins) * ix_model]), color = "#2e85ff")
                    plt.text(tmp_bins[bin_ix] + ofs, huse + 40, str(size_of_bin_negative_none[ix_model][bin_ix - len(total_bins) * ix_model]), color = "#ff120a")
    plt.ylim(0, 750)
    plt.xlim(0, start_space + (space_model + 1) * len(model_type_all) + 0.1)
    plt.xticks(xt, xl)
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides")
    plt.savefig(labuse + "_all.png", bbox_inches="tight")
    plt.savefig(labuse + "_all.svg", bbox_inches="tight")
    plt.savefig(labuse + "_all.pdf", bbox_inches="tight")
    plt.close()
    plt.rcParams.update({"font.size": 22})

def read_one_result(some_path, test_number):
    file = open(results_name(some_path, test_number), "r")
    lines = file.readlines()
    file.close()
    return lines


def read_ROC(test_labels, model_predictions, name):
    # Get false positive rate and true positive rate.
    fpr, tpr, thresholds = roc_curve(test_labels, model_predictions)

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ix = np.argmax(gmeans)

    plt.figure()
    fig = plt.gcf()
    ax = fig.gca()
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    plt.arrow(
        fpr[ix],
        tpr[ix],
        -fpr[ix],
        1 - tpr[ix],
        length_includes_head=True,
        head_width=0.02,
    )

    # Plot ROC curve.
    plt.plot(fpr, tpr, "r", label="model performance")
    plt.plot(fpr[ix], tpr[ix], "o", markerfacecolor="r", markeredgecolor="k")

    # Plot random guessing ROC curve.
    plt.plot([0, 1], [0, 1], "c", label="random guessing")

    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.savefig(
        "../seeds/all_seeds/" + name + "_ROC.png",
        bbox_inches="tight",
    )
    plt.savefig(
        "../seeds/all_seeds/" + name + "_ROC.svg",
        bbox_inches="tight",
    )
    plt.savefig(
        "../seeds/all_seeds/" + name + "_ROC.pdf",
        bbox_inches="tight",
    )
    plt.close()


def read_PR(test_labels, model_predictions, name):
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

    plt.figure()
    plt.title(name + " model" + "\nPrecision - Recall (PR) curve")
    fig = plt.gcf()
    ax = fig.gca()
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

    plt.arrow(
        recall[ix],
        precision[ix],
        1 - recall[ix],
        1 - precision[ix],
        length_includes_head=True,
        head_width=0.02,
    )

    # Plot PR curve.
    plt.plot(recall, precision, "r", label="model performance")
    plt.plot(recall[ix], precision[ix], "o", markerfacecolor="r", markeredgecolor="k")

    # Calculate the no skill line as the proportion of the positive class
    num_positive = 0
    for value in test_labels:
        if value == 1:
            num_positive += 1
    no_skill = num_positive / len(test_labels)

    # Plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], "c", label="random guessing")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.savefig(
        "../seeds/all_seeds/" + name + "_PR.png",
        bbox_inches="tight",
    )
    plt.savefig(
        "../seeds/all_seeds/" + name + "_PR.svg",
        bbox_inches="tight",
    )
    plt.savefig(
        "../seeds/all_seeds/" + name + "_PR.pdf",
        bbox_inches="tight",
    )
    plt.close()


# Increase font size of all elements
plt.rcParams.update({"font.size": 22})

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
paths = [
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
]
NUM_TESTS = 5
all_preds = []
all_labels_new = []
all_model_types = []
list_paths = []
list_nums = []
list_labels = []
list_preds = []
list_TSNE_paths = []
list_TSNE_nums = []
list_TSNE_labels = []
list_TSNE_preds = []
for some_path in paths:
    seed_predictions = []
    seed_labels = []
    for seed in seed_list:
        set_seed(seed)
        all_predictions, all_labels = read_all_model_predictions(some_path, 1, 5)
        for pred in all_predictions:
            seed_predictions.append(pred)
            all_preds.append(pred)
        for label in all_labels:
            seed_labels.append(label)
            if label == 0.0:
                all_labels_new.append("NSA")
            else:
                all_labels_new.append("SA")
            all_model_types.append(
                PATH_TO_NAME[some_path]
                .replace("SP and AP", "Hybrid AP-SP")
                .replace("TSNE", "t-SNE")
            )

    hist_predicted_merged(
        some_path,
        0,
        seed_labels,
        seed_predictions,
        "../seeds/all_seeds/" + PATH_TO_EXTENSION[some_path] + "_hist_merged_seeds",
    )
    hist_predicted_merged_numbers(
        some_path,
        0,
        seed_labels,
        seed_predictions,
        "../seeds/all_seeds/" + PATH_TO_EXTENSION[some_path] + "_hist_merged_seeds",
    )
    if "TSNE" in some_path:
        list_TSNE_paths.append(some_path)
        list_TSNE_nums.append(0)
        list_TSNE_labels.append(seed_labels)
        list_TSNE_preds.append(seed_predictions)
    else:
        list_paths.append(some_path)
        list_nums.append(0)
        list_labels.append(seed_labels)
        list_preds.append(seed_predictions)

tf = [True, False]

for use_number in tf:
    for use_scatter in tf:
        for use_line in tf:
            for use_kde in tf:
                name_curr = "../seeds/all_seeds/all_models_hist_merged_seeds"
                if use_number:
                    name_curr += "_numbers"
                if use_scatter:
                    name_curr += "_dot"
                if use_line:
                    name_curr += "_line"
                if not use_kde:
                    name_curr += "_no_bar"

                hist_predicted_merged_numbers_models(
                    list_TSNE_paths,
                    list_TSNE_nums,
                    list_TSNE_labels,
                    list_TSNE_preds,
                    name_curr + "_TSNE", 1.5, 0.5,
                    use_number, use_scatter, use_line, use_kde
                )

                hist_predicted_merged_numbers_models(
                    list_paths,
                    list_nums,
                    list_labels,
                    list_preds,
                    name_curr + "_no_TSNE", 0, 0.5,
                    use_number, use_scatter, use_line, use_kde
                )

APpreds = []
SPpreds = []
SPAPpreds = []
TSNESPpreds = []
TSNESPAPpreds = []
APN = []
SPN = []
SPAPN = []
TSNESPN = []
TSNESPAPN = []
APY = []
SPY = []
SPAPY = []
TSNESPY = []
TSNESPAPY = []

for i in range(len(all_preds)):
    if all_model_types[i] == "AP":
        APpreds.append(all_preds[i])
        if all_labels_new[i] == "SA":
            APY.append(all_preds[i])
        else:
            APN.append(all_preds[i])
        continue
    if all_model_types[i] == "SP":
        SPpreds.append(all_preds[i])
        if all_labels_new[i] == "SA":
            SPY.append(all_preds[i])
        else:
            SPN.append(all_preds[i])
        continue
    if all_model_types[i] == "TSNE SP":
        TSNESPpreds.append(all_preds[i])
        if all_labels_new[i] == "SA":
            TSNESPY.append(all_preds[i])
        else:
            TSNESPN.append(all_preds[i])
        continue
    if all_model_types[i] == "TSNE AP-SP":
        TSNESPAPpreds.append(all_preds[i])
        if all_labels_new[i] == "SA":
            TSNESPAPY.append(all_preds[i])
        else:
            TSNESPAPN.append(all_preds[i])
        continue
    SPAPpreds.append(all_preds[i])
    if all_labels_new[i] == "SA":
        SPAPY.append(all_preds[i])
    else:
        SPAPN.append(all_preds[i])

d = {
    "Predicted self assembly probability": all_preds,
    "Self assembly status": all_labels_new,
    "Model": all_model_types,
}
df = pd.DataFrame(data=d)
plt.figure()
g = sns.displot(
    data=df,
    x="Predicted self assembly probability",
    kde=True,
    bins=total_bins,
    hue="Self assembly status",
    col="Model",
    palette={"NSA": "#ff120a", "SA": "#2e85ff"},
)
g.set_axis_labels("Self assembly probability", "Number of peptides")
g.set_titles("{col_name} model")
plt.close()

names = ["SP", "Hybrid AP-SP", "AP", "t-SNE SP", "t-SNE AP-SP"]
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
paths = [
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
]
NUM_TESTS = 5
all_preds = []
all_labels_new = []
all_model_types = []
ind = 0
for some_path in paths:
    seed_predictions = []
    seed_labels = []
    for seed in seed_list:
        set_seed(seed)
        all_predictions, all_labels = read_all_model_predictions(some_path, 1, 5)
        for pred in all_predictions:
            seed_predictions.append(pred)
            all_preds.append(pred)
        for label in all_labels:
            seed_labels.append(label)
            if label == 0.0:
                all_labels_new.append("NSA")
            else:
                all_labels_new.append("SA")
            all_model_types.append(
                PATH_TO_NAME[some_path]
                .replace("SP and AP", "Hybrid AP-SP")
                .replace("TSNE", "t-SNE")
            )

    read_PR(seed_labels, seed_predictions, names[ind])
    read_ROC(seed_labels, seed_predictions, names[ind])

    ind += 1
