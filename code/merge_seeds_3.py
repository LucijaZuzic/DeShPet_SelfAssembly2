from utils import (
    set_seed,
    predictions_name,
    final_history_name,
    PATH_TO_EXTENSION,
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
)
from custom_plots import (
    merge_type_test_number,
    results_name,
    my_accuracy_calculate,
    weird_division,
    convert_to_binary,
)
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    auc,
    f1_score,
)


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
    lines_dict["Minimal loss = "].append(np.min(all_loss))
    lines_dict["Accuracy = "].append(np.mean(all_acc) * 100)
    lines_dict["Loss = "].append(np.mean(all_loss))
    sd_dict["Accuracy = "].append(np.std(all_acc) * 100)
    sd_dict["Loss = "].append(np.std(all_loss))


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

    lines_dict["ROC thr = "].append(rocthr)
    lines_dict["gmean (0.5) = "].append(
        returnGMEAN(test_labels, model_predictions_binary)
    )
    lines_dict["gmean (PR thr) = "].append(
        returnGMEAN(test_labels, model_predictions_binary_thrPR)
    )
    lines_dict["gmean (ROC thr) = "].append(
        returnGMEAN(test_labels, model_predictions_binary_thrROC)
    )
    lines_dict["ROC AUC = "].append(roc_auc_score(test_labels, model_predictions))
    lines_dict["Accuracy (ROC thr) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, rocthr)
    )


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

    lines_dict["PR thr = "].append(prthr)
    lines_dict["PR AUC = "].append(auc(recall, precision))
    lines_dict["F1 (0.5) = "].append(f1_score(test_labels, model_predictions_binary))
    lines_dict["F1 (PR thr) = "].append(
        f1_score(test_labels, model_predictions_binary_thrPR)
    )
    lines_dict["F1 (ROC thr) = "].append(
        f1_score(test_labels, model_predictions_binary_thrROC)
    )
    lines_dict["Accuracy (PR thr) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, prthr)
    )
    lines_dict["Accuracy (0.5) = "].append(
        my_accuracy_calculate(test_labels, model_predictions, 0.5)
    )


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


def hist_predicted_multiple(
    model_type, test_number, test_labels, model_predictions, seeds, save
):
    # Create a histogram of the predicted probabilities only for the peptides that show self-assembly
    plot_label = []
    for seed in seeds:
        plot_label.append("Seed " + str(seed))

    model_predictions_true = []
    for model_num in range(len(model_predictions)):
        model_predictions_true.append([])
        for x in range(len(test_labels[model_num])):
            if test_labels[model_num][x] == 1.0:
                model_predictions_true[-1].append(
                    float(model_predictions[model_num][x])
                )

    plt.figure()
    plt.title(
        merge_type_test_number(model_type, test_number).replace(
            "Test 0 Weak 1", "All tests"
        )
        + "\nHistogram of predicted self assembly probability\nfor peptides with self assembly"
    )
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides")
    plt.hist(
        model_predictions_true,
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        label=plot_label,
    )
    plt.legend()
    #plt.savefig(save + "_SA.png")
    plt.close()

    # Create a histogram of the predicted probabilities only for the peptides that don't show self-assembly

    model_predictions_false = []
    for model_num in range(len(model_predictions)):
        model_predictions_false.append([])
        for x in range(len(test_labels[model_num])):
            if test_labels[model_num][x] == 0.0:
                model_predictions_false[-1].append(
                    float(model_predictions[model_num][x])
                )

    plt.figure()
    plt.title(
        merge_type_test_number(model_type, test_number).replace(
            "Test 0 Weak 1", "All tests"
        )
        + "\nHistogram of predicted self assembly probability\nfor peptides without self assembly"
    )
    plt.xlabel("Predicted self assembly probability")
    plt.ylabel("Number of peptides")
    plt.hist(
        model_predictions_false,
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        label=plot_label,
    )
    plt.legend()
    #plt.savefig(save + "_NSA.png")
    plt.close()


def read_one_result(some_path, test_number):
    file = open(results_name(some_path, test_number), "r")
    lines = file.readlines()
    file.close()
    return lines


def interesting_lines(some_path, test_number, find_str):
    lines = read_one_result(some_path, test_number)
    new_lines = []
    for line_num in range(len(lines)):
        if lines[line_num].count(find_str) > 0 and (
            lines[line_num].count("Weak 1") > 0
            or lines[line_num].count("Best params") > 0
        ):
            new_lines.append(lines[line_num])
    return new_lines


def findValInLine(line, name):
    if line.find(name) == -1:
        return False
    start = line.find(name) + len(name)
    isFloat = False
    end = start
    while (ord(line[end]) >= ord("0") and ord(line[end]) <= ord("9")) or line[
        end
    ] == ".":
        if line[end] == ".":
            isFloat = True
        end += 1

    if isFloat:
        return float(line[start:end])
    else:
        return int(line[start:end])


def doubleArrayToTable(array1, array2, addArray, addPercent):
    retstr = ""
    for i in range(len(array1)):
        if addArray:
            if not addPercent:
                retstr += " & %.5f (%.5f)" % (array1[i], array2[i])
            else:
                retstr += " & %.2f\\%% (%.2f\\%%)" % (array1[i], array2[i])
    if not addPercent:
        retstr += " & %.5f (%.5f)" % (np.mean(array1), np.mean(array2))
    else:
        retstr += " & %.2f\\%% (%.2f\\%%)" % (np.mean(array1), np.mean(array2))
    return retstr + " \\\\"


def arrayToTable(array, addArray, addAvg, addMostFrequent):
    retstr = ""
    suma = 0
    freqdict = dict()
    for i in array:
        if addArray:
            if addMostFrequent:
                retstr += " & %d" % (i)
            if addAvg:
                retstr += " & %.5f" % (i)
        suma += i
        if i in freqdict:
            freqdict[i] = freqdict[i] + 1
        else:
            freqdict[i] = 1
    most_freq = []
    num_most = 0
    for i in freqdict:
        if freqdict[i] >= num_most:
            if freqdict[i] > num_most:
                most_freq = []
            most_freq.append(i)
            num_most = freqdict[i]
    if addAvg:
        retstr += " & %.5f" % (np.mean(array))
    if addMostFrequent:
        most_freq_str = str(sorted(most_freq)[0])
        for i in sorted(most_freq[1:]):
            most_freq_str += ", " + str(i)
        retstr += " & %s (%d/%d)" % (most_freq_str, num_most, len(array))
    return retstr + " \\\\"


def arrayToTableOnlyFreq(array):
    retstr = ""
    all_freqdict = dict()
    all_most_freq = []
    all_num_most = 0
    for start in range(0, len(array), NUM_TESTS):
        freqdict = dict()
        end = start + NUM_TESTS
        for i in array[start:end]:
            if i in freqdict:
                freqdict[i] = freqdict[i] + 1
            else:
                freqdict[i] = 1
            if i in all_freqdict:
                all_freqdict[i] = all_freqdict[i] + 1
            else:
                all_freqdict[i] = 1
        most_freq = []
        num_most = 0
        for i in freqdict:
            if freqdict[i] >= num_most:
                if freqdict[i] > num_most:
                    most_freq = []
                most_freq.append(i)
                num_most = freqdict[i]
        for i in all_freqdict:
            if all_freqdict[i] >= all_num_most:
                if all_freqdict[i] > all_num_most:
                    all_most_freq = []
                all_most_freq.append(i)
                all_num_most = all_freqdict[i]
        most_freq_str = str(sorted(most_freq)[0])
        for i in sorted(most_freq)[1:]:
            most_freq_str += ", " + str(i)
        retstr += " & %s (%d/%d)" % (most_freq_str, num_most, len(array[start:end]))
    all_most_freq_str = str(sorted(all_most_freq)[0])
    for i in sorted(all_most_freq[1:]):
        all_most_freq_str += ", " + str(i)
    retstr += " & %s (%d/%d)" % (all_most_freq_str, all_num_most, len(array))
    return retstr + " \\\\"


if not os.path.exists("../seeds/all_seeds/"):
    os.makedirs("../seeds/all_seeds/")
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


0.2581173828
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
paths = [
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
]
NUM_TESTS = 5

for some_path in paths:
    seed_predictions = []
    seed_labels = []
    for seed in seed_list:
        set_seed(seed)
        all_predictions, all_labels = read_all_model_predictions(some_path, 1, 5)
        seed_predictions.append(all_predictions)
        seed_labels.append(all_labels)
    hist_predicted_multiple(
        some_path,
        0,
        seed_labels,
        seed_predictions,
        seed_list,
        "../seeds/all_seeds/" + PATH_TO_EXTENSION[some_path] + "_hist_all_seeds",
    )

vals_in_lines = [
    "num_cells: ",
    "kernel_size: ",
    "dense: ",
    "Maximum accuracy = ",
    "Minimal loss = ",
    "Accuracy = ",
    "Loss = ",
    "ROC thr = ",
    "PR thr = ",
    "ROC AUC = ",
    "gmean (ROC thr) = ",
    "F1 (ROC thr) = ",
    "Accuracy (ROC thr) = ",
    "PR AUC = ",
    "gmean (PR thr) = ",
    "F1 (PR thr) = ",
    "Accuracy (PR thr) = ",
    "gmean (0.5) = ",
    "F1 (0.5) = ",
    "Accuracy (0.5) = ",
]

for some_path in paths:
    lines_dict = dict()
    sd_dict = dict()

    for val in vals_in_lines:
        lines_dict[val] = []
        sd_dict[val] = []

    for seed in seed_list:
        set_seed(seed)
        interesting_lines_list = []
        for test_number in range(1, NUM_TESTS + 1):
            for line in interesting_lines(some_path, test_number, "Test"):
                interesting_lines_list.append(line)
        for i in range(len(interesting_lines_list)):
            for val in vals_in_lines:
                if val.find(":") == -1:
                    continue
                res = findValInLine(interesting_lines_list[i], val)
                if res != False:
                    lines_dict[val].append(res)
                    if val == "Loss = " or val == "Accuracy = ":
                        index = interesting_lines_list[i].find(val)
                        sd_dict[val].append(
                            findValInLine(interesting_lines_list[i][index:], "(")
                        )
        read_all_final_history(some_path, lines_dict, sd_dict)
        model_predictions, test_labels = read_all_model_predictions(some_path, 1, 5)
        read_PR(test_labels, model_predictions, lines_dict, PRthr[some_path], ROCthr[some_path])
        read_ROC(test_labels, model_predictions, lines_dict, PRthr[some_path], ROCthr[some_path])

    print(some_path)
    for val in vals_in_lines:
        if len(lines_dict[val]) == 0:
            continue
        if val == "Loss = ":
            print(
                val.replace(" = ", "")
                + doubleArrayToTable(lines_dict[val], sd_dict[val], True, True)
            )
            continue
        if val == "Accuracy = ":
            print(
                val.replace(" = ", "")
                + doubleArrayToTable(lines_dict[val], sd_dict[val], True, True)
            )
            continue
        else:
            if val.find(":") == -1:
                print(
                    val.replace(" = ", "")
                    + arrayToTable(lines_dict[val], True, True, True)
                )
            else:
                print(
                    val.replace(": ", "").replace("_", " ")
                    + arrayToTableOnlyFreq(lines_dict[val])
                )
