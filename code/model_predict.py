import tensorflow as tf
import numpy as np
import pandas as pd
from utils import convert_list, percent_grade_names, DATA_PATH
from custom_plots import (
    convert_to_binary,
    make_ROC_plots,
    make_PR_plots,
    output_metrics,
    hist_predicted,
)
from reshape import reshape
from merge_data import merge_data
from load_data import load_data_SA


def model_predict(
    path_used,
    test_data,
    test_labels,
    best_model_file,
    best_model,
    used_batch_size=600,
    names=["AP"],
):
    # Load the best model.
    if best_model_file != "":
        best_model = tf.keras.models.load_model(best_model_file)

    # Get model predictions on the test data.
    test_data, test_labels = reshape(path_used, test_data, test_labels, names)
    model_predictions = best_model.predict(test_data, batch_size=used_batch_size)
    model_predictions = convert_list(model_predictions)
    return model_predictions


def common_designed_peptides(
    test_number,
    path_used,
    model_file,
    model,
    used_batch_size=600,
    names=["AP"],
    offset=1,
    properties=[],
    masking_value=2,
    thr=0.5,
):
    # Get sequences for peptides with no labels and predictions from the model without machine learning
    resulteval = DATA_PATH + "RESULTEVAL.csv"
    df = pd.read_csv(resulteval, skipinitialspace=True, sep=";")
    sequences = list(df["Dizajnirani peptid"])
    seq_example = ""
    for i in range(24):
        seq_example += "A"
    sequences.append(seq_example)
    past_grades = list(df["Postotak svojstava koja imaju AUC > 0,5 koja su dala SA"])
    past_classes = list(df["SA"])

    SA_data = {}
    for i in range(len(sequences)):
        SA_data[sequences[i]] = "0"

    SA, NSA = load_data_SA(path_used, SA_data, names, offset, properties, masking_value)
    all_data, all_labels = merge_data(path_used, SA, NSA)
    model_predictions = model_predict(
        path_used, all_data, all_labels, model_file, model, used_batch_size, names
    )

    all_data = all_data[:-1]
    all_labels = all_labels[:-1]
    sequences = sequences[:-1]

    percentage_filename, grade_filename = percent_grade_names(path_used, test_number)

    # Write SA probability to file
    percentage_file = open(percentage_filename, "w", encoding="utf-8")
    percentage_string_to_write = (
        "Sequence;Multiple properties model;Method without RNN\n"
    )

    for x in range(len(sequences)):
        percentage_string_to_write += (
            sequences[x]
            + ";"
            + str(np.round(model_predictions[x] * 100, 2))
            + ";"
            + past_grades[x]
            + "\n"
        )
    percentage_string_to_write = percentage_string_to_write.replace(".", ",")
    percentage_file.write(percentage_string_to_write)
    percentage_file.close()

    # Write class based on the threshold of probability to file
    threshold_amino = thr

    model_predictions = convert_to_binary(model_predictions, threshold_amino)

    grade_file = open(grade_filename, "w", encoding="utf-8")
    grade_string_to_write = "Sequence;Multiple properties model;Method without RNN\n"

    correct_amino = 0
    for x in range(len(sequences)):
        if (model_predictions[x] == 1 and past_classes[x] == "Y") or (
            model_predictions[x] == 0 and past_classes[x] == "N"
        ):
            correct_amino += 1

        part1 = (
            sequences[x]
            + ";"
            + str(model_predictions[x])
            + ";"
            + past_classes[x]
            + "\n"
        )
        part1 = part1.replace(".0", "")
        part1 = part1.replace("1", "Y")
        part1 = part1.replace("0", "N")
        grade_string_to_write += part1
    last_line = (
        "Number of matches with method without RNN;" + str(correct_amino) + ";\n"
    )
    last_line = last_line.replace(".", ",")
    grade_string_to_write += last_line
    grade_file.write(grade_string_to_write)
    grade_file.close()


def common_predict(path_used, model_predictions, test_number, test_labels):
    # Plot ROC curves for all models
    make_ROC_plots(path_used, test_number, test_labels, model_predictions)

    # Plot PR curves for all models
    make_PR_plots(path_used, test_number, test_labels, model_predictions)

    # Output adjusted accuracy, F1 score and ROC AUC score for all models
    output_metrics(path_used, test_number, test_labels, model_predictions)

    # Output histograms that show the distribution of predicted probabilities of self-assembly for the SA and NSA class separately and for each model separately
    hist_predicted(path_used, test_number, test_labels, model_predictions)


def generate_predictions(
    model,
    test_number,
    path_used,
    test_data,
    test_labels,
    used_batch_size=600,
    names=["AP"],
    offset=1,
    properties=[],
    masking_value=2,
    thr=0.5,
):
    model_predictions = model_predict(
        path_used, test_data, test_labels, "", model, used_batch_size, names
    )
    common_predict(path_used, model_predictions, test_number, test_labels)
    # Generate predictions on data that has no label beforehand
    common_designed_peptides(
        test_number,
        path_used,
        "",
        model,
        used_batch_size,
        names,
        offset,
        properties,
        masking_value,
        thr,
    )
