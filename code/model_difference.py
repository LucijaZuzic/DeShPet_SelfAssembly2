from utils import (
    set_seed,
    predictions_name,
    PATH_TO_NAME,
    PATH_TO_EXTENSION,
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
)
import numpy as np
import pandas as pd

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
paths = [
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
]
N_FOLDS_FIRST = 5

header = "Sequence;Label;Prediction;Seed;Model;\n"
all_lines = header + ""

for some_seed in seed_list:
    set_seed(some_seed)
    seed_lines = header + ""

    for some_path in paths:
        names = ["AP"]
        if some_path == SP_DATA_PATH or some_path == TSNE_SP_DATA_PATH:
            names = []
        offset = 1
        properties = np.ones(95)
        properties[0] = 0
        masking_value = 2

        for test_number in range(1, N_FOLDS_FIRST + 1):
            file_df = pd.read_csv(
                "../seeds/seed_"
                + str(some_seed)
                + "/similarity/"
                + "test_fold_"
                + str(test_number)
                + ".csv"
            )
            file = open(predictions_name(some_path, test_number), "r")
            lines = file.readlines()
            predictions = eval(lines[0])
            labels = eval(lines[1])
            file.close()

            for i in range(len(predictions)):
                model_line = (
                    file_df["sequence"][i]
                    + ";"
                    + str(file_df["label"][i])
                    + ";"
                    + str(predictions[i])
                    + ";"
                    + str(some_seed)
                    + ";"
                    + PATH_TO_NAME[some_path]
                    + ";\n"
                )
                seed_lines += model_line
                all_lines += model_line

    file = open(
        "../seeds/all_seeds/predictions_all_models_seed_" + str(some_seed) + ".csv", "w"
    )
    file.write(seed_lines)
    file.close()

file = open("../seeds/all_seeds/predictions_all_models_all_seeds.csv", "w")
file.write(all_lines)
file.close()

for some_path in paths:
    names = ["AP"]
    if some_path == SP_DATA_PATH or some_path == TSNE_SP_DATA_PATH:
        names = []
    offset = 1
    properties = np.ones(95)
    properties[0] = 0
    masking_value = 2
    model_lines_all_seeds = header + ""

    for some_seed in seed_list:
        set_seed(some_seed)

        for test_number in range(1, N_FOLDS_FIRST + 1):
            file_df = pd.read_csv(
                "../seeds/seed_"
                + str(some_seed)
                + "/similarity/"
                + "test_fold_"
                + str(test_number)
                + ".csv"
            )
            file = open(predictions_name(some_path, test_number), "r")
            lines = file.readlines()
            predictions = eval(lines[0])
            labels = eval(lines[1])
            file.close()

            for i in range(len(predictions)):
                model_lines_all_seeds += (
                    file_df["sequence"][i]
                    + ";"
                    + str(file_df["label"][i])
                    + ";"
                    + str(predictions[i])
                    + ";"
                    + str(some_seed)
                    + ";"
                    + PATH_TO_NAME[some_path]
                    + ";\n"
                )

    file = open(
        "../seeds/all_seeds/predictions_all_seeds_model_"
        + PATH_TO_EXTENSION[some_path]
        + ".csv",
        "w",
    )
    file.write(model_lines_all_seeds)
    file.close()
