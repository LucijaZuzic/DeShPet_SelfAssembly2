import numpy as np
import os
import sys
from sklearn.model_selection import StratifiedKFold
from load_data import load_data_SA
from automate_training import basic_training, data_and_labels_from_indices
from merge_data import merge_data
from utils import (
    set_seed,
    results_name,
    log_name,
    basic_dir,
    DATA_PATH,
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
)

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]
path_list = [
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
]

for some_seed in seed_list:
    set_seed(some_seed)
    for some_path in path_list:
        # Algorithm settings
        N_FOLDS_FIRST = 5
        N_FOLDS_SECOND = 5
        EPOCHS = 70
        names = ["AP"]
        if some_path == SP_DATA_PATH or some_path == TSNE_SP_DATA_PATH:
            names = []
        offset = 1
        properties = np.ones(95)
        properties[0] = 0
        masking_value = 2
        used_thr = 0.5

        SA_data = np.load(DATA_PATH + "data_SA_updated.npy", allow_pickle=True).item()

        SA, NSA = load_data_SA(
            some_path, SA_data, names, offset, properties, masking_value
        )

        # Calculate weight factor for NSA peptides.
        # In our data, there are more peptides that do exhibit self assembly property than are those that do not. Therefore,
        # during model training, we must adjust weight factors to combat this data imbalance.
        factor_NSA = len(SA) / len(NSA)

        # Merge SA nad NSA data the train and validation subsets.
        all_data, all_labels = merge_data(some_path, SA, NSA)

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

            # Convert train and validation indices to train and validation data and train and validation labels
            (
                train_and_validation_data,
                train_and_validation_labels,
            ) = data_and_labels_from_indices(
                all_data, all_labels, train_and_validation_data_indices
            )

            # Convert test indices to test data and test labels
            test_data, test_labels = data_and_labels_from_indices(
                all_data, all_labels, test_data_indices
            )

            # Python program to check if a path exists
            # If it doesn’t exist we create one
            if not os.path.exists(basic_dir(some_path, test_number)):
                os.makedirs(basic_dir(some_path, test_number))

            # Write output to file
            other_output = open(
                results_name(some_path, test_number), "w", encoding="utf-8"
            )
            other_output.write("")
            other_output.close()

            # Write output to file
            sys.stdout = open(log_name(some_path, test_number), "w", encoding="utf-8")

            # Train the ansamble model
            basic_training(
                some_path,
                test_number,
                train_and_validation_data,
                train_and_validation_labels,
                kfold_second,
                EPOCHS,
                factor_NSA,
                test_data,
                test_labels,
                names,
                offset,
                properties,
                masking_value,
                used_thr,
            )

            # Close output file
            sys.stdout.close()
