
from sklearn.model_selection import StratifiedKFold
N_FOLDS_FIRST = 5
import numpy as np
from utils import DATA_PATH
import pandas as pd

# Algorithm settings
N_FOLDS_FIRST = 5
SA_data = np.load(DATA_PATH + "data_SA_updated.npy", allow_pickle=True).item()
sequences = []
labels = []
MAX_LEN = 24

for peptide in SA_data:
    if SA_data[peptide] != "1":
        continue
    if len(peptide) > MAX_LEN or SA_data[peptide] == "-1":
        continue
    sequences.append(peptide)
    labels.append(SA_data[peptide])

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276] 

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

is_in_test = dict()

for n in new:
    is_in_test[n] = []

for some_seed in seed_list: 

    test_number = 0
    for test_number in range(N_FOLDS_FIRST):
        train_name = (
            "../seeds/seed_"
            + str(some_seed)
            + "/similarity/"
            + "train_fold_"
            + str(test_number + 1) + ".csv"
        )
        test_name = (
            "../seeds/seed_"
            + str(some_seed)
            + "/similarity/"
            + "test_fold_"
            + str(test_number + 1) + ".csv"
        )
        file_train = pd.read_csv(train_name)
        file_test = pd.read_csv(test_name)
        test_seqs = list(file_test["sequence"])
        train_seqs = list(file_train["sequence"]) 
        train_test_label = []
        for sq in new:
            if sq in test_seqs:
                train_test_label.append("test")
                is_in_test[sq].append((some_seed, test_number + 1))
            if sq in train_seqs:
                train_test_label.append("train")
        print(some_seed, test_number + 1, train_test_label.count("train"), train_test_label.count("test"))

print(is_in_test)
        