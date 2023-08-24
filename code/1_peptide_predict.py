import numpy as np
import sys
from automate_training import MAX_BATCH_SIZE
from model_predict import model_predict
from load_data import load_data_SA, MAX_LEN
from merge_data import merge_data
from utils import (
    seed_h5_and_png,
    PATH_TO_NAME,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
)

if len(sys.argv) > 1 and len(sys.argv[1]) <= MAX_LEN:
    dict_peptides = {sys.argv[1]: "1"}
    model = "AP-SP"
    if len(sys.argv) > 2 and sys.argv[2] in PATH_TO_NAME.values():
        model = sys.argv[2]
    some_path = AP_SP_DATA_PATH
    for val in PATH_TO_NAME:
        if PATH_TO_NAME[val] == model:
            some_path = val

    actual_AP = [1]

    seq_example = ""
    for i in range(MAX_LEN):
        seq_example += "A"
    dict_peptides[seq_example] = "1"

    best_batch_size = 600
    best_model = ""
    NUM_TESTS = 5

    offset = 1

    properties = np.ones(95)
    properties[0] = 0
    masking_value = 2

    names = ["AP"]
    if some_path == SP_DATA_PATH or some_path == TSNE_SP_DATA_PATH:
        names = []

    num_props = len(names) * 3
    SA, NSA = load_data_SA(
        some_path, dict_peptides, names, offset, properties, masking_value
    )
    all_data, all_labels = merge_data(some_path, SA, NSA)
    print("encoded")

    best_model_file, best_model_image = seed_h5_and_png(some_path)

    model_predictions_peptides = model_predict(
        some_path,
        all_data,
        all_labels,
        best_model_file,
        best_model,
        MAX_BATCH_SIZE,
        names,
    )[:-1]

    print(model_predictions_peptides)
else:
    if len(sys.argv) <= 1:
        print("No peptide")
    if len(sys.argv[0]) > MAX_LEN:
        print("Peptide too long")
