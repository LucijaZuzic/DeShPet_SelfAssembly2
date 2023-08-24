import pandas as pd
import numpy as np
from model_predict import model_predict
from load_data import load_data_SA, MAX_LEN
from merge_data import merge_data
from automate_training import MAX_BATCH_SIZE
from utils import (
    predictions_hex_name,
    seed_h5_and_png,
    DATA_PATH,
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
)

df = pd.read_csv(
    DATA_PATH + "41557_2022_1055_MOESM3_ESM_Figure3a_5mer_score_shortMD.csv"
)

dict_hex = {}
for i in df["pep"]:
    dict_hex[i] = "1"

actual_AP = []
for i in df["AP"]:
    actual_AP.append(i)

seq_example = ""
for i in range(MAX_LEN):
    seq_example += "A"
dict_hex[seq_example] = "1"

offset = 1
properties = np.ones(95)
properties[0] = 0
masking_value = 2

path_list = [
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
]

for some_path in path_list:
    # Algorithm settings
    names = ["AP"]
    if some_path == SP_DATA_PATH or some_path == TSNE_SP_DATA_PATH:
        names = []

    SA_AP, NSA_AP = load_data_SA(
        some_path, dict_hex, names, offset, properties, masking_value
    )
    all_data_AP, all_labels_AP = merge_data(some_path, SA_AP, NSA_AP)

    best_model = ""
    model_file, model_picture = seed_h5_and_png(some_path)
    model_predictions = model_predict(
        some_path,
        all_data_AP,
        all_labels_AP,
        model_file,
        best_model,
        MAX_BATCH_SIZE,
        names,
    )

    fileopen = open(predictions_hex_name(some_path), "w")
    fileopen.write(str(model_predictions))
    fileopen.close()
