from utils import (
    PATH_TO_EXTENSION,
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
)
import pandas as pd

path_list = [
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
]

letters = {
    AP_DATA_PATH: "d",
    SP_DATA_PATH: "e",
    AP_SP_DATA_PATH: "f",
    TSNE_SP_DATA_PATH: "g",
    TSNE_AP_SP_DATA_PATH: "h",
}
sheet_name_and_data = dict()
for some_path in path_list:
    hex_file = "../seeds/all_seeds/" + PATH_TO_EXTENSION[some_path] + "_hist_merged_seeds_all.csv"
    df = pd.read_csv(hex_file) 
    sheet_name_and_data["3" + letters[some_path]] = df
for some_path in path_list:
    mode_determine = "a" 
    if some_path == path_list[0]:
        mode_determine = "w"
    writer = pd.ExcelWriter("my_merged_hist.xlsx", engine = 'openpyxl', mode = mode_determine)
    sheet_name_and_data["3" + letters[some_path]].to_excel(writer, sheet_name = "3" + letters[some_path], index = False)
    writer.close()