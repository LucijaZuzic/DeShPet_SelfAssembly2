from utils import (
    scatter_name,
    PATH_TO_NAME,
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
)
import csv
import pandas as pd

path_list = [
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
]

sheet_name_and_data = dict()
for some_path in path_list:
    hex_file = scatter_name(some_path).replace(".png", "") + "_fixed.csv"
    df = pd.read_csv(hex_file) 
    sheet_name_and_data[PATH_TO_NAME[some_path]] = df
writer = pd.ExcelWriter("my_merged_hex.xlsx", engine = 'openpyxl')
for some_path in path_list:
    sheet_name_and_data[PATH_TO_NAME[some_path]].to_excel(writer, sheet_name = PATH_TO_NAME[some_path], index = False)
writer.close()