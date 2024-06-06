import pandas as pd
import numpy as np

model_list_old = ['Transformer', 'RNN', 'LSTM', 'Bi-LSTM', 'MLP']
model_list_new = ["AP", "SP", "AP-SP", "t-SNE SP", "t-SNE AP-SP"]

def print_latex_format(model_list, path_to_file):
    pd_file = pd.read_csv(path_to_file)
    rows_string = "Metric"
    for model in model_list:
        rows_string += " & " + model
    rows_string += "\\\\ \\hline\n"

    if "AP" not in model_list or "Spearman" not in pd_file["Metric"]:
        metric_ord = list(range(len(pd_file["Metric"])))
    else:
        metric_ord = [2, 7, 8, 9, 3, 10, 11, 12, 4, 5, 6, 0, 1]

    for ix in metric_ord:
        rows_string_one = pd_file["Metric"][ix]
        max_row = -1
        max_part = "be"
        for model in model_list:
            round_val = 3
            if "Acc" in pd_file["Metric"][ix]:
                round_val = 1
            part = str(np.round(pd_file[model][ix], round_val))
            if "Acc" in pd_file["Metric"][ix]:
                part += "\\%"
            rows_string_one += " & " + part
            if pd_file[model][ix] > max_row:
                max_row = pd_file[model][ix]
                max_part = part
        rows_string_one += " \\\\ \\hline\n"
        rows_string += rows_string_one.replace(max_part, "\\textbf{" + max_part + "}")
    print(rows_string)

print_latex_format(model_list_new, "review/short/3_24.csv")
print_latex_format(model_list_new, "review_20/short/3_24.csv")
print_latex_format(model_list_new, "review_6000/short/5_5.csv")
print_latex_format(model_list_new, "review_62000/short/5_10.csv")