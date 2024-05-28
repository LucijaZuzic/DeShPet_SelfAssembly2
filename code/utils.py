DATA_PATH = "../data/"
AP_DATA_PATH = "../AP_model_data/"
SP_DATA_PATH = "../SP_model_data/"
AP_SP_DATA_PATH = "../AP_SP_data/"
TSNE_SP_DATA_PATH = "../TSNE_SP_model_data/"
TSNE_AP_SP_DATA_PATH = "../TSNE_AP_SP_model_data/"

PATH_TO_NAME = {
    AP_DATA_PATH: "AP",
    SP_DATA_PATH: "SP",
    AP_SP_DATA_PATH: "AP-SP",
    TSNE_SP_DATA_PATH: "TSNE SP",
    TSNE_AP_SP_DATA_PATH: "TSNE AP-SP",
}

PATH_TO_EXTENSION = {
    AP_DATA_PATH: "AP",
    SP_DATA_PATH: "SP",
    AP_SP_DATA_PATH: "AP_SP",
    TSNE_SP_DATA_PATH: "TSNE_SP",
    TSNE_AP_SP_DATA_PATH: "TSNE_AP_SP",
}


def set_seed(x):
    seed_file = open(DATA_PATH + "seed.txt", "w")
    seed_file.write(str(x))
    seed_file.close()


def get_seed():
    seed_file = open(DATA_PATH + "seed.txt", "r")
    line = seed_file.readlines()[0].replace("\n", "")
    seed_file.close()
    return int(line)


def name_plus_test(some_path, test_number):
    return PATH_TO_EXTENSION[some_path] + "_test_" + str(test_number)


def basic_dir(some_path, test_number):
    return (
        "../seeds/seed_"
        + str(get_seed())
        + some_path.replace("..", "")
        + name_plus_test(some_path, test_number)
        + "/"
    )


def final_dir(some_path):
    return "../final" + some_path.replace("..", "")


def basic_path(some_path, test_number):
    return (
        basic_dir(some_path, test_number) + name_plus_test(some_path, test_number) + "_"
    )


def final_path(some_path):
    return final_dir(some_path) + PATH_TO_EXTENSION[some_path] + "_"


def params_plus_fold(params_nr, fold_nr):
    return "_params_" + str(params_nr) + "_fold_" + str(fold_nr)


def h5_and_png(some_path, test_number, params_nr, fold_nr):
    start = (
        basic_path(some_path, test_number)
        + "rnn"
        + params_plus_fold(params_nr, fold_nr)
    )
    return start + ".h5", start + ".png"


def final_h5_and_png(some_path, test_number):
    start = basic_path(some_path, test_number) + "rnn"
    return start + ".h5", start + ".png"


def seed_h5_and_png(some_path):
    start = final_path(some_path) + "rnn"
    return start + ".h5", start + ".png"


def percent_grade_names(some_path, test_number):
    start = basic_path(some_path, test_number)
    end = ".csv"
    return start + "percentage" + end, start + "grade" + end


def hist_names(some_path, test_number):
    start = basic_path(some_path, test_number) + "hist_"
    end = "SA.png"
    return start + end, start + "N" + end


def ROC_name(some_path, test_number):
    return basic_path(some_path, test_number) + "ROC.png"


def PR_name(some_path, test_number):
    return basic_path(some_path, test_number) + "PR.png"


def seed_loss_txt_name(some_path):
    return final_path(some_path) + "loss.txt"


def seed_acc_txt_name(some_path):
    return final_path(some_path) + "acc.txt"


def seed_loss_png_name(some_path):
    return final_path(some_path) + "loss.png"


def seed_acc_png_name(some_path):
    return final_path(some_path) + "acc.png"


def seed_res_name(some_path):
    return final_path(some_path) + "acc.txt"


def final_loss_name(some_path, test_number):
    return basic_path(some_path, test_number) + "loss.png"


def final_acc_name(some_path, test_number):
    return basic_path(some_path, test_number) + "results_accuracy_loss.png"


def loss_name(some_path, test_number, params_nr, fold_nr):
    return (
        basic_path(some_path, test_number)
        + "loss"
        + params_plus_fold(params_nr, fold_nr)
        + ".png"
    )


def acc_name(some_path, test_number, params_nr, fold_nr):
    return (
        basic_path(some_path, test_number)
        + "acc"
        + params_plus_fold(params_nr, fold_nr)
        + ".png"
    )


def log_name(some_path, test_number):
    return basic_path(some_path, test_number) + "log.txt"


def final_log_name(some_path):
    return final_path(some_path) + "log.txt"


def scatter_name(some_path):
    return final_path(some_path) + "scatter_plot_hex.png"

def scatter_name_long(some_path):
    return final_path(some_path) + "scatter_plot_long.png"

def results_name(some_path, test_number):
    return basic_path(some_path, test_number) + "results.txt"


def predictions_name(some_path, test_number):
    return basic_path(some_path, test_number) + "predictions.txt"


def predictions_thr_name(some_path, test_number, params_nr, fold_nr):
    return (
        basic_path(some_path, test_number)
        + "predictions"
        + params_plus_fold(params_nr, fold_nr)
        + ".txt"
    )

def predictions_longest_name(some_path):
    return final_path(some_path) + "predictions_longest.txt"

def predictions_hex_name(some_path):
    return final_path(some_path) + "predictions_hex.txt"


def history_name(some_path, test_number, params_nr, fold_nr):
    start = basic_path(some_path, test_number)
    end = params_plus_fold(params_nr, fold_nr) + ".txt"
    return (
        start + "acc" + end,
        start + "val_acc" + end,
        start + "loss" + end,
        start + "val_loss" + end,
    )


def final_history_name(some_path, test_number):
    start = basic_path(some_path, test_number)
    end = ".txt"
    return start + "acc" + end, start + "loss" + end


def convert_list(model_predictions):
    new_predictions = []
    for j in range(len(model_predictions)):
        new_predictions.append(model_predictions[j][0])
    return new_predictions


def scale(AP_dictionary, offset=1):
    data = [AP_dictionary[key] for key in AP_dictionary]

    # Determine min and max AP scores.
    min_val = min(data)
    max_val = max(data)

    # Scale AP scores to range [- offset, offset].
    for key in AP_dictionary:
        AP_dictionary[key] = (AP_dictionary[key] - min_val) / (
            max_val - min_val
        ) * 2 * offset - offset


def split_amino_acids(sequence, amino_acids_AP_scores):
    ap_list = []

    # Replace each amino acid in the sequence with a corresponding AP score.
    for letter in sequence:
        ap_list.append(amino_acids_AP_scores[letter])

    return ap_list


def split_dipeptides(sequence, dipeptides_AP_scores):
    ap_list = []

    # Replace each dipeptide in the sequence with a corresponding AP score.
    for i in range(len(sequence) - 1):
        ap_list.append(dipeptides_AP_scores[sequence[i : i + 2]])

    return ap_list


def padding(array, len_to_pad, value_to_pad):
    new_array = [value_to_pad for i in range(len_to_pad)]
    for val_index in range(len(array)):
        if val_index < len(new_array):
            new_array[val_index] = array[val_index]
    return new_array


def split_tripeptides(sequence, tripeptides_AP_scores):
    ap_list = []

    # Replace each tripeptide in the sequence with a corresponding AP score.
    for i in range(len(sequence) - 2):
        ap_list.append(tripeptides_AP_scores[sequence[i : i + 3]])

    return ap_list
