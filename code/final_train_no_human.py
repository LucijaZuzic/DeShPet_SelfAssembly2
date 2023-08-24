import numpy as np
import os
import sys
import tensorflow as tf
from load_data import load_data_SA
from automate_training import return_callbacks, MAX_DROPOUT, MAX_BATCH_SIZE
from merge_data import merge_data
from reshape import reshape
import new_model
import matplotlib.pyplot as plt
from utils import (
    final_log_name,
    final_dir,
    seed_h5_and_png,
    seed_loss_txt_name,
    seed_acc_txt_name,
    seed_res_name,
    seed_loss_png_name,
    seed_acc_png_name,
    DATA_PATH,
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
    PATH_TO_NAME,
)

path_list = [
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
]

LEARNING_RATE_SET = 0.01
MAX_DROPOUT = 0.5
MAX_BATCH_SIZE = 600

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

    SA_data = np.load(
        DATA_PATH + "data_SA_no_human_updated.npy", allow_pickle=True
    ).item()

    SA, NSA = load_data_SA(some_path, SA_data, names, offset, properties, masking_value)

    # Calculate weight factor for NSA peptides.
    # In our data, there are more peptides that do exhibit self assembly property than are those that do not. Therefore,
    # during model training, we must adjust weight factors to combat this data imbalance.
    factor_NSA = len(SA) / len(NSA)

    # Merge SA nad NSA data the train and validation subsets.
    all_data, all_labels = merge_data(some_path, SA, NSA)

    # Convert train and validation indices to train and validation data and train and validation labels
    train_data, train_labels = reshape(some_path, all_data, all_labels, names)

    # Python program to check if a path exists
    # if it doesn’t exist we create one
    if not os.path.exists(final_dir(some_path).replace("final", "final_no_human")):
        os.makedirs(final_dir(some_path).replace("final", "final_no_human"))

    # Write output to file
    sys.stdout = open(
        final_log_name(some_path).replace("final", "final_no_human"),
        "w",
        encoding="utf-8",
    )

    # Save model to correct file based on number of fold
    model_file, model_picture = seed_h5_and_png(some_path)
    model_file = model_file.replace("final", "final_no_human")
    model_picture = model_picture.replace("final", "final_no_human")

    conv = 5
    lstm = 5
    my_lambda = 0.0
    dropout = MAX_DROPOUT
    batch_size = MAX_BATCH_SIZE
    masking_value = 2

    if some_path == AP_DATA_PATH:
        dense = 64
    if some_path == SP_DATA_PATH:
        numcells = 64
        kernel = 4
    if some_path == AP_SP_DATA_PATH:
        numcells = 32
        kernel = 4
        dense = 64
    if some_path == TSNE_SP_DATA_PATH:
        numcells = 48
        kernel = 6
    if some_path == TSNE_AP_SP_DATA_PATH:
        numcells = 64
        kernel = 6
        dense = 128

    # Choose correct model and instantiate model
    if some_path == AP_DATA_PATH:
        model = new_model.only_amino_di_tri_model(
            3 * len(names),
            lstm1=lstm,
            lstm2=lstm,
            dense=dense,
            dropout=dropout,
            lambda2=my_lambda,
            mask_value=masking_value,
        )

    if some_path == SP_DATA_PATH or some_path == TSNE_SP_DATA_PATH:
        model = new_model.create_seq_model(
            input_shape=np.shape(train_data[0]),
            conv1_filters=conv,
            conv2_filters=conv,
            conv_kernel_size=kernel,
            num_cells=numcells,
            dropout=dropout,
            mask_value=masking_value,
        )

    if some_path == AP_SP_DATA_PATH or some_path == TSNE_AP_SP_DATA_PATH:
        model = new_model.amino_di_tri_model(
            3 * len(names),
            input_shape=np.shape(train_data[3 * len(names)][0]),
            conv=conv,
            numcells=numcells,
            kernel_size=kernel,
            lstm1=lstm,
            lstm2=lstm,
            dense=dense,
            dropout=dropout,
            lambda2=my_lambda,
            mask_value=masking_value,
        )

    # Save graphical representation of the model to a file.
    tf.keras.utils.plot_model(model, to_file=model_picture, show_shapes=True)

    # Print model summary.
    model.summary()

    callbacks = return_callbacks(model_file, "loss")

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_SET)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    count_len_SA = 0
    count_len_NSA = 0
    for label in train_labels:
        if label == 1:
            count_len_SA += 1
        else:
            count_len_NSA += 1
    factor_NSA = count_len_SA / count_len_NSA

    history = model.fit(
        train_data,
        train_labels,
        class_weight={0: factor_NSA, 1: 1.0},
        epochs=EPOCHS,
        batch_size=600,
        callbacks=callbacks,
        verbose=1,
    )

    accuracy = history.history["accuracy"]
    loss = history.history["loss"]

    # Output accuracy, validation accuracy, loss and validation loss for all models
    accuracy_max = np.max(accuracy)
    loss_min = np.min(loss)

    other_output = open(
        seed_res_name(some_path).replace("final", "final_no_human"),
        "w",
        encoding="utf-8",
    )
    other_output.write(
        "Maximum accuracy = %.12f%% Minimal loss = %.12f"
        % (accuracy_max * 100, loss_min)
    )
    other_output.write("\n")
    other_output.write(
        "Accuracy = %.12f%% (%.12f%%) Loss = %.12f (%.12f)"
        % (np.mean(accuracy) * 100, np.std(accuracy) * 100, np.mean(loss), np.std(loss))
    )
    other_output.write("\n")
    other_output.close()

    other_output = open(
        seed_acc_txt_name(some_path).replace("final", "final_no_human"),
        "w",
        encoding="utf-8",
    )
    other_output.write(str(accuracy))
    other_output.write("\n")
    other_output.close()

    other_output = open(
        seed_loss_txt_name(some_path).replace("final", "final_no_human"),
        "w",
        encoding="utf-8",
    )
    other_output.write(str(loss))
    other_output.write("\n")
    other_output.close()

    # Plot the history

    # Summarize history for accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.title("Final model no human " + PATH_TO_NAME[some_path] + "\n Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig(
        seed_acc_png_name(some_path).replace("final", "final_no_human"),
        bbox_inches="tight",
    )
    plt.close()
    # Summarize history for loss
    plt.figure()
    plt.plot(history.history["loss"], label="Loss")
    plt.title("Final model no human " + PATH_TO_NAME[some_path] + "\n Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig(
        seed_loss_png_name(some_path).replace("final", "final_no_human"),
        bbox_inches="tight",
    )
    plt.close()

    # Close output file
    sys.stdout.close()
