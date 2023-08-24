import tensorflow as tf
import numpy as np
from custom_plots import (
    merge_type_params,
    merge_type_test_number,
    plt_model,
    plt_model_final,
    decorate_stats,
    decorate_stats_avg,
    decorate_stats_final,
)
from utils import (
    final_h5_and_png,
    h5_and_png,
    results_name,
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
)
from model_predict import generate_predictions
from reshape import reshape

import new_model

LEARNING_RATE_SET = 0.01
MAX_DROPOUT = 0.5
MAX_BATCH_SIZE = 600


# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def return_callbacks(model_file, metric):
    callbacks = [
        # Save the best model (the one with the lowest validation loss).
        tf.keras.callbacks.ModelCheckpoint(
            model_file, save_best_only=True, monitor=metric, mode="min"
        ),
        tf.keras.callbacks.LearningRateScheduler(scheduler),
    ]
    return callbacks


def data_and_labels_from_indices(all_data, all_labels, indices):
    data = []
    labels = []

    for i in indices:
        data.append(all_data[i])
        labels.append(all_labels[i])

    return data, labels


def common_final_train(
    metric,
    model,
    model_picture,
    model_file,
    best_batch_size,
    epochs,
    train_and_validation_data,
    train_and_validation_labels,
    val_data=[],
    val_labels=[],
):
    # Save graphical representation of the model to a file.
    tf.keras.utils.plot_model(model, to_file=model_picture, show_shapes=True)

    # Print model summary.
    model.summary()

    callbacks = return_callbacks(model_file, metric)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_SET)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    count_len_SA = 0
    count_len_NSA = 0
    for label in train_and_validation_labels:
        if label == 1:
            count_len_SA += 1
        else:
            count_len_NSA += 1
    factor_NSA = count_len_SA / count_len_NSA

    # Train the model.
    # After model training, the `history` variable will contain important parameters for each epoch, such as
    # train loss, train accuracy, learning rate, and so on.
    if len(val_data) != 0:
        return model.fit(
            train_and_validation_data,
            train_and_validation_labels,
            validation_data=[val_data, val_labels],
            class_weight={0: factor_NSA, 1: 1.0},
            epochs=epochs,
            batch_size=best_batch_size,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        return model.fit(
            train_and_validation_data,
            train_and_validation_labels,
            class_weight={0: factor_NSA, 1: 1.0},
            epochs=epochs,
            batch_size=best_batch_size,
            callbacks=callbacks,
            verbose=1,
        )


def basic_training(
    used_path,
    test_number,
    train_and_validation_data,
    train_and_validation_labels,
    kfold_second,
    epochs,
    factor_NSA,
    test_data,
    test_labels,
    names=["AP"],
    offset=1,
    properties=[],
    masking_value=2,
    thr=0.5,
):
    params_nr = 0
    min_val_loss = 1000

    hyperparameter_conv = [5]
    hyperparameter_numcells = [32, 48, 64]
    hyperparameter_kernel_size = [4, 6, 8]
    hyperparameter_lstm = [5]
    hyperparameter_dense = [15]
    hyperparameter_lambda = [0.0]
    hyperparameter_dropout = [MAX_DROPOUT]
    hyperparameter_batch_size = [MAX_BATCH_SIZE]

    if used_path == AP_DATA_PATH:
        hyperparameter_kernel_size = [4]

    best_conv = 0
    best_numcells = 0
    best_kernel = 0
    best_lstm = 0
    best_dense = 0
    best_lambda = 0
    best_dropout = 0
    best_batch_size = 0

    indices = []
    for train_data_indices, validation_data_indices in kfold_second.split(
        train_and_validation_data, train_and_validation_labels
    ):
        indices.append([train_data_indices, validation_data_indices])

    for conv in hyperparameter_conv:
        for numcells in hyperparameter_numcells:
            for kernel in hyperparameter_kernel_size:
                for lstm_NEW in hyperparameter_lstm:
                    for dense_NEW in hyperparameter_dense:
                        for my_lambda in hyperparameter_lambda:
                            for dropout in hyperparameter_dropout:
                                for batch in hyperparameter_batch_size:
                                    params_nr += 1
                                    fold_nr = 0
                                    history_val_loss = []
                                    history_val_acc = []
                                    history_loss = []
                                    history_acc = []

                                    lstm = conv
                                    dense = numcells * 2

                                    for pair in indices:
                                        train_data_indices = pair[0]

                                        validation_data_indices = pair[1]

                                        fold_nr += 1

                                        # Convert train indices to train data and train labels
                                        (
                                            train_data,
                                            train_labels,
                                        ) = data_and_labels_from_indices(
                                            train_and_validation_data,
                                            train_and_validation_labels,
                                            train_data_indices,
                                        )

                                        train_data, train_labels = reshape(
                                            used_path, train_data, train_labels, names
                                        )

                                        # Convert validation indices to validation data and validation labels
                                        (
                                            val_data,
                                            val_labels,
                                        ) = data_and_labels_from_indices(
                                            train_and_validation_data,
                                            train_and_validation_labels,
                                            validation_data_indices,
                                        )

                                        val_data, val_labels = reshape(
                                            used_path, val_data, val_labels, names
                                        )

                                        # Save model to correct file based on number of fold
                                        model_file, model_picture = h5_and_png(
                                            used_path, test_number, params_nr, fold_nr
                                        )

                                        # Choose correct model and instantiate model
                                        if used_path == AP_DATA_PATH:
                                            model = new_model.only_amino_di_tri_model(
                                                3 * len(names),
                                                lstm1=lstm,
                                                lstm2=lstm,
                                                dense=dense,
                                                dropout=dropout,
                                                lambda2=my_lambda,
                                                mask_value=masking_value,
                                            )

                                        if (
                                            used_path == SP_DATA_PATH
                                            or used_path == TSNE_SP_DATA_PATH
                                        ):
                                            model = new_model.create_seq_model(
                                                input_shape=np.shape(train_data[0]),
                                                conv1_filters=conv,
                                                conv2_filters=conv,
                                                conv_kernel_size=kernel,
                                                num_cells=numcells,
                                                dropout=dropout,
                                                mask_value=masking_value,
                                            )

                                        if (
                                            used_path == AP_SP_DATA_PATH
                                            or used_path == TSNE_AP_SP_DATA_PATH
                                        ):
                                            model = new_model.amino_di_tri_model(
                                                3 * len(names),
                                                input_shape=np.shape(
                                                    train_data[3 * len(names)][0]
                                                ),
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

                                        history = common_final_train(
                                            "val_loss",
                                            model,
                                            model_picture,
                                            model_file,
                                            batch,
                                            epochs,
                                            train_data,
                                            train_labels,
                                            val_data,
                                            val_labels,
                                        )

                                        history_val_loss += history.history["val_loss"]
                                        history_val_acc += history.history[
                                            "val_accuracy"
                                        ]
                                        history_loss += history.history["loss"]
                                        history_acc += history.history["accuracy"]

                                        # Output accuracy, validation accuracy, loss and validation loss for all models
                                        other_output = open(
                                            results_name(used_path, test_number),
                                            "a",
                                            encoding="utf-8",
                                        )

                                        if used_path == AP_DATA_PATH:
                                            other_output.write(
                                                "%s: lstm: %d dense: %d lambda: %.2f dropout: %.2f batch_size: %d"
                                                % (
                                                    merge_type_params(
                                                        used_path,
                                                        fold_nr,
                                                        params_nr,
                                                        test_number,
                                                    ),
                                                    lstm,
                                                    dense,
                                                    my_lambda,
                                                    dropout,
                                                    batch,
                                                )
                                            )

                                        if (
                                            used_path == SP_DATA_PATH
                                            or used_path == TSNE_SP_DATA_PATH
                                        ):
                                            other_output.write(
                                                "%s: conv: %d num_cells: %d kernel_size: %d dropout: %.2f batch_size: %d"
                                                % (
                                                    merge_type_params(
                                                        used_path,
                                                        fold_nr,
                                                        params_nr,
                                                        test_number,
                                                    ),
                                                    conv,
                                                    numcells,
                                                    kernel,
                                                    dropout,
                                                    batch,
                                                )
                                            )

                                        if (
                                            used_path == AP_SP_DATA_PATH
                                            or used_path == TSNE_AP_SP_DATA_PATH
                                        ):
                                            other_output.write(
                                                "%s: conv: %d num_cells: %d kernel_size: %d lstm: %d dense: %d lambda: %.2f dropout: %.2f batch_size: %d"
                                                % (
                                                    merge_type_params(
                                                        used_path,
                                                        fold_nr,
                                                        params_nr,
                                                        test_number,
                                                    ),
                                                    conv,
                                                    numcells,
                                                    kernel,
                                                    lstm,
                                                    dense,
                                                    my_lambda,
                                                    dropout,
                                                    batch,
                                                )
                                            )

                                        other_output.write("\n")
                                        other_output.close()

                                        # Output accuracy, validation accuracy, loss and validation loss for all models
                                        decorate_stats(
                                            used_path,
                                            test_number,
                                            history,
                                            params_nr,
                                            fold_nr,
                                        )

                                        # Plot the history
                                        plt_model(
                                            used_path,
                                            test_number,
                                            history,
                                            params_nr,
                                            fold_nr,
                                        )

                                    decorate_stats_avg(
                                        used_path,
                                        test_number,
                                        history_acc,
                                        history_val_acc,
                                        history_loss,
                                        history_val_loss,
                                        params_nr,
                                    )
                                    avg_val_loss = np.mean(history_val_loss)

                                    if avg_val_loss < min_val_loss:
                                        min_val_loss = avg_val_loss
                                        best_conv = conv
                                        best_numcells = numcells
                                        best_kernel = kernel
                                        best_lstm = lstm
                                        best_dense = dense
                                        best_lambda = my_lambda
                                        best_dropout = dropout
                                        best_batch_size = batch

    other_output = open(results_name(used_path, test_number), "a", encoding="utf-8")

    if used_path == AP_DATA_PATH:
        other_output.write(
            "%s Best params: lstm: %d dense: %d lambda: %.2f dropout: %.2f batch_size: %d"
            % (
                merge_type_test_number(used_path, test_number),
                best_lstm,
                best_dense,
                best_lambda,
                best_dropout,
                best_batch_size,
            )
        )

    if used_path == SP_DATA_PATH or used_path == TSNE_SP_DATA_PATH:
        other_output.write(
            "%s Best params: conv: %d num_cells: %d kernel_size: %d dropout: %.2f batch_size: %d"
            % (
                merge_type_test_number(used_path, test_number),
                best_conv,
                best_numcells,
                best_kernel,
                best_dropout,
                best_batch_size,
            )
        )

    if used_path == AP_SP_DATA_PATH or used_path == TSNE_AP_SP_DATA_PATH:
        other_output.write(
            "%s Best params: conv: %d num_cells: %d kernel_size: %d lstm: %d dense: %d lambda: %.2f dropout: %.2f batch_size: %d"
            % (
                merge_type_test_number(used_path, test_number),
                best_conv,
                best_numcells,
                best_kernel,
                best_lstm,
                best_dense,
                best_lambda,
                best_dropout,
                best_batch_size,
            )
        )

    other_output.write("\n")
    other_output.close()

    model_file, model_picture = final_h5_and_png(used_path, test_number)

    train_and_validation_data, train_and_validation_labels = reshape(
        used_path, train_and_validation_data, train_and_validation_labels, names
    )

    # Choose correct model and instantiate model
    if used_path == AP_DATA_PATH:
        model = new_model.only_amino_di_tri_model(
            3 * len(names),
            lstm1=best_lstm,
            lstm2=best_lstm,
            dense=best_dense,
            dropout=best_dropout,
            lambda2=best_lambda,
            mask_value=masking_value,
        )
    if used_path == SP_DATA_PATH or used_path == TSNE_SP_DATA_PATH:
        model = new_model.create_seq_model(
            input_shape=np.shape(train_and_validation_data[0]),
            conv1_filters=best_conv,
            conv2_filters=best_conv,
            conv_kernel_size=best_kernel,
            num_cells=best_numcells,
            dropout=best_dropout,
            mask_value=masking_value,
        )
    if used_path == AP_SP_DATA_PATH or used_path == TSNE_AP_SP_DATA_PATH:
        model = new_model.amino_di_tri_model(
            3 * len(names),
            input_shape=np.shape(train_and_validation_data[3 * len(names)][0]),
            conv=best_conv,
            numcells=best_numcells,
            kernel_size=best_kernel,
            lstm1=best_lstm,
            lstm2=best_lstm,
            dense=best_dense,
            dropout=best_dropout,
            lambda2=best_lambda,
            mask_value=masking_value,
        )

    history = common_final_train(
        "loss",
        model,
        model_picture,
        model_file,
        best_batch_size,
        epochs,
        train_and_validation_data,
        train_and_validation_labels,
    )

    # Output accuracy, validation accuracy, loss and validation loss for all models
    decorate_stats_final(used_path, test_number, history)

    # Plot the history
    plt_model_final(used_path, test_number, history)

    generate_predictions(
        model,
        test_number,
        used_path,
        test_data,
        test_labels,
        MAX_BATCH_SIZE,
        names,
        offset,
        properties,
        masking_value,
        thr,
    )
