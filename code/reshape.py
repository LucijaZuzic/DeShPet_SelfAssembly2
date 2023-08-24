import numpy as np
from utils import (
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
)


def reshape(path_used, all_data, all_labels, names=["AP"]):
    num_props = 3 * len(names)
    if path_used == AP_DATA_PATH:
        return reshape_AP(all_data, all_labels, num_props)
    if path_used == SP_DATA_PATH:
        return reshape_SP(all_data, all_labels)
    if path_used == AP_SP_DATA_PATH:
        return reshape_AP_SP(all_data, all_labels, num_props)
    if path_used == TSNE_SP_DATA_PATH:
        return reshape_TSNE_SP(all_data, all_labels)
    if path_used == TSNE_AP_SP_DATA_PATH:
        return reshape_TSNE_AP_SP(all_data, all_labels, num_props)


def reshape_AP(all_data, all_labels, num_props=3):
    data = [[] for i in range(len(all_data[0]))]
    labels = []
    for i in range(len(all_data)):
        for j in range(len(all_data[0])):
            data[j].append(all_data[i][j])
        labels.append(all_labels[i])
    new_data = []
    for i in range(len(data)):
        if len(data[i]) > 0 and i < num_props:
            new_data.append(np.array(data[i]))
    labels = np.array(labels)
    return new_data, labels


def reshape_SP(all_data, all_labels):
    data = []
    labels = []
    for i in range(len(all_data)):
        data.append(np.array(all_data[i]))
        labels.append(all_labels[i])
    if len(data) > 0:
        data = np.array(data)
    if len(labels) > 0:
        labels = np.array(labels)
    return data, labels


def reshape_AP_SP(all_data, all_labels, num_props=3):
    data = [[] for i in range(len(all_data[0]))]
    labels = []
    for i in range(len(all_data)):
        for j in range(len(all_data[0])):
            data[j].append(all_data[i][j])
        labels.append(all_labels[i])
    new_data = []
    last_data = []
    for i in range(len(data)):
        if len(data[i]) > 0 and i < num_props:
            new_data.append(np.array(data[i]))
        if len(data[i]) > 0 and i >= num_props:
            last_data.append(np.array(data[i]))
    if len(last_data) > 0:
        last_data = np.array(last_data).transpose(1, 2, 0)
    new_data.append(last_data)
    labels = np.array(labels)
    return new_data, labels


def reshape_TSNE_SP(all_data, all_labels):
    data = []
    labels = []
    for i in range(len(all_data)):
        data.append(np.array(all_data[i]))
        labels.append(all_labels[i])
    if len(data) > 0:
        data = np.array(data)
    if len(labels) > 0:
        labels = np.array(labels)
    return data, labels


def reshape_TSNE_AP_SP(all_data, all_labels, num_props=3):
    data = [[] for i in range(len(all_data[0]))]
    labels = []
    for i in range(len(all_data)):
        for j in range(len(all_data[0])):
            data[j].append(all_data[i][j])
        labels.append(all_labels[i])
    new_data = []
    last_data = []
    for i in range(len(data)):
        if len(data[i]) > 0 and i < num_props:
            new_data.append(np.array(data[i]))
        if len(data[i]) > 0 and i >= num_props:
            last_data.append(np.array(data[i]))
    if len(last_data) > 0:
        last_data = np.array(last_data).transpose(1, 2, 0)
    new_data.append(last_data)
    labels = np.array(labels)
    return new_data, labels
