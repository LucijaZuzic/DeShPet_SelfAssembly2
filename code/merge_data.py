import numpy as np
from utils import (
    AP_DATA_PATH,
    SP_DATA_PATH,
    AP_SP_DATA_PATH,
    TSNE_SP_DATA_PATH,
    TSNE_AP_SP_DATA_PATH,
)


def merge_data(path_used, SA, NSA):
    if path_used == AP_DATA_PATH:
        return merge_data_AP(SA, NSA)
    if path_used == SP_DATA_PATH:
        return merge_data_SP(SA, NSA)
    if path_used == AP_SP_DATA_PATH:
        return merge_data_AP_SP(SA, NSA)
    if path_used == TSNE_SP_DATA_PATH:
        return merge_data_TSNE(SA, NSA)
    if path_used == TSNE_AP_SP_DATA_PATH:
        return merge_data_TSNE(SA, NSA)


def merge_data_AP(SA, NSA):
    # Merge the bins and add labels
    merged_data = []
    for i in SA:
        merged_data.append(i)
    for i in NSA:
        merged_data.append(i)

    merged_labels = np.ones(len(SA) + len(NSA))
    merged_labels[len(SA) :] *= 0
    return merged_data, merged_labels


def merge_data_SP(SA, NSA):
    # Merge the bins and add labels
    merged_data = []
    for i in SA:
        merged_data.append(i)
    for i in NSA:
        merged_data.append(i)
    if len(merged_data) > 0:
        merged_data = np.array(merged_data)
    merged_labels = np.ones(len(SA) + len(NSA))
    merged_labels[len(SA) :] *= 0

    return merged_data, merged_labels


def merge_data_AP_SP(SA, NSA):
    # Merge the bins and add labels
    merged_data = []
    for i in SA:
        merged_data.append(i)
    for i in NSA:
        merged_data.append(i)

    merged_labels = np.ones(len(SA) + len(NSA))
    merged_labels[len(SA) :] *= 0
    return merged_data, merged_labels


def merge_data_TSNE(SA, NSA):
    # Merge the bins and add labels
    merged_data = []
    for i in SA:
        merged_data.append(i)
    for i in NSA:
        merged_data.append(i)

    merged_labels = np.ones(len(SA) + len(NSA))
    merged_labels[len(SA) :] *= 0
    return merged_data, merged_labels
