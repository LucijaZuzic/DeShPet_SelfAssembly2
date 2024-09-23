import numpy as np
from utils import DATA_PATH
import matplotlib.pyplot as plt

SA_data = np.load(DATA_PATH + "data_SA_updated.npy", allow_pickle=True).item()

suma = 0

yes_num = 0
no_num = 0

all_yes = 0
all_no = 0

all_lens = {}
all_len_list = []

all_yes_lens = {}
all_yes_len_list = []

all_no_lens = {}
all_no_len_list = []

lens = {}
len_list = []

yes_lens = {}
yes_len_list = []

no_lens = {}
no_len_list = []

for pep in SA_data:
    peptide = [pep, SA_data[pep]]
    if peptide[1] != "1" and peptide[1] != "0":
        print(peptide[1])
            