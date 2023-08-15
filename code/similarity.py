import numpy as np 
from utils import DATA_PATH 
from sklearn.model_selection import StratifiedKFold
from plot_similarities_seqprops import main_sim, double_sim
from load_data import MAX_LEN
import os

# Algorithm settings 
N_FOLDS_FIRST = 5
SA_data = np.load(DATA_PATH+'data_SA_updated.npy', allow_pickle=True).item()
sequences = []
labels = [] 

for peptide in SA_data:
    if SA_data[peptide] != '1':
        continue  
    if len(peptide) > MAX_LEN or SA_data[peptide] == '-1':
        continue
    sequences.append(peptide)
    labels.append(SA_data[peptide]) 

for peptide in SA_data:
    if SA_data[peptide] == '1':
        continue  
    if len(peptide) > MAX_LEN or SA_data[peptide] == '-1':
        continue
    sequences.append(peptide)
    labels.append(SA_data[peptide]) 

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276]

for some_seed in seed_list:
    SEED = some_seed

    if not os.path.exists("../seeds/seed_" + str(SEED) + "/similarity/"):
        os.makedirs("../seeds/seed_" + str(SEED) + "/similarity/")
    # Define N-fold cross validation test harness for splitting the test data from the train and validation data
    kfold_first = StratifiedKFold(n_splits=N_FOLDS_FIRST, shuffle=True, random_state=SEED)

    test_number = 0
    for train_and_validation_data_indices, test_data_indices in kfold_first.split(sequences, labels):
        test_number += 1

        # Convert train and validation indices to train and validation data and train and validation labels
        train_test_save = "sequence,label\n"
        train_save = "sequence,label\n" 
        for i in train_and_validation_data_indices: 
            train_save += sequences[i] + "," + labels[i] + "\n" 
            train_test_save += sequences[i] + "," + labels[i] + "\n"

        train_name = "../seeds/seed_" + str(SEED) + "/similarity/" + 'train_fold_' + str(test_number)  
        train_csv = train_name + ".csv"
        train_png = train_name + ".png"
        train_output = open(train_csv, "w", encoding="utf-8") 
        train_output.write(train_save)
        train_output.close()
        main_sim(train_csv, train_png)
            
        # Convert test indices to test data and test labels
        test_save = "sequence,label\n"
        test_data = []
        test_labels = [] 
        for i in test_data_indices: 
            test_save += sequences[i] + "," + labels[i] + "\n" 
            train_test_save += sequences[i] + "," + labels[i] + "\n" 

        test_name = "../seeds/seed_" + str(SEED) + "/similarity/" + 'test_fold_' + str(test_number)  
        test_csv = test_name + ".csv"
        test_png = test_name + ".png"
        test_output = open(test_csv, "w", encoding="utf-8") 
        test_output.write(test_save)
        test_output.close()
        main_sim(test_csv, test_png)

        train_test_name = "../seeds/seed_" + str(SEED) + "/similarity/" + 'train_test_fold_' + str(test_number)  
        train_test_csv = train_test_name + ".csv"
        train_test_png = train_test_name + ".png"
        train_test_output = open(train_test_csv, "w", encoding="utf-8") 
        train_test_output.write(train_test_save)
        train_test_output.close() 
        double_sim(train_test_csv, train_test_png, len(train_and_validation_data_indices))
