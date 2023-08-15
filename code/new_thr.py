import pandas as pd 
import numpy as np 
from model_predict import model_predict
from merge_data import merge_data  
from sklearn.model_selection import StratifiedKFold
from load_data import load_data_SA
from automate_training import data_and_labels_from_indices, MAX_BATCH_SIZE
from utils import set_seed, predictions_thr_name, h5_and_png, AP_DATA_PATH, SP_DATA_PATH, AP_SP_DATA_PATH, TSNE_SP_DATA_PATH, TSNE_AP_SP_DATA_PATH, DATA_PATH

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276] 
path_list = [AP_DATA_PATH, SP_DATA_PATH, AP_SP_DATA_PATH, TSNE_SP_DATA_PATH, TSNE_AP_SP_DATA_PATH]  
path_list = [AP_DATA_PATH]

N_FOLDS_FIRST = 5
N_FOLDS_SECOND = 5
offset = 1
properties = np.ones(95)
properties[0] = 0
masking_value = 2

for some_seed in seed_list:
    set_seed(some_seed)
    for some_path in path_list: 
       		if some_path == AP_DATA_PATH:
       			params_nr = 1
       		if some_path == SP_DATA_PATH:
       			params_nr = 7
       		if some_path == AP_SP_DATA_PATH:
       			params_nr = 1
       		if some_path == TSNE_SP_DATA_PATH:
       			params_nr = 5
       		if some_path == TSNE_AP_SP_DATA_PATH:
       			params_nr = 8
       			
       		# Algorithm settings  
       		names = ["AP"]   
       		if some_path == SP_DATA_PATH or some_path == TSNE_SP_DATA_PATH:
       			names = []     

       		SA_data = np.load(DATA_PATH + 'data_SA_updated.npy', allow_pickle = True).item()

       		SA, NSA = load_data_SA(some_path, SA_data, names, offset, properties, masking_value)
 
       		# Merge SA nad NSA data the train and validation subsets.
       		all_data, all_labels = merge_data(some_path, SA, NSA) 
       		
       		# Define N-fold cross validation test harness for splitting the test data from the train and validation data
       		kfold_first = StratifiedKFold(n_splits = N_FOLDS_FIRST, shuffle = True, random_state = some_seed)
       		# Define N-fold cross validation test harness for splitting the validation from the train data
       		kfold_second = StratifiedKFold(n_splits = N_FOLDS_SECOND, shuffle = True, random_state = some_seed) 
       		
       		test_number = 0
        
       		for train_and_validation_data_indices, test_data_indices in kfold_first.split(all_data, all_labels):
       		       		test_number += 1
			         
       		       		# Convert train and validation indices to train and validation data and train and validation labels
       		       		train_and_validation_data, train_and_validation_labels = data_and_labels_from_indices(all_data, all_labels, train_and_validation_data_indices)
  
       		       		fold_nr = 0
       		       		
       		       		for train_data_indices, validation_data_indices in kfold_second.split(train_and_validation_data, train_and_validation_labels): 
       		       		       		fold_nr += 1
				 
       		       		       		# Convert validation indices to validation data and validation labels
       		       		       		validation_data, validation_labels = data_and_labels_from_indices(train_and_validation_data, train_and_validation_labels, validation_data_indices)
       		       		       		
       		       		       		model_file, model_picture = h5_and_png(some_path, test_number, params_nr, fold_nr)
  
       		       		       		model_predictions = model_predict(some_path, validation_data, validation_labels, model_file, "", MAX_BATCH_SIZE, names)
					
       		       		       		fileopen = open(predictions_thr_name(some_path, test_number, params_nr, fold_nr), "w")
       		       		       		fileopen.write(str(model_predictions))
       		       		       		fileopen.close()
