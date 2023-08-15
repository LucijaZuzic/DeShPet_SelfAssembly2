import pandas as pd
import numpy as np 
from model_predict import model_predict
from load_data import load_data_SA
from merge_data import merge_data  
from automate_training import MAX_BATCH_SIZE
from utils import set_seed, predictions_hex_name, seed_h5_and_png, DATA_PATH, AP_DATA_PATH, SP_DATA_PATH, AP_SP_DATA_PATH, TSNE_SP_DATA_PATH, TSNE_AP_SP_DATA_PATH
    
df = pd.read_csv(DATA_PATH + "human_AI.csv", sep = ";")

dict_human = {}
for i in range(len(df["pep"])):
    if df["expert"][i] != "Human":
        continue
    dict_human[df["pep"][i]] = str(df["agg"][i])  
    
dict_AI = {}
for i in range(len(df["pep"])):
    if df["expert"][i] != "AI":
        continue
    dict_AI[df["pep"][i]] = str(df["agg"][i])  

actual_AP_human = []
for i in range(len(df["pep"])):
    if df["expert"][i] != "Human":
        continue
    actual_AP_human.append(df["AP"][i])  
    
actual_AP_AI = []
for i in range(len(df["pep"])):
    if df["expert"][i] != "AI":
        continue
    actual_AP_AI.append(df["AP"][i])  

seq_example = ""
for i in range(24):
    seq_example += "A"
dict_human[seq_example] = "1"
dict_AI[seq_example] = "1"
  
offset = 1  
properties = np.ones(95)
properties[0] = 0
masking_value = 2 

seed_list = [305475974, 369953070, 879273778, 965681145, 992391276] 
path_list = [AP_DATA_PATH, SP_DATA_PATH, AP_SP_DATA_PATH, TSNE_SP_DATA_PATH, TSNE_AP_SP_DATA_PATH] 
 
for some_path in path_list:
	# Algorithm settings  
	names = ["AP"]   
	if some_path == SP_DATA_PATH or some_path == TSNE_SP_DATA_PATH:
		names = []   
     
	SA_human, NSA_human = load_data_SA(some_path, dict_human, names, offset, properties, masking_value)
	all_data_human, all_labels_human = merge_data(some_path, SA_human, NSA_human) 
	
	SA_AI, NSA_AI = load_data_SA(some_path, dict_AI, names, offset, properties, masking_value)
	all_data_AI, all_labels_AI = merge_data(some_path, SA_AI, NSA_AI) 

	best_model = ""
	
	model_file_no_human, model_picture_no_human = seed_h5_and_png(some_path) 
	model_file_no_human = model_file_no_human.replace("final", "final_no_human")
	model_picture_no_human = model_picture_no_human.replace("final", "final_no_human")
	
	model_file_no_AI, model_picture_no_AI = seed_h5_and_png(some_path) 
	model_file_no_AI = model_file_no_AI.replace("final", "final_no_AI")
	model_picture_no_AI = model_picture_no_AI.replace("final", "final_no_AI")
	
	model_predictions_human = model_predict(some_path, all_data_human, all_labels_human, model_file_no_human, best_model, MAX_BATCH_SIZE, names)
	
	model_predictions_AI = model_predict(some_path, all_data_AI, all_labels_AI, model_file_no_AI, best_model, MAX_BATCH_SIZE, names)

	filehuman = open(predictions_hex_name(some_path).replace("final", "final_no_human").replace("hex", "human"), "w")
	filehuman.write(str(model_predictions_human))
	filehuman.close()
	
	fileAI = open(predictions_hex_name(some_path).replace("final", "final_no_AI").replace("hex", "AI"), "w")
	fileAI.write(str(model_predictions_AI))
	fileAI.close()
