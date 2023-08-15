import pandas as pd
import numpy as np 
from scipy import stats
import sklearn
import matplotlib.pyplot as plt
from model_predict import model_predict
from load_data import load_data_SA
from merge_data import merge_data  
from automate_training import MAX_BATCH_SIZE
from utils import set_seed, predictions_hex_name, seed_h5_and_png, scatter_name, PATH_TO_NAME, DATA_PATH, AP_DATA_PATH, SP_DATA_PATH, AP_SP_DATA_PATH, TSNE_SP_DATA_PATH, TSNE_AP_SP_DATA_PATH
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    auc,
    f1_score,
)
from custom_plots import my_accuracy_calculate, weird_division, convert_to_binary
 
plt.rcParams.update({'font.size': 22})
PRthr = {AP_DATA_PATH: 0.2581173828, SP_DATA_PATH: 0.31960518800000004, AP_SP_DATA_PATH: 0.3839948796,
        TSNE_SP_DATA_PATH: 0.30602415759999996, TSNE_AP_SP_DATA_PATH: 0.34321978799999997}
ROCthr = {AP_DATA_PATH: 0.5024869916000001, SP_DATA_PATH: 0.5674178616000001, AP_SP_DATA_PATH: 0.5708274524,
        TSNE_SP_DATA_PATH: 0.566148754, TSNE_AP_SP_DATA_PATH: 0.5588111376}
         
def returnGMEAN(actual, pred):
    tn = 0
    tp = 0
    apo = 0
    ane = 0
    for i in range(len(pred)):
        a = actual[i]
        p = pred[i]
        if a == 1:
            apo += 1
        else:
            ane += 1
        if p == a:
            if a == 1:
                tp += 1
            else:
                tn += 1
    
    return np.sqrt(tp / apo * tn / ane)

def read_ROC(test_labels, model_predictions, lines_dict, oldPR, oldROC):   
    # Get false positive rate and true positive rate.
    fpr, tpr, thresholds = roc_curve(test_labels, model_predictions) 

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ix = np.argmax(gmeans) 

    # Get recall and precision.
    precision, recall, thresholdsPR = precision_recall_curve(
        test_labels, model_predictions
    ) 

    # Calculate the F1 score for each threshold
    fscore = []
    for i in range(len(precision)):
        fscore.append(
            weird_division(2 * precision[i] * recall[i], precision[i] + recall[i])
        )

    # Locate the index of the largest F1 score
    ixPR = np.argmax(fscore)
 
    model_predictions_binary_thrPR_new = convert_to_binary(model_predictions, oldPR) 
    model_predictions_binary_thrROC_new = convert_to_binary(model_predictions, oldROC)

    model_predictions_binary = convert_to_binary(model_predictions, 0.5)
 
    lines_dict['ROC thr new = '].append(oldROC) 
    lines_dict['ROC AUC = '].append(roc_auc_score(test_labels, model_predictions))
    lines_dict['gmean (0.5) = '].append(returnGMEAN(test_labels, model_predictions_binary)) 
    lines_dict['gmean (PR thr new) = '].append(returnGMEAN(test_labels, model_predictions_binary_thrPR_new))
    lines_dict['gmean (ROC thr new) = '].append(returnGMEAN(test_labels, model_predictions_binary_thrROC_new)) 
    lines_dict['Accuracy (ROC thr new) = '].append(my_accuracy_calculate(test_labels, model_predictions, oldROC))

def read_PR(test_labels, model_predictions, lines_dict, oldPR, oldROC):  
    # Get recall and precision.
    precision, recall, thresholds = precision_recall_curve(
        test_labels, model_predictions
    ) 

    # Calculate the F1 score for each threshold
    fscore = []
    for i in range(len(precision)):
        fscore.append(
            weird_division(2 * precision[i] * recall[i], precision[i] + recall[i])
        )

    # Locate the index of the largest F1 score
    ix = np.argmax(fscore)

    # Get false positive rate and true positive rate.
    fpr, tpr, thresholdsROC = roc_curve(test_labels, model_predictions) 

    # Calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))

    # Locate the index of the largest g-mean
    ixROC = np.argmax(gmeans)  

    model_predictions_binary_thrPR_new = convert_to_binary(model_predictions, oldPR)
    model_predictions_binary_thrROC_new = convert_to_binary(model_predictions, oldROC)
    model_predictions_binary = convert_to_binary(model_predictions, 0.5)
    
    lines_dict['PR thr new = '].append(oldPR)
    lines_dict['PR AUC = '].append(auc(recall, precision))  
    lines_dict['F1 (0.5) = '].append(f1_score(test_labels, model_predictions_binary))
    lines_dict['F1 (PR thr new) = '].append(f1_score(test_labels, model_predictions_binary_thrPR_new))
    lines_dict['F1 (ROC thr new) = '].append(f1_score(test_labels, model_predictions_binary_thrROC_new))
    lines_dict['Accuracy (PR thr new) = '].append(my_accuracy_calculate(test_labels, model_predictions_binary_thrPR_new, oldPR))
    lines_dict['Accuracy (0.5) = '].append(my_accuracy_calculate(test_labels, model_predictions, 0.5))

offset = 1
properties = np.ones(95)
properties[0] = 0
masking_value = 2

df = pd.read_csv(DATA_PATH + "human_AI.csv", sep = ";")

dict_human = {}
lab_human = []
for i in range(len(df["pep"])):
    if df["expert"][i] != "Human":
        continue
    dict_human[df["pep"][i]] = str(df["agg"][i])  
    lab_human.append(int(df["agg"][i]))
    
dict_AI = {}
lab_AI = []
for i in range(len(df["pep"])):
    if df["expert"][i] != "AI":
        continue
    dict_AI[df["pep"][i]] = str(df["agg"][i])  
    lab_AI.append(int(df["agg"][i]))

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
        
seed_list = [305475974, 369953070, 879273778, 965681145, 992391276] 
path_list = [AP_DATA_PATH, SP_DATA_PATH, AP_SP_DATA_PATH, TSNE_SP_DATA_PATH, TSNE_AP_SP_DATA_PATH] 
 
vals_in_lines = [   
'ROC AUC = ',  
'PR AUC = ',  
'gmean (0.5) = ', 'F1 (0.5) = ', 'Accuracy (0.5) = ',  
'ROC thr new = ', 'PR thr new = ', 
'gmean (ROC thr new) = ', 'F1 (ROC thr new) = ', 'Accuracy (ROC thr new) = ', 
'gmean (PR thr new) = ', 'F1 (PR thr new) = ', 'Accuracy (PR thr new) = '  
 ]
 
lines_human = dict()
for val in vals_in_lines:
	lines_human[val] = []
	
lines_AI = dict()
for val in vals_in_lines:
	lines_AI[val] = []
        
lines_all = dict()
for val in vals_in_lines:
	lines_all[val] = []
	
for some_path in path_list: 
	filehuman = open(predictions_hex_name(some_path).replace("final", "final_no_human").replace("hex", "human"), "r")
	pred_human = eval(filehuman.readlines()[0].replace("\n", ""))
	pred_human = pred_human[:-1]
	filehuman.close()
	
	fileAI = open(predictions_hex_name(some_path).replace("final", "final_no_AI").replace("hex", "AI"), "r")
	pred_AI = eval(fileAI.readlines()[0].replace("\n", ""))
	pred_AI = pred_AI[:-1]
	fileAI.close()
	
	pred = []
	lab = []
	
	for i in range(len(lab_human)):
		pred.append(pred_human[i])
		lab.append(lab_human[i])
		
	for i in range(len(lab_AI)):
		pred.append(pred_AI[i])
		lab.append(lab_AI[i])
	  
	read_ROC(lab_human, pred_human, lines_human, PRthr[some_path], ROCthr[some_path])
	read_ROC(lab_AI, pred_AI, lines_AI, PRthr[some_path], ROCthr[some_path])
	read_ROC(lab, pred, lines_all, PRthr[some_path], ROCthr[some_path])
	
	read_PR(lab_human, pred_human, lines_human, PRthr[some_path], ROCthr[some_path]) 
	read_PR(lab_AI, pred_AI, lines_AI, PRthr[some_path], ROCthr[some_path]) 
	read_PR(lab, pred, lines_all, PRthr[some_path], ROCthr[some_path])
	  
for x in lines_human:
	print(x)
	print(np.round(lines_human[x], 3))
	
for x in lines_AI:
	print(x)
	print(np.round(lines_AI[x], 3))
	
for x in lines_all:
	print(x)
	print(np.round(lines_all[x], 3))
