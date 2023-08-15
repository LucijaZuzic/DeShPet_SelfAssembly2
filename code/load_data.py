import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from seqprops import SequentialPropertiesEncoder
from utils import scale, split_amino_acids, split_dipeptides, split_tripeptides, padding, DATA_PATH, AP_DATA_PATH, SP_DATA_PATH, AP_SP_DATA_PATH, TSNE_SP_DATA_PATH, TSNE_AP_SP_DATA_PATH

MAX_LEN = 24

def read_from_npy_SA(SA_data):
    sequences = []
    labels = []
    for peptide in SA_data:
        if len(peptide) > MAX_LEN or SA_data[peptide] == '-1':
            continue
        sequences.append(peptide)
        labels.append(SA_data[peptide])

    return sequences, labels
    
def load_data_AP(name = "AP", offset = 1):
    # Load AP scores. 
    amino_acids_AP = np.load(DATA_PATH + "amino_acids_" + name + ".npy", allow_pickle = True).item()
    dipeptides_AP = np.load(DATA_PATH + "dipeptides_" + name + ".npy", allow_pickle = True).item()
    tripeptides_AP = np.load(DATA_PATH + "tripeptides_" + name + ".npy", allow_pickle = True).item()
    
    # Scale scores to range [-1, 1].
    scale(amino_acids_AP, offset)
    scale(dipeptides_AP, offset)
    scale(tripeptides_AP, offset)

    return amino_acids_AP, dipeptides_AP, tripeptides_AP
    
def load_data_SA(path_used, SA_data, names = ["AP"], offset = 1, properties_to_include = [], masking_value = 2):
    if path_used == AP_DATA_PATH:
    	return load_data_SA_AP(SA_data, names, offset, masking_value)
    if path_used == SP_DATA_PATH:
    	return load_data_SA_SP(SA_data, names, offset, properties_to_include, masking_value)
    if path_used == AP_SP_DATA_PATH:
    	return load_data_SA_AP_SP(SA_data, names, offset, properties_to_include, masking_value)
    if path_used == TSNE_SP_DATA_PATH:
    	return load_data_SA_TSNE(SA_data, names, offset, masking_value)
    if path_used == TSNE_AP_SP_DATA_PATH:  
    	return load_data_SA_TSNE(SA_data, names, offset, masking_value)
    
def load_data_SA_AP(SA_data, names = ["AP"], offset = 1, masking_value = 2):
    sequences, labels = read_from_npy_SA(SA_data)
     
    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    SA = []
    NSA = []
    for index in range(len(sequences)):
        new_props = []
        for name in names:
            amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(name, offset)  
            amino_acids_ap = split_amino_acids(sequences[index], amino_acids_AP)
            dipeptides_ap = split_dipeptides(sequences[index], dipeptides_AP)
            tripeptides_ap = split_tripeptides(sequences[index], tripeptides_AP)
                    
            amino_acids_ap_padded = padding(amino_acids_ap, MAX_LEN, masking_value)
            dipeptides_acids_ap_padded = padding(dipeptides_ap, MAX_LEN, masking_value)
            tripeptides_ap_padded = padding(tripeptides_ap, MAX_LEN, masking_value)  
        
            new_props.append(amino_acids_ap_padded)
            new_props.append(dipeptides_acids_ap_padded)
            new_props.append(tripeptides_ap_padded)  
        
        if labels[index] == "1":
            SA.append(new_props) 
        elif labels[index] == "0":
            NSA.append(new_props) 
            
    return SA, NSA
    
def load_data_SA_SP(SA_data, names = ["AP"], offset = 1, properties_to_include = [], masking_value = 2):
    sequences, labels = read_from_npy_SA(SA_data)
            
    # Encode sequences
    encoder = SequentialPropertiesEncoder(scaler = MinMaxScaler(feature_range=(-offset, offset)))
    encoded_sequences = encoder.encode(sequences)
     
    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    SA = []
    NSA = []
    for index in range(len(sequences)):
        new_props = []
        for name in names:
            amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(name, offset)  
            amino_acids_ap = split_amino_acids(sequences[index], amino_acids_AP)
            dipeptides_ap = split_dipeptides(sequences[index], dipeptides_AP)
            tripeptides_ap = split_tripeptides(sequences[index], tripeptides_AP)
                    
            amino_acids_ap_padded = padding(amino_acids_ap, MAX_LEN, masking_value)
            dipeptides_acids_ap_padded = padding(dipeptides_ap, MAX_LEN, masking_value)
            tripeptides_ap_padded = padding(tripeptides_ap, MAX_LEN, masking_value)  

            new_props.append(amino_acids_ap_padded)
            new_props.append(dipeptides_acids_ap_padded)
            new_props.append(tripeptides_ap_padded)  

        other_props = np.transpose(encoded_sequences[index])  

        for prop_index in range(len(properties_to_include)):
            if prop_index < len(other_props) and properties_to_include[prop_index] == 1:
                array = other_props[prop_index]
                for i in range(len(array)):
                    if i >= len(sequences[index]):
                        array[i] = masking_value
                new_props.append(array)
                 
        new_props = np.transpose(new_props) 

        if labels[index] == "1":
            SA.append(new_props) 
        elif labels[index] == "0":
            NSA.append(new_props) 
    if len(SA) > 0:
        SA = np.array(SA)
    if len(NSA) > 0:
        NSA = np.array(NSA)
    return SA, NSA
    
def load_data_SA_AP_SP(SA_data, names = ["AP"], offset = 1, properties_to_include = [], masking_value = 2):
    sequences, labels = read_from_npy_SA(SA_data)
            
    # Encode sequences
    encoder = SequentialPropertiesEncoder(scaler=MinMaxScaler(feature_range=(-offset, offset)))
    encoded_sequences = encoder.encode(sequences)
     
    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    SA = []
    NSA = []
    for index in range(len(sequences)):
        new_props = []
        for name in names:
            amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(name, offset)  
            amino_acids_ap = split_amino_acids(sequences[index], amino_acids_AP)
            dipeptides_ap = split_dipeptides(sequences[index], dipeptides_AP)
            tripeptides_ap = split_tripeptides(sequences[index], tripeptides_AP)
                    
            amino_acids_ap_padded = padding(amino_acids_ap, MAX_LEN, masking_value)
            dipeptides_acids_ap_padded = padding(dipeptides_ap, MAX_LEN, masking_value)
            tripeptides_ap_padded = padding(tripeptides_ap, MAX_LEN, masking_value)  
        
            new_props.append(amino_acids_ap_padded)
            new_props.append(dipeptides_acids_ap_padded)
            new_props.append(tripeptides_ap_padded) 
        
        other_props = np.transpose(encoded_sequences[index])  

        for prop_index in range(len(properties_to_include)):
            if prop_index < len(other_props) and properties_to_include[prop_index] == 1:
                array = other_props[prop_index]
                for i in range(len(array)):
                    if i >= len(sequences[index]):
                        array[i] = masking_value
                new_props.append(array) 
        
        if labels[index] == "1":
            SA.append(new_props) 
        elif labels[index] == "0":
            NSA.append(new_props) 
            
    return SA, NSA
    
def load_data_SA_TSNE(SA_data, names = ["AP"], offset = 1, masking_value = 2):
    sequences, labels = read_from_npy_SA(SA_data)
     
    # Split peptides in two bins.
    # SA - has self-assembly, NSA - does not have self-assembly.
    # Convert each peptide sequence to three lists of AP scores: amino acid, dipeptide and tripeptide scores.
    SA = []
    NSA = []
    for index in range(len(sequences)):
        new_props = []

        for name in names:
            amino_acids_AP, dipeptides_AP, tripeptides_AP = load_data_AP(name, offset)  
            amino_acids_ap = split_amino_acids(sequences[index], amino_acids_AP)
            dipeptides_ap = split_dipeptides(sequences[index], dipeptides_AP)
            tripeptides_ap = split_tripeptides(sequences[index], tripeptides_AP)
                    
            amino_acids_ap_padded = padding(amino_acids_ap, MAX_LEN, masking_value)
            dipeptides_acids_ap_padded = padding(dipeptides_ap, MAX_LEN, masking_value)
            tripeptides_ap_padded = padding(tripeptides_ap, MAX_LEN, masking_value)  
        
            new_props.append(amino_acids_ap_padded)
            new_props.append(dipeptides_acids_ap_padded)
            new_props.append(tripeptides_ap_padded)  

        feature_dict_1  = np.load(DATA_PATH + "amino_acids_NEW1.npy", allow_pickle = True).item() 
        feature_dict_2  = np.load(DATA_PATH + "amino_acids_NEW2.npy", allow_pickle = True).item() 
        feature_dict_3  = np.load(DATA_PATH + "amino_acids_NEW3.npy", allow_pickle = True).item() 

        feature1 = split_amino_acids(sequences[index], feature_dict_1)
        feature1_padded = padding(feature1, MAX_LEN, masking_value)

        feature2 = split_amino_acids(sequences[index], feature_dict_2)
        feature2_padded = padding(feature2, MAX_LEN, masking_value)

        feature3 = split_amino_acids(sequences[index], feature_dict_3)
        feature3_padded = padding(feature3, MAX_LEN, masking_value)

        new_props.append(feature1_padded)
        new_props.append(feature2_padded)
        new_props.append(feature3_padded) 
        
        if labels[index] == "1":
            SA.append(new_props) 
        elif labels[index] == "0":
            NSA.append(new_props) 
            
    return SA, NSA 
