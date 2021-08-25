import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import random
random.seed(42)
import numpy as np
import itertools
import model_train_config as config
from pdb import set_trace as bp
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit


def get_train_meta_data(train_spk_file):

    '''
    Takes the path of the train_spk file. 
    Returns train speakers and labels as a list. 
    '''
    train_spk = []
    train_labels=np.array([])
    with open(train_spk_file) as file_in:
        for line in file_in:
            line = line.strip()
            spkid = line.split('/')[1]
            label=1 if line.split('/')[0] == 'ad' else 0
            train_spk.append(spkid)
            train_labels = np.append(train_labels,label)

    return train_spk, train_labels

def get_cv_indices(train_labels,val_split,num_splits):

    '''
    Takes train labels and number of splits as input. 
    Returns indices for balanced splits.
    val_split_indices = [[train_indices,val_indices],[train_indices,val_indices]..]
    '''
    n_samples = train_labels.shape[0]
    val_split_indices =[]
    ad_indices = np.where(train_labels==1)[0]
    cn_indices = np.where(train_labels==0)[0]

    cn_count = len(cn_indices)
    ad_count = len(ad_indices)

    for i in range(num_splits):
        train_indices=np.array([])
        val_indices=np.array([])
        ad_val_indices = np.random.choice(ad_indices,int(val_split*len(ad_indices)),replace=False)
        ad_train_indices = ad_indices[~np.isin(ad_indices,ad_val_indices)]
        cn_val_indices = np.random.choice(cn_indices,int(val_split*len(cn_indices)),replace=False)
        cn_train_indices = cn_indices[~np.isin(cn_indices,cn_val_indices)]

        train_indices = np.concatenate((ad_train_indices,cn_train_indices),axis=0)
        val_indices = np.concatenate((ad_val_indices,cn_val_indices),axis=0)
        temp_indices = [train_indices,val_indices]
        val_split_indices.append(temp_indices)

    return val_split_indices


def diagnosis_ad(train_spk,test_spk,test_labels,feat_name,model,output_folder):

    '''
    TODO: ADD COMMENTS
    '''

    train_spk, train_labels = get_train_meta_data(train_spk)
    val_split_indices = get_cv_indices(train_labels,val_split=0.3,num_splits=5)
    model_arr = []
    val_fscore_arr = []
    
    for i in range(num_splits):
        train_indices,val_indices = val_split_indices[i] # code is written till here. 
        X_train,y_train,X_val,y_val = load_train_val_data(train_indices,val_indices)
        model,val_fscore = run_train_val(X_train,y_train,X_val,y_val)
        model_arr.append(model)
        val_fscore_arr.append(val_fscore)

    X_test,y_test_truth = load_test_data(test_spk,test_labels)
    best_model = model_arr[val_fscore_arr.index(max(val_fscore_arr))]
    run_eval(X_test,y_test,feats_name, output_path, model = best_model)



if __name__ == "__main__":

    feat_name = config.acoustic_feats
    model = config.model
    output_folder = config.output_folder


    train_spk_file = config.train_spk_file
    test_spk_file = config.test_spk_file
    test_labels_file = config.test_label_file


    diagnosis_ad(train_spk_file,test_spk_file,test_labels_file,feat_name,model,output_folder)


