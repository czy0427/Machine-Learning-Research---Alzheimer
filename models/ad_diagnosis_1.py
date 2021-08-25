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
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import csv

def draw_confusion_matrix(y, yhat, classes, name, output_folder):
    '''
        Draws a confusion matrix for the given target and predictions
        Adapted from scikit-learn and discussion example.
    '''
    plt.cla()
    plt.clf()
    matrix = confusion_matrix(y, yhat)
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    num_classes = len(classes)
    plt.xticks(np.arange(num_classes), classes, rotation=90)
    plt.yticks(np.arange(num_classes), classes)
    
    fmt = 'd'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    path= output_folder + name + '.png'
    plt.savefig(path)

def classifier_metrics(ytest, ypred, name, output_folder,average='binary'):
    print("Accuracy score for %s: %f" %(name, accuracy_score(ytest, ypred)))
    print("Recall score for %s: %f" %(name, recall_score(ytest,ypred, average=average)))
    print("Precision score for %s: %f" %(name, precision_score(ytest, ypred,average=average)))
    print("F-1 score for %s: %f" %(name, f1_score(ytest, ypred,average=average)))
    
    f=open(output_folder+"metrics.txt", "w")
    f.write("Accuracy score for %s: %f" %(name, accuracy_score(ytest, ypred)))
    f.write("Recall score for %s: %f" %(name, recall_score(ytest,ypred)))
    f.write("Precision score for %s: %f" %(name, precision_score(ytest, ypred)))
    f.write("F-1 score for %s: %f" %(name, f1_score(ytest, ypred)))
    f.close()
    #how to write to a file and store in path???
    
    
def get_train_meta_data(train_spk_file):
#"ad/adrso10"
# should feed in all speaker, the
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

def load_train_val_data(train_spk, train_labels, train_indices,val_indices, train_path):

    X_train=np.array([])
    y_train=[]
    X_val=np.array([])
    y_val=[]
    
    for i in train_indices:
        with open(train_path,'r') as reader:
            for j in reader:
                f=j.rstrip('\n')
                spk_part=f[:f.rfind('_seg')]
                if train_spk[i] in spk_part:
                    X_temp=np.load(f)

                    #X.T for mfcc only?
                    #X_temp=X_temp.T
                    
                    y_temp=[train_labels[i]]*X_temp.shape[0]
                    y_train.append(y_temp)
            
                    if X_train.size==0:
                        X_train=X_temp
                    else:
                        X_train=np.vstack((X_train,X_temp))
                print(X_train.shape)

    for i in val_indices:
        with open(train_path,'r') as reader:
            for j in reader:
                f=j.rstrip('\n')
                spk_part=f[:f.rfind('_seg')]
                if train_spk[i] in spk_part:
                    X_temp=np.load(f)
                    
                    #X_temp=X_temp.T
                    
                    y_temp=[train_labels[i]]*X_temp.shape[0]
                    y_val.append(y_temp)
                                
                    if X_val.size==0:
                        X_val=X_temp
                    else:
                        X_val=np.vstack((X_val,X_temp))
                print(X_val.shape)

    #print("y_val", len(y_val)) 
    y_train=np.hstack(y_train)
    y_val=np.hstack(y_val)
    
    return X_train, y_train, X_val, y_val
    #X_train: mfcc array
    #y_train: [[adrso10,1],...] ==> changed to [1,1,0,...], no need the labels attached!

def run_train_val(X_train, y_train, X_val, y_val, model):
    if model=='svm':
        clf=LinearSVC(C=0.001,random_state=42,max_iter=5000,verbose=True)
        
    #clf.fit(X_train, y_train[:,1].astype(int))
    clf.fit(X_train, y_train)
    ypred=clf.predict(X_val)
    val_fscore=f1_score(y_val, ypred)
    return clf, val_fscore
    
def load_test_data(test_spk_file):
    data=np.array([])
    label=[] #speaker tag
    
    with open(test_spk_file,'r') as reader:
        for i in reader:
            f=i.rstrip('\n')
            X_temp=np.load(f)
            
            #X_temp=X_temp.T

            if data.size==0:
                data=X_temp
            else:
                data=np.vstack((data,X_temp))
                
            start1='adrsdt'
            end1='_seg'
            key=f[f.find(start1)+len(start1):f.rfind(end1)]
            
            seg=[]
            seg.append(key)
            y_temp=[seg]*X_temp.shape[0] #simplify?
            label.append(y_temp)
    
    label=np.vstack(label)
    label=np.hstack(label)#change code here???
    
    return data, label
    
def run_eval(X_test,X_tag,feat_name,model_name, output_folder, model, truth_dict):
    pred=model.predict(X_test)
    
    #change code here
    test_dict={}
    for index,i in enumerate(X_tag):
        test_dict.setdefault(i,[])
        test_dict[i].append(pred[index])
        
    test_pred={}
    for key,value in test_dict.items():
        result=1
        if value.count(1)<value.count(0):
            result=0
        test_pred[key]=result
    
    prediction=[]
    actual=[]
    for key, value in test_pred.items():
        prediction.append(value)
        actual.append(truth_dict.get(key))
        
    classifier_metrics(actual,prediction,name="test_"+model_name+"_"+feat_name, output_folder=output_folder)
    classnames=['cn','ad']
    draw_confusion_matrix(actual,prediction,classnames, name="cfmx_"+model_name+"_"+feat_name, output_folder=output_folder)


def diagnosis_ad(train_spk_file,test_spk_file,test_labels,feat_name,model_name,output_folder, train_path, test_path,num_splits):

    '''
    TODO: ADD COMMENTS
    '''

    train_spk, train_labels = get_train_meta_data(train_spk_file)
    val_split_indices = get_cv_indices(train_labels,val_split=0.3,num_splits=num_splits)
    model_arr = []
    val_fscore_arr = []

   # t,v=val_split_indices[0]
   # print("train index:", t)
   # print("val index:", v)
    
    for i in range(num_splits):
        train_indices,val_indices = val_split_indices[i]
        # got indices of label, the corresponding indices of speaker "adrso10"
        X_train,y_train,X_val,y_val = load_train_val_data(train_spk, train_labels, train_indices,val_indices, train_path)
        curr_model,val_fscore = run_train_val(X_train,y_train,X_val,y_val,model_name)
        model_arr.append(curr_model)
        val_fscore_arr.append(val_fscore)

    X_test,X_tag = load_test_data(test_path)
    best_model = model_arr[val_fscore_arr.index(max(val_fscore_arr))]
    
    #ground_truth store into dictionary
    truth_dict={}
    with open(test_labels, mode='r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            if rows[1]=="Control":
                truth_dict[rows[0][6:]]=0
            elif rows[1]=="ProbableAD":
                truth_dict[rows[0][6:]]=1
            else:
                continue
        
    run_eval(X_test,X_tag,feat_name, model_name, output_folder, model = best_model, truth_dict=truth_dict)



if __name__ == "__main__":

    feat_name = config.acoustic_feats
    model_name = config.model_name
    output_folder = config.output_folder


    train_spk_file = config.train_spk_file
    test_spk_file = config.test_spk_file
    test_labels_file = config.test_label_file
    
    #editted
    train_path=config.train_path #the file containing path to npy files for training data
    test_path=config.test_path
    num_splits=config.num_splits

    diagnosis_ad(train_spk_file,test_spk_file,test_labels_file,feat_name,model_name,output_folder, train_path, test_path,num_splits)


