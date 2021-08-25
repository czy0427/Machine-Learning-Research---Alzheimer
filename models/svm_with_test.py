import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import random
random.seed(42)
import numpy as np
import itertools

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC

def draw_confusion_matrix(y, yhat, classes, name):
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
    path='/home/intern/summer_2021/database/plots/' + name + '.png'
    plt.savefig(path)

def classifier_metrics(ytest, ypred, name="", average='binary'):
    print("Accuracy score for %s: %f" %(name, accuracy_score(ytest, ypred)))
    print("Recall score for %s: %f" %(name, recall_score(ytest,ypred, average=average)))
    print("Precision score for %s: %f" %(name, precision_score(ytest, ypred, average=average)))
    print("F-1 score for %s: %f" %(name, f1_score(ytest, ypred, average=average)))

def plot_roc(ytest,decision_function, name=""):
    falsepr=dict()
    truepr=dict()
    rocauc=dict()
    falsepr, truepr, thresholds = roc_curve(ytest, decision_function)
    rocauc=auc(falsepr, truepr)
    lw=2
    plt.plot(falsepr, truepr, color='darkorange', lw=lw, label='ROC curve (area=%0.4f)' %rocauc)
    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate');
    plt.title('%s ROC curve' %name); plt.legend(loc='lower right')
    path='/home/intern/databse/plots/roc_'+name
    plt.savefig(path)

X_train=np.array([])
y_train=[]
X_test=np.array([])
y_test=[]

with open('/home/intern/summer_2021/ctrl_files/trial_1/train_feat.list','r') as reader:
    for i in reader:
        f=i.rstrip('\n')
        key=f[47:50]
        val=f[54:-4]
        X_temp=np.load(f)
        X_temp=X_temp.T
        
        if X_train.size==0:
            X_train=X_temp
        else:
            X_train=np.vstack((X_train,X_temp))

        if '/ad/' in i:
            seg=[]
            seg.append(key)
            seg.append(val)
            seg.append(1)
            y_temp=[seg]*X_temp.shape[0]
            y_train.append(y_temp)
                
        elif '/cn/' in i:
            seg=[]
            seg.append(key)
            seg.append(val)
            seg.append(0)
            y_temp=[seg]*X_temp.shape[0]
            y_train.append(y_temp)
            
'''
with open('/home/intern/summer_2021/ctrl_files/trial_1/val_feat.list','r') as reader:
    for i in reader:
        f=i.rstrip('\n')
        key=f[47:50]
        val=f[54:-4]
        X_temp=np.load(f)
        X_temp=X_temp.T
        
        if X_test.size==0:
            X_test=X_temp
        else:
            X_test=np.vstack((X_test,X_temp))

        if '/ad/' in i:
            seg=[]
            seg.append(key)
            seg.append(val)
            seg.append(1)
            y_temp=[seg]*X_temp.shape[0]
            y_test.append(y_temp)
                
        elif '/cn/' in i:
            seg=[]
            seg.append(key)
            seg.append(val)
            seg.append(0)
            y_temp=[seg]*X_temp.shape[0]
            y_test.append(y_temp)
'''               
y_train=np.vstack(y_train)
'''
classnames=['cn','ad']


all_ad={} #key: speaker #value: 'seg' list
all_cn={}

#store all_ad and all_cn
with open('/home/intern/summer_2021/database/seg_feat.list','r') as reader:
    for i in reader:
        if '/ad/' in i:
            filename=i.rstrip('\n')
            key=filename[35:38]
            all_ad.setdefault(key,[])
            all_ad[key].append(filename[42:-4])
            
        elif '/cn/' in i:
            filename=i.rstrip('\n')
            key=filename[35:38]
            all_cn.setdefault(key,[])
            all_cn[key].append(filename[42:-4])
'''

#model
lsvc=LinearSVC(C=0.001,random_state=42,max_iter=5000,verbose=True)
lsvc.fit(X_train, y_train[:,2].astype(int))
'''
ypred=lsvc.predict(X_test)

#frame level
print("frame level:")
real=y_test[:,2].astype(int)
classifier_metrics(real,ypred,name="frame_SVM")
draw_confusion_matrix(real, ypred, classnames, 'frame_level_svm0')


#segment level
print("segment level:")
d={}
for index,i in enumerate(y_test):
    temp=i[0]+","+i[1]
    d.setdefault(temp,[])
    d[temp].append(ypred[index])

pred=[]
real=[]
for key,value in d.items():
    result=1
    if value.count(1)<value.count(0):
        result=0
    pred.append(result)
    
    k=key.split(",")
    
    if k[0] in all_ad: #has_key removed from Python3
        real.append(1)
    else:
        real.append(0)

classifier_metrics(real,pred,name="segment_SVM")
draw_confusion_matrix(real, pred, classnames, 'segment_level_svm0')


#speaker level
print("speaker level:")
d1={}
for index,key in enumerate(d):
    temp=key.split(",")[0]

    d1.setdefault(temp,[])
    d1[temp].append(pred[index])

        
pred1=[]
real1=[]
for key,value in d1.items():
    result=1
    if value.count(1)<value.count(0):
        result=0
    pred1.append(result)
    
    if key in all_ad: #has_key removed from Python3
        real1.append(1)
    else:
        real1.append(0)
classifier_metrics(real1,pred1,name="speaker_SVM")
draw_confusion_matrix(real1, pred1, classnames, 'speaker_level_svm0')
'''

#===============for test set==================#
data=np.array([])
label=[]
with open('/home/intern/summer_2021/code/extract_feats/mfcc/test/seg_mfcc.list','r') as reader:
    for i in reader:
        f=i.rstrip('\n')
        start1='adrsdt'
        end1='_seg'
        key=f[f.find(start1)+len(start1):f.rfind(end1)]
        start2='seg'
        end2='.npy'
        val=f[f.find(start2)+len(start2):f.rfind(end2)]
        X_temp=np.load(f)
        X_temp=X_temp.T
        
        seg=[]
        seg.append(key)
        seg.append(val)
        y_temp=[seg]*X_temp.shape[0]
        label.append(y_temp)
            
        if data.size==0:
            data=X_temp
        else:
            data=np.vstack((data,X_temp))

label=np.vstack(label)

test_pred=lsvc.predict(data)

#to segment
test_seg_dict={}
for index,i in enumerate(label):
    temp=i[0]+","+i[1]
    test_seg_dict.setdefault(temp,[])
    test_seg_dict[temp].append(test_pred[index])

seg_pred=[]
for key,value in test_seg_dict.items():
    result=1
    if value.count(1)<value.count(0):
        result=0
    seg_pred.append(result)
print("seg_pred:",seg_pred)
#to_speaker
test_spk_dict={}
for index,key in enumerate(test_seg_dict):
    temp=key.split(",")[0]

    test_spk_dict.setdefault(temp,[])
    test_spk_dict[temp].append(seg_pred[index])

        
spk_pred={}
for key,value in test_spk_dict.items():
    result=1
    if value.count(1)<value.count(0):
        result=0
    spk_pred[key]=result
print(spk_pred[key])
print("spk:",spk_pred)

#write dictionary to csv (https://pythonspot.com/save-a-dictionary-to-a-file/)
import csv

w = csv.writer(open("/home/intern/summer_2021/ctrl_files/trial_1/trial1_model_result.csv", "w"))

for key, val in spk_pred.items():
    print("key:",key,";value:",value)
    w.writerow([key, val])
