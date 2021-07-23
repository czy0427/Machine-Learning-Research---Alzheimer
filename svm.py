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
    path='/home/intern/database/plots/' + name + '.png'
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

def helper(dictionary, key, cat_string, val, label_list, data_array):
    for value in dictionary[key]:
            f='/home/intern/summer_2021/database/mfcc/' + cat_string+'adrso'+key+'_seg'+value+'.npy'
            X_temp=np.load(f)
            X_temp=X_temp.T
            #modify here
            X_temp=X_temp.tolist()
    
            seg=[]
            seg.append(key)
            seg.append(value)
            seg.append(val)
    
            #y_temp=[seg]*X_temp.shape[0]
            y_temp=[seg]*len(X_temp)
        
            label_list.append(y_temp)
            data_array.append(X_temp)
            '''
            if data_array.size==0:
                data_array=X_temp
            else:
                data_array=np.vstack((data_array,X_temp))
            '''

classnames=['cn','ad']

#X_train=np.array([])
data_train=[]
label_train=[]
#X_test=np.array([])
data_test=[]
label_test=[]

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
            
#create training set and testing set
for i, key in enumerate(all_ad):
    if i<60:
        helper(all_ad, key, 'ad/', 1, label_train, data_train)

    else:
        helper(all_ad, key, 'ad/', 1, label_test, data_test)
        
for i, key in enumerate(all_cn):
    if i<51:
        helper(all_cn, key, 'cn/', 0, label_train, data_train)

    else:
        helper(all_cn, key, 'cn/', 0, label_test, data_test)

X_train=np.vstack(data_train)
X_test=np.vstack(data_test)

y_train=np.vstack(label_train)
y_test=np.vstack(label_test)

#model
lsvc=LinearSVC(C=0.001,random_state=42,max_iter=1000,verbose=True)
lsvc.fit(X_train, y_train[:,2].astype(int))
ypred=lsvc.predict(X_test)

#frame level
real=y_test[:,2].astype(int)
classifier_metrics(real,ypred,name="frame_SVM")
draw_confusion_matrix(real, ypred, classnames, 'frame_level_svm')


#segment level
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
draw_confusion_matrix(real, pred, classnames, 'segment_level_svm')


#speaker level
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
draw_confusion_matrix(real, pred, classnames, 'speaker_level_svm')
