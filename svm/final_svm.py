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
from sklearn.calibration import CalibratedClassifierCV

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

def final_svm(train_path, val_path, seg_feat_path, trial, feat_name, test_set_path):
#in this function, we would receive path to MFCC numpy array from different trials,
#apply svm, and produce frame, segment, speaker level prediction
#then apply the trial model on the test set,
#store the test set prediction in a NPY file

    X_train=np.array([])
    y_train=[]
    X_test=np.array([])
    y_test=[]

    with open(train_path,'r') as reader:
        for i in reader:
            f=i.rstrip('\n')
            start1='adrso'
            end1='_seg'
            key=f[f.find(start1)+len(start1):f.rfind(end1)]
            start2='_seg'
            end2='.npy'
            val=f[f.find(start2)+len(start2):f.rfind(end2)]
            
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
                

    with open(val_path,'r') as reader:
        for i in reader:
            f=i.rstrip('\n')
            start1='adrso'
            end1='_seg'
            key=f[f.find(start1)+len(start1):f.rfind(end1)]
            start2='_seg'
            end2='.npy'
            val=f[f.find(start2)+len(start2):f.rfind(end2)]

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
                    
    y_train=np.vstack(y_train)
    y_test=np.vstack(y_test)
    print("X_train.shape: ", X_train.shape)
    print("y_train.shape: ", y_train.shape)
    print("X_test.shape: ", X_test.shape)
    print("y_test.shape: ", y_test.shape)

    classnames=['cn','ad']

    all_ad={} #key: speaker #value: 'seg' list
    all_cn={}

    #store all_ad and all_cn
    with open(seg_feat_path,'r') as reader:
        for i in reader:
            f=i.rstrip('\n')

            start1='adrso'
            end1='_seg'
            key=f[f.find(start1)+len(start1):f.rfind(end1)]
            start2='_seg'
            end2='.npy'
            val=f[f.find(start2)+len(start2):f.rfind(end2)]

            if '/ad/' in i:
                all_ad.setdefault(key,[])
                all_ad[key].append(val)
                
            elif '/cn/' in i:
                all_cn.setdefault(key,[])
                all_cn[key].append(val)


    #model
    svm=LinearSVC(C=0.001,random_state=42,max_iter=1000,verbose=True)
    clf = CalibratedClassifierCV(svm)
    clf.fit(X_train, y_train[:,2].astype(int))
    y_proba = clf.predict_proba(X_test)
    
    # probability logits in y_proba ([x,y],[]), which one represents 0 and which for 1
    ypred=[]
    for i in y_proba:
        if i[0]<i[1]:
            ypred.append(1)
        else:
            ypred.append(0)
            
#====================validation set=========================
    #frame level
    print("frame level:")
    real=y_test[:,2].astype(int)
    classifier_metrics(real,ypred,name="frame_SVM")
    title='frame_'+feat_name+'_'+trial
    draw_confusion_matrix(real, ypred, classnames, title)


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
    title='segment_'+feat_name+'_'+trial
    draw_confusion_matrix(real, pred, classnames, title)


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
    title='speaker_'+feat_name+'_'+trial
    draw_confusion_matrix(real1, pred1, classnames, title)


#===============for test set==================#
    data=np.array([])
    with open(test_set_path,'r') as reader:
        for i in reader:
            f=i.rstrip('\n')
            X_temp=np.load(f)
            X_temp=X_temp.T

            if data.size==0:
                data=X_temp
            else:
                data=np.vstack((data,X_temp))


    test_pred=clf.predict_proba(data)
    title='/home/intern/summer_2021/ctrl_files/test_pred/' + trial+'_'+feat_name+'.npy'
    np.save(title,test_pred)



def final_test(test_set_path,trial1,trial2,trial3, out_path):
# this function is to get the averaging prediction of test set
# we read in predictions from 3 trials, take the average on frame level
# then apply same method to obtain speaker level prediction
    
    #read 3 trials of test set prediction, take average, got the test_pred
    X1=np.load(trial1)
    X2=np.load(trial2)
    X3=np.load(trial3)
    
    proba=np.mean([X1,X2,X3],axis=0)
    
    test_pred=[]
    for i in proba:
        if i[0]<i[1]:
            test_pred.append(1)
        else:
            test_pred.append(0)
    

    #read label
    label=[]
    with open(test_set_path,'r') as reader:
        for i in reader:
            f=i.rstrip('\n')
            X_temp=np.load(f)
            X_temp=X_temp.T
            
            start1='adrsdt'
            end1='_seg'
            key=f[f.find(start1)+len(start1):f.rfind(end1)]
            start2='seg'
            end2='.npy'
            val=f[f.find(start2)+len(start2):f.rfind(end2)]
            
            seg=[]
            seg.append(key)
            seg.append(val)
            y_temp=[seg]*X_temp.shape[0]
            label.append(y_temp)

    label=np.vstack(label)

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

    w = csv.writer(open(out_path, "w"))

    for key, val in spk_pred.items():
        print("key:",key,";value:",value)
        w.writerow([key, val])




#mfcc
final_svm('/home/intern/summer_2021/ctrl_files/trial_1/train_feat.list', '/home/intern/summer_2021/ctrl_files/trial_1/val_feat.list', '/home/intern/summer_2021/database/seg_feat.list', 'trial1', 'mfcc', '/home/intern/summer_2021/code/extract_feats/mfcc/test/seg_mfcc.list')

final_svm('/home/intern/summer_2021/ctrl_files/trial_2/train_feat.list', '/home/intern/summer_2021/ctrl_files/trial_2/val_feat.list', '/home/intern/summer_2021/database/seg_feat.list', 'trial2', 'mfcc', '/home/intern/summer_2021/code/extract_feats/mfcc/test/seg_mfcc.list')

final_svm('/home/intern/summer_2021/ctrl_files/trial_3/train_feat.list', '/home/intern/summer_2021/ctrl_files/trial_3/val_feat.list', '/home/intern/summer_2021/database/seg_feat.list', 'trial3', 'mfcc', '/home/intern/summer_2021/code/extract_feats/mfcc/test/seg_mfcc.list')

final_test('/home/intern/summer_2021/code/extract_feats/mfcc/test/seg_mfcc.list', '/home/intern/summer_2021/ctrl_files/test_pred/trial1_mfcc.npy',
    '/home/intern/summer_2021/ctrl_files/test_pred/trial2_mfcc.npy',
    '/home/intern/summer_2021/ctrl_files/test_pred/trial3_mfcc.npy',
    "/home/intern/summer_2021/ctrl_files/test_pred/final_result_mfcc.csv")


