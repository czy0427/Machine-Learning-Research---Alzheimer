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
from sklearn.decomposition import PCA

def plot_confm(confm, classes, title='Confusion Matrix', cmap_name='Blues'):
    plt.imshow(confm,interpolation='nearest',cmap=plt.get_cmap(cmap_name))
    plt.title(title)
    plt.colorbar()
    tickmarks=np.arange(len(classes))
    plt.xticks(tickmarks, classes)
    plt.yticks(tickmarks, classes)
    
    fmt='d'
    thresh=confm.max()/2.
    for i,j in itertools.product(range(confm.shape[0]),range(confm.shape[1])):
        plt.text(j,i,format(confm[i,j],fmt), 
                horizontalalignment="center",color="white" if confm[i,j]>thresh else "black" )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    path='/home/intern/database/plots/'+ title
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

#store ad and cn data into X, ad as 1 and cn as 0 into y
X=np.array([])
y=[]
with open('/home/intern/database/seg_feat.list','r') as reader:
    for i in reader:
        if '/ad/' in i:
            X_temp=np.load(i.rstrip('\n'))
            X_temp=X_temp.T
            y=np.hstack((y,[1 for j in range(X_temp.shape[0])]))
            if X.size==0:
                X=X_temp
            else:
                X=np.vstack((X,X_temp))
        elif '/cn/' in i:
            X_temp=np.load(i.rstrip('\n'))
            X_temp=X_temp.T
            y=np.hstack((y,[0 for j in range(X_temp.shape[0])]))
            if X.size==0:
                X=X_temp
            else:
                X=np.vstack((X,X_temp))
#y=np.array(y)
print("X dimension:", X.shape)
print("y dimension:", y.shape)

#PCA
#X_pca=PCA(n_components=100).fit_transform(X)

#train test split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#model
classnames=['ad','cn']

lsvc=LinearSVC(C=0.01,random_state=42,max_iter=100000,verbose=True)

lsvc.fit(X_train,y_train)
'''
#if GridSearch:

params={'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
crof=GridSearchCV(lsvc,params,cv=5,scoring='accuracy')
crof.fit(X_train, y_train)

ypred_cv=crof.best_estimator_.predict(X_test)
bestsvm_gamma=crof.best_estimator_.C

print("Grid search results for SVM: ", crof.cv_results_)
print("Best estimator for SVM: ", crof.best_estimator_)
print("Best parameters for SVM: ", crof.best_params_)
print("Best score for SVM: ", crof.best_score_)
print("Best Gamma for SVM: ", bestsvm_gamma)

classifier_metrics(y_test,ypred_cv,name="Best Gamma SVM")
cv_confm=confusion_matrix(y_test,ypred_cv)
plot_confm(cv_confm, classes=classnames,title="Best Gamma SVM Confusion Matrix")
'''

ypred=lsvc.predict(X_test)

classifier_metrics(y_test,ypred,name="testSVM")
cm=confusion_matrix(y_test,ypred)
plot_confm(cm,classes=classnames,title="testSVM Confusion Matrix")
