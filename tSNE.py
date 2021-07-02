import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import csv
#from mpl_toolkits.mplot3d import Axes3D

#reader=csv.reader(open('/home/intern/datatbase/seg_feat.list'),'r')

t_ad=[]
t_cn=[]
with open('/home/intern/database/seg_feat.list','r') as reader:

    for i in reader:
        if '/ad/' in i:
            print("1: ",i)
            X1=np.load(i.rstrip('\n'))
            tsne1=TSNE().fit_transform(X1)
            t_ad.append(tsne1)
            plt.scatter(*zip(*tsne1[:,:2]),c='r')

        elif '/cn/' in i:
            print("2: ",i)
            X2=np.load(i.rstrip('\n'))
            tsne2=TSNE().fit_transform(X2)
            t_cn.append(tsne2)
            plt.scatter(*zip(*tsne2[:,:2]),c='b')

    plt.savefig('/home/intern/database/plots/t_sne')
