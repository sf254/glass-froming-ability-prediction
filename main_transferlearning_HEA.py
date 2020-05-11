# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:13:22 2020

@author: fs
""" 
import pickle
import pandas as pd
import numpy as np 

import aux_fun1
import cnn_extractor_f as cf
import model_evaluation_utils_V1 as meu

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


[x0,dataY]=pickle.load(open('gao_data.txt', 'rb'))#原始数据集

x_im=np.array([aux_fun1.image(i) for i in x0])#map to 2D image & to np
dataY=np.array(dataY)

#choose one extractor from CNN_PTR_best0 to CNN_PTR_best9
dataX = cf.cnn_extractor(model = cf.cnn_struture0(),\
                         pathWb = "CNN_PTR_best7.h5",\
                         x0 = x_im, path_dense = '')



features = dataX
pca = PCA(n_components=10, whiten=True)#, whiten=True
features_pca = pca.fit_transform(features)
# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_pca.shape[1])
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


score_sum = []
score_mean_best = 0
kfold = StratifiedKFold(n_splits=5, random_state=7,shuffle=True).split(features_pca,dataY)#dataX
randomforest = RandomForestClassifier(random_state=7, n_jobs=-1,n_estimators=200,class_weight='balanced')
score=[]
for k, (train, test) in enumerate(kfold):
    print('\n\n 5-fold cross-validation: No.'+str(k+1))
    randomforest.fit(dataX[train], dataY[train])
    yte_pred=randomforest.predict(dataX[test])
    meu.display_model_performance_metrics(true_labels=dataY[test], 
                                      predicted_labels=yte_pred, 
                                      classes=list(set(dataY[test])))
    score.append(meu.get_metrics(true_labels=dataY[test], predicted_labels=yte_pred)) 
score = pd.DataFrame(score,columns=['Accuracy','Precision','Recall','F1 Score'])
#print(score_sum) 

print('\n\n\nAverage score under 5-fold cross-validation')
print(score.mean()) 

  









