# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 23:07:08 2020

@author: fs
"""

import aux_define_model
import csv
import pickle
from keras.models import load_model
from  keras.callbacks import ModelCheckpoint

#load data
[X_all,Y_all] = pickle.load(open('dataset_RPTR.txt', 'rb')) 

name = 'CNN_RPTR_'
epoch = 2000
best_acc = []
#------------------------------------------------------------------------------
for k in range(10):
    #--------------------------------------------------------------------------
    i_tr=[i for i in range(len(Y_all))]
    i_te=[i for i in range(len(Y_all))]
    i_te=i_te[2*k::20]+i_te[1+2*k::20]
    for i in i_te:
        i_tr.remove(i)

    x_train=X_all[i_tr]
    y_train=Y_all[i_tr]
    x_test=X_all[i_te]
    y_test=Y_all[i_te]
    
    name_best = name +'best'+str(k)+'.h5'
    name_last = name +'last'+str(k)+'.h5'
    
    callback_lists = [ModelCheckpoint(filepath=name_best, monitor='val_acc',verbose=1,\
                      save_best_only='True',mode='auto',period=1)]
    
    model = aux_define_model.CNNmodel_PTR()
    
    history=model.fit(x_train, y_train, epochs=epoch, batch_size=64,\
                  validation_data=(x_test, y_test),callbacks=callback_lists)
    
    model.save(filepath = name_last) 
    model = load_model(name_best)
    #Evaluate the model with the metrics we defined earlier
    loss_te, accuracy_te = model.evaluate(x_test, y_test)
    loss_tr, accuracy_tr = model.evaluate(x_train, y_train)

    
    best_acc.append([k,loss_tr, loss_te,accuracy_tr, accuracy_te])
    
    with open(name + '.csv',"a",newline ='') as csvfile: 
        writer = csv.writer(csvfile) 
        writer.writerow([k,loss_tr, loss_te,accuracy_tr, accuracy_te])
        









          
        
