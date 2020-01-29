# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:20:48 2020

@author: fs
"""
import re
import pickle
import copy
import numpy as np
from keras.utils import np_utils

gfa_dataset = pickle.load(open('gfa_dataset.txt', 'rb')) 

[property_name_list,property_list,element_name,_]=pickle.load(open('element_property.txt', 'rb'))

Z_row_column = pickle.load(open('Z_row_column.txt', 'rb'))


def PTR(i):#periodical table representation
    #i='4 La$_{66}$Al$_{14}$Cu$_{10}$Ni$_{10}$ [c][15]'
    X= [[[0.0 for ai in range(18)]for aj in range(9)] for ak in range(1) ]
    gfa=re.findall('\[[a-c]?\]',i)[0]
    
    tx1_element=re.findall('[A-Z][a-z]?', i)#[B, Fe, P,No]
    tx2_temp=re.findall('\$_{[0-9.]+}\$', i)#[$_{[50]}$, ] [50 30 20]
    tx2_value=[float(re.findall('[0-9.]+', i_tx2)[0]) for i_tx2 in tx2_temp]
    for j in range(len(tx2_value)):
        index=int(property_list[element_name.index(tx1_element[j])][1])#atomic number Z
        xi=int(Z_row_column[index-1][1])#row num
        xj=int(Z_row_column[index-1][2])#col num
        X[0][xi-1][xj-1]=tx2_value[j]/100.0
    X_BMG=copy.deepcopy(X)
    X_BMG[0][0][8]=1.0 #processing parameter
    
    if gfa=='[c]':
        Y=[0,0]
    if gfa=='[b]': 
        Y=[1,0]
    if gfa=='[a]' :
        Y=[1,1]

    return [X,X_BMG],Y 

#------------------------------------------------------------------------------
#group data
gfa_i=[]
gfa_a=[]
gfa_b=[]
gfa_c=[]
for i in  gfa_dataset:
    tx_gfa=re.findall('\[[a-c]?\]', i)#[B, Fe, P,No]
    gfa_i.extend(tx_gfa)
    if tx_gfa[0]=='[a]':
        gfa_a.append(gfa_dataset.index(i))
    elif tx_gfa[0]=='[b]':
        gfa_b.append(gfa_dataset.index(i)) 
    else:
        gfa_c.append(gfa_dataset.index(i))
        
gfa_data_form=[]
gfa_data_form_b=[]
#------------------------------------------------------------------------------
#map raw data to 2-D image using PTR
for i in gfa_a:
    x,y = PTR(gfa_dataset[i])
    gfa_data_form=gfa_data_form+x
    gfa_data_form_b=gfa_data_form_b+y
for i in gfa_c:
    x,y = PTR(gfa_dataset[i])
    gfa_data_form=gfa_data_form+x
    gfa_data_form_b=gfa_data_form_b+y 
for i in gfa_b:
    x,y = PTR(gfa_dataset[i])
    gfa_data_form=gfa_data_form+[x[0]]
    gfa_data_form_b=gfa_data_form_b+[y[0]]  

X_all = np.array(gfa_data_form).reshape(-1, 1,9, 18) 
Y_all = np_utils.to_categorical(np.array(gfa_data_form_b), num_classes=2)#one-hot coding

#write out dataset after process
pickle.dump([X_all,Y_all],open('dataset_PTR.txt', 'wb')) 
#------------------------------------------------------------------------------
 
