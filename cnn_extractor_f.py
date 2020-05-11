# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:55:15 2020

@author: fs
"""
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from keras.models import Model
import pickle

def cnn_struture1():
    #8-8-pol-16-16-pol-32-32-pol-10-1 output
    model = Sequential()
    # Conv layer 1 output shape (32, 28, 28)
    model.add(Convolution2D(batch_input_shape=(None, 1, 9, 18),filters=8,
        kernel_size=3,strides=1, padding='same',
        data_format='channels_first', activation='relu'))
    model.add(Convolution2D(8, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides=2,padding='same',data_format='channels_first',))
    model.add(Convolution2D(16, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))#16
    model.add(Convolution2D(16, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
    model.add(Convolution2D(32, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))
    model.add(Convolution2D(32, 3, strides=1, padding='same', 
                            data_format='channels_first', activation='relu'))
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))#32
    model.add(Flatten(name='dense1'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1,activation='relu'))
    return model
def cnn_struture0():
    #8-pol-16-pol-32-pol-2 output
    model = Sequential()
    model.add(Convolution2D(batch_input_shape=(None, 1, 9, 18),filters=8,
        kernel_size=3,strides=1, padding='same',
        data_format='channels_first', activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides=2,padding='same',data_format='channels_first',))
    model.add(Convolution2D(16, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))#16
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
    model.add(Convolution2D(32, 3, strides=1, padding='same', 
                            data_format='channels_first', activation='relu'))
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))#32
    model.add(Flatten(name='dense1'))
    model.add(Dense(2,activation='softmax',name='output'))
    return model

def cnn_struture11by11():
    #8-pol-16-pol-32-pol-2 output
    model = Sequential()
    model.add(Convolution2D(batch_input_shape=(None, 1, 11, 11),filters=8,
        kernel_size=3,strides=1, padding='same',
        data_format='channels_first', activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides=2,padding='same',data_format='channels_first',))
    model.add(Convolution2D(16, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))#16
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
    model.add(Convolution2D(32, 3, strides=1, padding='same', 
                            data_format='channels_first', activation='relu'))
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))#32
    model.add(Flatten(name='dense1'))
    model.add(Dense(2,activation='softmax',name='output'))
    return model

def cnn_struture7by32():
    model = Sequential()
    # Conv layer 1 output shape (32, 28, 28)
    model.add(Convolution2D(batch_input_shape=(None, 1, 7, 32),filters=8,
        kernel_size=3,strides=1, padding='same',
        data_format='channels_first', activation='relu'))
    model.add(Convolution2D(8, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides=2,padding='same',data_format='channels_first',))
    model.add(Convolution2D(16, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))#16
    model.add(Convolution2D(16, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
    model.add(Convolution2D(32, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))
    model.add(Convolution2D(32, 3, strides=1, padding='same', 
                            data_format='channels_first', activation='relu'))
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))#32
    model.add(Flatten(name='dense1'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1,activation='relu'))
    return model


def cnn_extractor(model,pathWb,x0,path_dense=''): 
    #define cnn structure
    #model = cnn_struture1()
    #load Wb
    model.load_weights(pathWb)
    #已有的model在load权重过后
    #取某一层的输出为输出新建为model，采用函数模型
    dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense1').output)
    #以这个model的预测值作为输出
    dense1_output = dense1_layer_model.predict(x0)
    print('cnn features shape', dense1_output.shape)
    if path_dense != '':
        pickle.dump(dense1_output,open(path_dense, 'wb')) 
    return dense1_output
    #path='C://pyHEA//cnn_Ef_features.txt'
        