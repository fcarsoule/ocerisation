#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 9:39:25 2022

@author: frederic
"""



import numpy as np


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout, Flatten, Layer, Reshape
from tensorflow.keras.models import Model


def create_model_dense(image_width, image_height, num_classes):
    #inputs_dense=Input(shape = (height, width,1), name = "Input")
    input_img = Input(shape=(image_width, image_height, 1), name="image")
    dense_0 = Flatten()
    dense_1 = Dense(units = 512, kernel_initializer ='normal', activation ='relu')    
    dense_2 = Dense(units = 256, kernel_initializer ='normal', activation ='relu')     
    output = Dense(units = num_classes, kernel_initializer ='normal', activation='softmax')

    x=dense_0(input_img)
    x=dense_1(x)
    x = dense_2(x)
    outputs_dense=output(x)
    model = Model(inputs = input_img, outputs = outputs_dense, name='Dense_model')
    return model 

def create_model_cnn(width, height, num_classes):
    #inputs_cnn=Input(shape = (height, width,1), name = "Input")
    input_img = Input(shape=(width, height, 1), name="image")
    cnn_1 = Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu')
    cnn_2 = MaxPool2D(pool_size = (2, 2))
    cnn_3 = Dropout(rate = 0.2)
    cnn_4 = Flatten()
    cnn_5 = Dense(units = 128, activation = 'relu')
    cnn_6 = Dense(units = num_classes, activation='softmax')

    x=cnn_1(input_img)
    x=cnn_2(x)
    x=cnn_3(x)
    x=cnn_4(x)
    x=cnn_5(x)
    outputs_cnn=cnn_6(x)

    model = Model(inputs = input_img, outputs = outputs_cnn, name='CNN_model')
    return model

def create_model_lenet(image_width, image_height, num_classes):
    #inputs_img = Input(shape=(height, width, 1), name="image")
    input_img = Input(shape=(image_width,image_height, 1), name="image")
    conv_1 = Conv2D(filters=30,                     # Nombre de filtres
                    kernel_size=(5, 5),            # Dimensions du noyau
                    kernel_initializer="he_normal",
                    padding='valid',               # Mode de Dépassement
                    activation='relu',
                    name="conv1")             # Fonction d'activation
    max_pool_1 = MaxPool2D(pool_size=(2, 2))
    dropout1 = Dropout(rate=0.2)
    conv_2 = Conv2D(filters=50,
                    kernel_size=(3, 3),
                    kernel_initializer="he_normal",
                    padding='valid',
                    activation='relu', name="conv2")
    max_pool_2 = MaxPool2D(pool_size=(2, 2))
    dropout2 = Dropout(rate=0.2)
    flatten = Flatten()

    dense_1 = Dense(units=128,
                    activation='relu', name='dense1')
    dense_2 = Dense(units=num_classes,
                    activation='softmax',
                    name="dense2")


    x = conv_1(input_img)
    x = max_pool_1(x)
    x = dropout1(x)
    x = conv_2(x)
    x = max_pool_2(x)
    x = dropout2(x)
    x = flatten(x)
    x = dense_1(x)
    outputs = dense_2(x)

    model = Model(inputs=input_img, outputs=outputs, name="LeNet")
    return model 
