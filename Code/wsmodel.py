#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 10:23:40 2022

@author: frederic
"""

from importation import Importation
from preprocessor import Preprocess
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout, Flatten, Layer, Reshape
from tensorflow.keras.models import Model


def create_model_dense(image_width, image_height):
    input_img = Input(shape=(image_width,image_height, 1), name="image")
    dense_0 = Flatten()
    dense_1 = Dense(units = 512, kernel_initializer ='normal', activation ='relu')    
    dense_2 = Dense(units = 256, kernel_initializer ='normal', activation ='relu')     
    output = Dense(units = 1, kernel_initializer ='normal', activation='relu')

    x=dense_0(input_img)
    x=dense_1(x)
    x = dense_2(x)
    outputs_dense=output(x)
    model = Model(inputs = input_img, outputs = outputs_dense, name='Dense_model')
    return model 

def create_model_cnn(image_width, image_height):
    input_img = Input(shape=(image_width,image_height, 1), name="image")
    cnn_1 = Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu')
    cnn_2 = MaxPool2D(pool_size = (2, 2))
    cnn_3 = Dropout(rate = 0.2)
    cnn_4 = Flatten()
    cnn_5 = Dense(units = 128, activation = 'relu')
    cnn_6 = Dense(units = 1, activation='relu')

    x=cnn_1(input_img)
    x=cnn_2(x)
    x=cnn_3(x)
    x=cnn_4(x)
    x=cnn_5(x)
    outputs_cnn=cnn_6(x)

    model = Model(inputs = input_img, outputs = outputs_cnn, name='CNN_model')
    return model

def create_model_lenet(image_width, image_height):
    input_img = Input(shape=(image_width,image_height, 1), name="image")
    # size = Input(name="word_size", shape=(1))
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
    dense_2 = Dense(units=1,
                   # activation='relu',
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




if __name__ == '__main__':
    imp = Importation('database')
    df = imp.get_words().sample(frac=1, random_state=1).reset_index()
    # n_lettres_max=11
    # df = df[df.word_size<=n_lettres_max][['file', 'word_size', 'label']]
    df.word_size = df.word_size.astype('float32')
    print("Nombre d'enregistrements ", df.shape[0])


    df = df[['file', 'word_size', 'label']]
    df.info()
    image_width = 128
    image_height = 32
    batch_size = 64
    img_size = (image_width, image_height)
    prepro = Preprocess(img_size, batch_size=batch_size, gaussianBlur=True)

    df_train, df_valid, df_test = prepro.split_data(df, train_size=0.9)

    x_train, y_train = df_train["file"].values, df_train["word_size"].values
    x_test, y_test = df_test["file"].values, df_test["word_size"].values
    x_valid, y_valid = df_valid["file"].values, df_valid["word_size"].values

    train_ds = prepro.prepare_dataset_word_size(x_train, y_train)
    validation_ds = prepro.prepare_dataset_word_size(x_valid, y_valid)
    test_ds = prepro.prepare_dataset_word_size(x_test, y_test)

    model = create_model_lenet(image_width, image_height)
    model.summary()
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    errors = []

    epochs = 10
    training_history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        verbose=1
    )

    train_loss = training_history.history["loss"]
    val_loss = training_history.history["val_loss"]
    train_acc = training_history.history["mean_absolute_error"]
    val_acc = training_history.history["val_mean_absolute_error"]
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Model loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')

    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('Model accuracy per epoch')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')

# for example in test_ds.take(1):
#   image, word_size = example["image"], example["word_size"]
