#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 08:10:21 2022

@author: frederic
"""

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras
import numpy as np
import cv2

class Preprocess():
    def __init__(self, img_size, batch_size=None, gaussianBlur=False, max_len=None, characters=None):
        # np.random.seed(42)
        # tf.random.set_seed(42)
        print('initialisation')
        self.img_size = img_size
        self.max_len = max_len
        self.characters = characters
        self.batch_size = batch_size
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.padding_token = 99
        self.gaussianBlur = gaussianBlur
        if characters :
            self.char_to_num = StringLookup(vocabulary=sorted(self.characters), mask_token=None)
            self.num_to_char = StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
            )
      
    def preprocess_image(self, image_path):
        w, h = self.img_size
        image = tf.py_function(self.load_image, [image_path, h, w], [tf.float32])
        return tf.squeeze(image,axis=0)
        
    def load_image(self, image_path, h, w):
        #w, h = self.img_size
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, 1)
        image = self.distortion_free_resize(image, h, w)
        image = tf.cast(image, tf.float32) / 255.0
        if self.gaussianBlur:
            #edge = cv2.Canny(image.numpy(),100,200)
            image = cv2.GaussianBlur(np.squeeze(image.numpy()),(3,3),0)
            image = tf.convert_to_tensor(np.expand_dims(image, axis=2))
        return image
    
    def distortion_free_resize(self, image, h, w):
        # w, h = self.img_size
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
    
        # Check tha amount of padding needed to be done.
        pad_height = h - tf.shape(image)[0]
        pad_width = w - tf.shape(image)[1]
    
        # Only necessary if you want to do same amount of padding on both sides.
        if pad_height % 2 != 0:
            height = pad_height // 2
            pad_height_top = height + 1
            pad_height_bottom = height
        else:
            pad_height_top = pad_height_bottom = pad_height // 2
    
        if pad_width % 2 != 0:
            width = pad_width // 2
            pad_width_left = width + 1
            pad_width_right = width
        else:
            pad_width_left = pad_width_right = pad_width // 2
    
        image = tf.pad(
            image,
            paddings=[
                [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ],
        )
    
        image = tf.transpose(image, perm=[1, 0, 2])
        image = tf.image.flip_left_right(image)
        return image    
    
    # Images - Labels ************************    
    def prepare_dataset(self, image_paths, labels):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
            self.process_images_labels, num_parallel_calls=self.AUTOTUNE
        )
        return dataset.batch(self.batch_size).cache().prefetch(self.AUTOTUNE)
    
    def process_images_labels(self, image_path, label):
        image = self.preprocess_image(image_path)
        label = self.vectorize_label(label)
        return {"image": image, "label": label}
    
    
    
    #Vectorisation d'un label
    def vectorize_label(self, label):
        #char_to_num, _ = self.map_characters()
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = self.max_len - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=self.padding_token)
        return label
    
    def get_images_labels(self, ds):
        images = []
        labels = []

        for batch in ds:
            images.append(batch["image"])
            labels.append(batch["label"])
        return images, labels
    
    
    # Image - Word size ************************
    def prepare_dataset_word_size(self, image_paths, sizes):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, sizes)).map(
            self.process_images_sizes, num_parallel_calls=self.AUTOTUNE
        )
        return dataset.batch(self.batch_size).cache().prefetch(self.AUTOTUNE)
    
    def process_images_sizes(self, image_path, word_size):
        image = self.preprocess_image(image_path)
        word_size = tf.cast(word_size, tf.float32)
        return image, word_size
        #return {"image": image, "word_size": word_size}
    
    
    def get_images_size(self, ds):
        images = []
        sizes = []

        for batch in ds:
            images.append(batch["image"])
            sizes.append(batch["word_size"])
        return images, sizes
    
    
    # Image - Top 10 ************************
    def prepare_dataset_top10(self, image_paths, label):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label)).map(
            self.process_images_top10, num_parallel_calls=self.AUTOTUNE
        )
        return dataset.batch(self.batch_size).cache().prefetch(self.AUTOTUNE)
    
    def process_images_top10(self, image_path, label):
        image = self.preprocess_image(image_path)

        return image, label
        #return {"image": image, "word_size": word_size}
    
    
    def get_images_top10(self, ds):
        images = []
        sizes = []

        for batch in ds:
            images.append(batch["image"])
            sizes.append(batch["label"])
        return images, sizes
    
    
    #Split data in train - validation - test set
    def split_data(self, df, train_size=0.9, split_valid_test=0.5):
        split = int(len(df) * train_size)
        df_train = df[:split]
        df_test = df[split:]
        split = int(len(df_test) * split_valid_test)
        df_valid = df_test[:split]
        df_test = df_test[split:]
        return df_train, df_valid, df_test
    
    # A utility function to decode the output of the network.
    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search.
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :self.max_len
        ]
        # Iterate over the results and get back the text.
        output_text = []
        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

