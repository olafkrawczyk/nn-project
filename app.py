#!/usr/bin/env python3
# coding: utf-8

print("Importing modules...")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras import backend
from keras.utils import to_categorical
from math import floor, ceil
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## Format data
print("Formatting data...")
col_names = ['ct', 'ucsi', 'ucsh', 'ma', 'bc', 'secs', 'bn', 'nn', 'mi', 'cl'];
dataset = pd.read_csv('./datasets/bcw_clean.csv', header=None, names=col_names)
dataset = dataset.replace('?', 0)
dataset = dataset.apply(pd.to_numeric)
dataset = shuffle(dataset)

# Target classes should be numbered from 0 to n-1
dataset['cl'] = dataset['cl'].replace(2, 0)
dataset['cl'] = dataset['cl'].replace(4, 1)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = keras.utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2)

print("Building network...")
model = Sequential()
dim = X_train.shape[1]

## Layer 1
model.add(Dense(9, input_dim = dim, activation='relu'))
model.add(Dropout(0.25))

## Layer 2
model.add(Dense(9, activation='relu'))
model.add(Dropout(0.25))

## Layer 3
model.add(Dense(9, activation='relu'))
model.add(Dropout(0.25))

## Output layer
model.add(Dense(2))
model.add(Activation('sigmoid'))
model.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics = ['accuracy'])

## Train the model
print("Training model...")
model.fit(X_train, y_train, batch_size = 32, epochs = 40, verbose = 0, validation_data = (X_test, y_test))

print("Please input 9 values (from 1 to 10) separated with \",\": ", end='')
i1 = input()
i1 = np.array([i1.split(',')])

print("Activation values: {}".format(model.predict(i1)[0]))
print("Prediction: {}".format(model.predict_classes(i1)[0]))
