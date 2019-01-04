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

from tkinter import *
from tkinter import ttk


class Application(Frame):

    def __init__(self, master):
        super(Application, self).__init__(master)
        self.build_model()
        self.grid()
        self.create_gui()


    def build_model(self):

        ## Format data
        print("Formatting data...")
        col_names = ['ct', 'ucsi', 'ucsh', 'ma', 'bc', 'secs', 'bn', 'nn', 'mi', 'cl'];
        self.data = pd.read_csv('./datasets/bcw_clean.csv', header=None, names=col_names)
        self.data = self.data.replace('?', 0)
        self.data = self.data.apply(pd.to_numeric)
        self.data = shuffle(self.data)
        
        # Target classes should be numbered from 0 to n-1
        self.data['cl'] = self.data['cl'].replace(2, 0)
        self.data['cl'] = self.data['cl'].replace(4, 1)
        
        X = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values
        y = keras.utils.to_categorical(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2)
        
        print("Building network...")
        self.model = Sequential()
        dim = X_train.shape[1]
        
        ## Layer 1
        self.model.add(Dense(9, input_dim = dim, activation='relu'))
        self.model.add(Dropout(0.25))
        
        ## Layer 2
        self.model.add(Dense(9, activation='relu'))
        self.model.add(Dropout(0.25))
        
        ## Layer 3
        self.model.add(Dense(9, activation='relu'))
        self.model.add(Dropout(0.25))
        
        ## Output layer
        self.model.add(Dense(2))
        self.model.add(Activation('sigmoid'))
        self.model.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics = ['accuracy'])
        
        ## Train the model
        print("Training model...")
        self.model.fit(X_train, y_train, batch_size = 32, epochs = 40, verbose = 0, validation_data = (X_test, y_test))
        

    def create_gui(self):
    
        row = 0

        Label(self, text = "Wprowadź poniższe informacje").grid(row = row, columnspan = 2); row += 1
    
        feature_data = self.data.iloc[:, :-1]
        class_data = self.data.iloc[:, -1]

        features = list(feature_data)
        classes = class_data.unique(); classes.sort()

        for f in features:
            values = self.data[f].unique()
            values.sort()
            Label(self, text = f).grid(row = row, column = 0)
            ttk.Combobox(self, values = list(values), state = 'readonly').grid(row = row, column = 1)
            row += 1

        Button(self, text = "Diagnozuj", command = self.do_diagnosis).grid(row = row, columnspan = 2); row += 1

        Label(self).grid(row = row); row += 1   # empty row

        Label(self, text = 'Wartości wyjściowe neuronów:').grid(row = row, columnspan = 2); row += 1
        self.neuron_output_fields = {}

        for c in classes:
            Label(self, text = c).grid(row = row, column = 0)
            self.neuron_output_fields[c] = Entry(self, state = 'disabled')
            self.neuron_output_fields[c].grid(row = row, column = 1)
            row += 1 

        Label(self).grid(row = row); row += 1   # empty row

        Label(self, text = 'Diagnoza:').grid(row = row, column = 0)
        self.diagnosis = Entry(self, state = 'disabled')              
        self.diagnosis.grid(row = row, column = 1)


    def do_diagnosis(self):
        # TODO
        instance = np.array([[1,1,1,1,1,1,1,1,1]])
        #neuron_outputs = self.model.predict(instance)[0]
        prediction = self.model.predict_classes(instance)[0]
        self.diagnosis['state'] = 'normal'
        self.diagnosis.insert(END, prediction)
        self.diagnosis['state'] = 'disabled'



## MAIN
root = Tk()
root.title("nn-project")
app = Application(root)
root.mainloop()
