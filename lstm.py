# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:24:39 2019

@author: Gaurav Bothra
"""

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

def lstm_model(input_dim, output_dim, return_sequences):
    model = Sequential()
    model.add(LSTM(input_shape=(None, input_dim), units=output_dim, return_sequences = return_sequences))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))
    return model