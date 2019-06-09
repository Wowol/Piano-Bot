from __future__ import print_function
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.layers import Dropout, TimeDistributed
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model as lm

import numpy as np
import random
import sys
import io
from midi import Midi


def create_model(size, unique_notes, optimizer=None, hidden_size=128, use_cudnnlstm=False):
    lstm = CuDNNLSTM if use_cudnnlstm else LSTM
    model = Sequential()
    model.add(lstm(hidden_size, input_shape=(
        size, unique_notes), return_sequences=True))
    model.add(lstm(hidden_size))
    model.add(Dropout(0.2))
    model.add(Dense(unique_notes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(
        lr=0.01) if optimizer == None else optimizer)
    return model


def load_model(name="model.h5"):
    return lm(name)


def save_model(model, name="model.h5"):
    model.save(name)
