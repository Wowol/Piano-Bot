from __future__ import print_function
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Dropout, TimeDistributed
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from model import Model

from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
from midi import Midi
from song import Song
import pretty_midi

NOTES_LENGTH = 100


midi = Midi("learn_midi", NOTES_LENGTH)


def create_song(model):
    song = Song(NOTES_LENGTH, midi.number_of_unique)
    song.create(model, midi.tokens, "qqq", first_random=True)


def create_model():
    model = Model()
    model.create(NOTES_LENGTH, midi.number_of_unique)
    model.learn(midi.inputs, midi.outputs, epochs=1)
    return model


def load_model():
    model = Model()
    model.load_from_file("cool_model.h5")
    return model


model = create_model()
model.save_to_file("model")
create_song(model)
