from __future__ import print_function
# import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
from midi import Midi
import pretty_midi


# from tensorflow.python.framework import ops
# ops.reset_default_graph()


def create_model(size, unique_notes):
    model = Sequential()
    model.add(CuDNNLSTM(512, input_shape=(
        size, unique_notes), return_sequences=False))

    model.add(Dense(unique_notes, activation='relu'))
#     model.compile(loss="mean_squared_error", optimizer="rmsprop")
#     model.add(Dropout(0.2))
#     model.add(CuDNNLSTM(512))
#     model.add(Dropout(0.2))
#     model.add(Dense(unique_notes, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer="rmsprop")
    return model

    # inputs = tf.keras.layers.Input(shape=(seq_len,))
    # embedding = tf.keras.layers.Embedding(
    #     input_dim=unique_notes+1, output_dim=output_emb, input_length=seq_len)(inputs)
    # forward_pass = tf.keras.layers.Bidirectional(
    #     tf.keras.layers.GRU(rnn_unit, return_sequences=True))(embedding)
    # # forward_pass, att_vector = SeqSelfAttention(
    # #     return_attention=True,
    # #     attention_activation='sigmoid',
    # #     attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    # #     attention_width=50,
    # #     kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    # #     bias_regularizer=tf.keras.regularizers.l1(1e-4),
    # #     attention_regularizer_weight=1e-4,
    # # )(forward_pass)
    # forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    # forward_pass = tf.keras.layers.Bidirectional(
    #     tf.keras.layers.GRU(rnn_unit, return_sequences=True))(forward_pass)
    # # forward_pass, att_vector2 = SeqSelfAttention(
    # #     return_attention=True,
    # #     attention_activation='sigmoid',
    # #     attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    # #     attention_width=50,
    # #     kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    # #     bias_regularizer=tf.keras.regularizers.l1(1e-4),
    # #     attention_regularizer_weight=1e-4,
    # # )(forward_pass)
    # forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    # forward_pass = tf.keras.layers.Bidirectional(
    #     tf.keras.layers.GRU(rnn_unit))(forward_pass)
    # forward_pass = tf.keras.layers.Dropout(dropout)(forward_pass)
    # forward_pass = tf.keras.layers.Dense(dense_unit)(forward_pass)
    # forward_pass = tf.keras.layers.LeakyReLU()(forward_pass)
    # outputs = tf.keras.layers.Dense(
    #     unique_notes+1, activation="softmax")(forward_pass)

    # model = tf.keras.Model(inputs=inputs, outputs=outputs,
    #                        name='generate_scores_rnn')
    # return model


note_size = 50


def predict_next(model):

    music_length = 1000

    # arr = np.random.rand(1, note_size, 128)*127
    # arr = np.random.choice(a=[False, True], size=(
    #     1, note_size, 128), p=[0.5, 0.5])

    midi = Midi("small_midi")


    inputs, outputs = midi.prepare_data(note_size)

    # arr = np.zeros((1, note_size, 128), dtype=float)
    arr = np.array(inputs)

    music = np.zeros((128, music_length), dtype=float)

    for note_index in range(music_length):
        predicted = model.predict(arr)[0]
        print(predicted)
        input()
        bool_predicted = [True if k > 0 else False for k in predicted]

        # kek_predicted = [255. - 255. / (255. + k) for k in predicted]
        # print(predicted)
        for i in range(len(predicted)):
            music[i][note_index] = predicted[i]#127 if predicted[i] > 0 else 0
        k = arr[0].tolist()
        k = k[1:]
        k.append(predicted)
        k = [k]
        arr = np.array(k)

    # music = np.norma
    generate_to_midi = Midi.piano_roll_to_pretty_midi(music, fs=10)
    generate_to_midi.write("kek.midi")
    print(music)


midi = Midi("small_midi")


inputs, outputs = midi.prepare_data(note_size)

# i = 0
# for k in outputs:
#     i+=1
#     print(i)
#     print(k)
#     input()

inputs = np.array(inputs)
outputs = np.array(outputs)

model = create_model(note_size, 128)


model.fit(inputs, outputs,
          batch_size=128,
          epochs=10, verbose=True)

model.save('my_model_bool.h5')

# model = load_model('my_model_bool.h5')
predict_next(model)
