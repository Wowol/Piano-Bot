from __future__ import print_function
# import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Dropout, TimeDistributed
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

    # model = Sequential()
    # model.add(CuDNNLSTM(256, input_shape=(size, unique_notes)))
    # model.add(Dense(unique_notes, activation='softmax'))
    # optimizer = RMSprop(lr=0.01)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    # return model

    hidden_size = 128
    model = Sequential()
    # model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
    model.add(CuDNNLSTM(hidden_size, input_shape=(
        size, unique_notes), return_sequences=True))
    model.add(CuDNNLSTM(hidden_size))
    model.add(Dropout(0.2))
    model.add(Dense(unique_notes))
    # model.add(TimeDistributed(Dense(unique_notes)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    # model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

#     model = Sequential()
#     model.add(CuDNNLSTM(512, input_shape=(
#         size, unique_notes), return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(CuDNNLSTM(512))
#     model.add(Dropout(0.2))

#     model.add(TimeDistributed(Dense(unique_notes)))
#     model.Add(Activation('softmax'))
#     # model.add(Dense(unique_notes, activation='softmax'))
# #     model.compile(loss="mean_squared_error", optimizer="rmsprop")
# #     model.add(Dense(unique_notes, activation='linear'))

#     model.compile(loss='categorical_crossentropy', optimizer="adam")
#     return model

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
note_size = 100

def sample(preds, temperature=0.8):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def predict_next(model, midi):
    unique_notes = midi.number_of_unique

    music_length = 5000

    # arr = np.random.rand(1, note_size, 128)*127
    # arr = np.random.choice(a=[False, True], size=(
    #     1, note_size, unique_notes), p=[0.5, 0.5])

    # arr = np.zeros((1, note_size, unique_notes), dtype=bool)

    arr = np.array([midi.inputs[0]])

    # for k in range(note_size):
    #     # arr[0][k][random.randint(0, unique_notes-1)] = True
    #     arr[0][k][1] = True

    music = np.zeros((128, music_length), dtype=float)

    notes = []
    for note_index in range(music_length):
        predicted = model.predict(arr)[0]
        # argmax = np.argmax(predicted)
        argmax = sample(predicted)
        # print(argmax)
        # argmax = np.random.choice(len(predicted), 1, p=predicted)[0]
        # print(predicted)
        # print(argmax)
        notes.append(argmax)

        # input()
        bool_predicted = np.zeros(unique_notes, dtype=bool)
        bool_predicted[argmax] = True

        # for i in range(len(predicted)):
        #     music[i][note_index] = predicted[i]#127 if predicted[i] > 0 else 0
        k = arr[0].tolist()
        k = k[1:]
        k.append(bool_predicted)
        k = [k]
        arr = np.array(k)

    # music = np.norma
    inverse_tokens = {v: k for k, v in midi.tokens.items()}
    for k in range(len(notes)):
        a = notes[k]
        for q in inverse_tokens[notes[k]]:
            music[q][k] = 127

    generate_to_midi = Midi.piano_roll_to_pretty_midi(music, fs=30)
    generate_to_midi.write("kek.midi")


midi = Midi("small_midi", note_size)


inputs = np.array(midi.inputs)
outputs = np.array(midi.outputs)
unique_notes = midi.number_of_unique
print("*******************************************")
print("Input size:", len(inputs))
print("UNIQUE:", unique_notes)
print("*******************************************")

# model = create_model(note_size, unique_notes)


# model.fit(inputs, outputs,
#           batch_size=256,
#           epochs=185, verbose=True)

# model.save('my_model_bool3.h5')

model = load_model('cool_model.h5')
predict_next(model, midi)
