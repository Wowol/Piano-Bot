from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.layers import Dropout, TimeDistributed
try:
    from tensorflow.python.keras.layers import CuDNNLSTM as lstm
except:
    from tensorflow.keras.layers import Dense, Activation, LSTM as lstm
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model as lm

import numpy as np
import random
import sys
import io
from midi import Midi


class Model:
    def create(self, size, unique_notes, optimizer=None, hidden_size=128):
        self.model = Sequential()
        self.model.add(lstm(hidden_size, input_shape=(
            size, unique_notes), return_sequences=True))
        self.model.add(lstm(hidden_size))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(unique_notes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=RMSprop(
            lr=0.01) if optimizer == None else optimizer)

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

        # model = Sequential()
    # model.add(CuDNNLSTM(256, input_shape=(size, unique_notes)))
    # model.add(Dense(unique_notes, activation='softmax'))
    # optimizer = RMSprop(lr=0.01)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    # return model

    def load_from_file(self, name="model.h5"):
        self.model = lm(name)

    def save_to_file(self, name="model.h5"):
        self.model.save(name)

    def learn(self, inputs, outputs, batch_size=256, epochs=185):
        self.model.fit(inputs, outputs,
                       batch_size=batch_size,
                       epochs=epochs, verbose=True)

    def predict(self, arr):
        return self.model.predict(arr)
