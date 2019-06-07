# from __future__ import print_function
# import tensorflow as tf
# from tensorflow import keras
# from midi import Midi
# from keras_self_attention import SeqSelfAttention
# from trainmodel import TrainModel
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.callbacks import LambdaCallback
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.optimizers import RMSprop
# from keras.utils.data_utils import get_file
# import numpy as np
# import random
# import sys
# import io

from midi import Midi

# def create_model(seq_len, unique_notes, dropout=0.3, output_emb=100, rnn_unit=128, dense_unit=64):

#     model = Sequential()
#     model.add(LSTM(256, input_shape=(50, unique_notes)))
#     model.add(Dense(unique_notes, activation='softmax'))
#     optimizer = RMSprop(lr=0.01)
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer)
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


midi = Midi("midi_files")

inputs, outputs, unique_notes = midi.prepare_data()

model = create_model(50, unique_notes)

model.fit(inputs, outputs,
          batch_size=128,
          epochs=1)



# seq_len = 50
# EPOCHS = 4
# BATCH_SONG = 16
# BATCH_NNET_SIZE = 96
# FRAME_PER_SECOND = 5

# train_class = TrainModel(EPOCHS, inputs, outputs, FRAME_PER_SECOND,
#                          BATCH_NNET_SIZE, BATCH_SONG, "adam", None, keras.losses.mean_squared_error,
#                          None, model)

# train_class.train()












# fashion_mnist = keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# train_images = train_images / 255.0

# test_images = test_images / 255.0

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=5)
# model.fit(train_images, train_labels, epochs=5)


# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print('Test accuracy:', test_acc)

# model = create_model(50, 10, 20)
# print(model.summary())


# model = tf.keras.models.Sequential()
# d = convert_to_dictionary(piano_roll)
# token, reversed_token = create_tokens(d)
# # print(d)
# inputs, outputs = generate_inputs_and_outputs_to_neural_network(d, token)

# print(token)

# model.add(Embedding(2500, embed_dim,input_length = X.shape[1], dropout = 0.2))
# model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
# model.add(Dense(2,activation='softmax'))
# model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
# print(model.summary())


# token, reversed_token = create_tokens(inputs)


# print(create_tokens(inputs))
# print(generate_inputs_and_outputs_to_neural_network(d)[1])

# print(type(piano_roll))
# # print(len(piano_roll[1]))
# # print(k for k in piano_roll[64])

# for k in piano_roll:
#     for i in k:
#         if i != 0:
#             print(i, end=' ')
#     print()
