import tensorflow as tf
from tensorflow import keras 
embed_dim = 128
lstm_out = 200
batch_size = 32


def create_model(batch_size, seq_length, unique_chars):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Embedding(input_dim=unique_chars, output_dim=512,
                                        batch_input_shape=(batch_size, seq_length)))

    model.add(tf.keras.layers.LSTM(256, return_sequences=True, stateful=True))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.LSTM(256, return_sequences=True, stateful=True))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.LSTM(256, return_sequences=True, stateful=True))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(unique_chars)))
    model.add(tf.keras.layers.Activation("softmax"))

    return model
print(tf.test.gpu_device_name())# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growthconfig = tf.ConfigProto()config.gpu_options.allow_growth = True


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
