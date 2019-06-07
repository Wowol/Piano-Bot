import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from random import shuffle
import numpy as np
from midi import Midi


class TrainModel:

    def __init__(self, epochs, midi_input, midi_output, frame_per_second,
                 batch_nnet_size, batch_song, optimizer, checkpoint, loss_fn,
                 checkpoint_prefix, model):
        self.epochs = epochs
        self.midi_input = midi_input
        self.midi_output = midi_output
        self.frame_per_second = frame_per_second
        self.batch_nnet_size = batch_nnet_size
        self.batch_song = batch_song
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        self.loss_fn = loss_fn
        self.checkpoint_prefix = checkpoint_prefix
        self.model = model

    def train(self):
        for epoch in tqdm(range(self.epochs), desc='epochs'):
            # for each epochs, we shufle the list of all the datasets
            # shuffle(self.sampled_200_midi)
            loss_total = 0
            steps = 0
            steps_nnet = 0

            # We will iterate all songs by self.song_size
            # for i in tqdm_notebook(range(0, self.total_songs, self.batch_song), desc='MUSIC'):

            steps += 1
            m = Midi("small_midi")
            inputs_nnet_large = self.midi_input
            outputs_nnet_large = self.midi_output
            # inputs_nnet_large = np.array(
            #     self.note_tokenizer.transform(inputs_nnet_large), dtype=np.int32)
            # outputs_nnet_large = np.array(
            #     self.note_tokenizer.transform(outputs_nnet_large), dtype=np.int32)

            index_shuffled = np.arange(
                start=0, stop=len(inputs_nnet_large))
            np.random.shuffle(index_shuffled)

         
            loss = self.train_step(inputs_nnet_large, outputs_nnet_large)
            loss_total += tf.math.reduce_sum(loss)
            if steps_nnet % 20 == 0:
                print("epochs {} | Steps {} | total loss : {}".format(
                    epoch + 1, steps_nnet, loss_total))

            # checkpoint.save(file_prefix=self.checkpoint_prefix)

    @tf.function
    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            loss = self.loss_fn(targets, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        return loss



