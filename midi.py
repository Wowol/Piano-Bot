from __future__ import division
import pretty_midi
import os
from tqdm import tqdm
import numpy as np
import sys
import argparse
import numpy as np
import pretty_midi

class Midi:
    def __init__(self, directory):
        self.tokens = {}
        self.number_of_unique = 1
        self.files = Midi._list_of_files_in_directory(directory)

    def prepare_data(self, batch_size=50):
        music = []
        next_music = []

        for file_name in tqdm(self.files):
            piano_roll = Midi._get_piano_roll(file_name)

            # # # # # # tr = np.array(piano_roll).transpose()
            # # # # # # new = np.zeros((128, len(tr)+batch_size), dtype=float)
            # # # # # # for x in range(128):
            # # # # # #     new[x][50:] = piano_roll[x]

            # # # # # # new = np.transpose(new)

            # # # # # # # tr = tr.astype(bool)

            # # # # # # next_chars = []
            # # # # # # for i in range(0, len(new) - batch_size, 1):
            # # # # # #     music.append(new[i: i + batch_size])
            # # # # # #     next_music.append(new[i + batch_size])

            dictionary = Midi._convert_to_dictionary(piano_roll)
            self.create_tokens(dictionary)
            print(len(self.tokens))
        print
        return music, next_music

        # inputs, outputs = self.generate_inputs_and_outputs_to_neural_network(
        #     dictionary)
        # print(len(inputs), len(outputs))

        # return inputs, outputs, self.number_of_unique

    def _list_of_files_in_directory(directory):
        result = []
        for path, subdirs, files in os.walk(directory):
            for name in files:
                result.append(os.path.join(path, name))
        return result

    def _convert_to_good(piano_roll):
        great_arr = []
        length = len(piano_roll[0])
        for x in range(length):
            great_arr.append([])
            for y in range(128):
                great_arr[-1].append(True if piano_roll[y][x] != 0. else False)

        return great_arr

    def _get_piano_roll(filename):
        midi_object = pretty_midi.PrettyMIDI(filename)
        piano = midi_object.instruments[0]
        piano_roll = piano.get_piano_roll()
        return piano_roll

    def _convert_to_dictionary(piano_roll):
        length = len(piano_roll[0])
        result = {}

        for note in range(len(piano_roll)):
            for i in range(length):
                if piano_roll[note][i] != 0:
                    if i not in result:
                        result[i] = []
                    result[i].append(note)
        return result

    def generate_inputs_and_outputs_to_neural_network(self, dictionary, size=3):
        length = len(dictionary)
        inputs = []

        def get_first_time_note():
            for i in range(len(dictionary)):
                if i in dictionary:
                    return i

        start = get_first_time_note()

        inputs.append(list([False]*128 for _ in range(size)))

        for k in dictionary[start]:
            inputs[0][size - 1][k] = True

        last_key = max(dictionary)
        outputs = [[False]*128]*last_key

        for begin in range(1, last_key):
            inputs.append(inputs[begin-1][:])
            inputs[begin].pop(0)
            inputs[begin].append([False]*128)
            if begin+start in dictionary:
                for k in dictionary[begin+start]:
                    inputs[begin][size-1][k] = True
                    outputs[begin - 1][k] = True
            #     dictionary[begin+start] if begin+start in dictionary else 0)

            # if begin+start - 1 in dictionary:
            #     for k in dictionary[begin+start - 1]:
            #         break
        return inputs, outputs

        # for begin in range(length - size):
        #     if begin < size:
        #         for i in range(size-begin):
        #             inputs[begin].append(self.tokens['n'])

        #         for i in range(begin):
        #             inputs[begin].append(self.tokens[dictionary[i]])

        #     if begin > size:
        #         outputs.append(self.tokens[dictionary[begin]])
        #     for s in range(begin, begin + size):
        #         inputs[begin].append(self.tokens[dictionary[s]])  # n - no note
        # return inputs, outputs

    def create_tokens(self, dictionary):
        for k in dictionary.values():
            if tuple(sorted(k)) not in self.tokens:
                self.tokens[tuple(sorted(k))] = self.number_of_unique
                self.number_of_unique += 1
        # return tokens#, {val: key for (key, val)in tokens.items()}

    def convert_input(self, input, token):
        for a in input:
            for k in a:
                pass




    def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
        '''Convert a Piano Roll array into a PrettyMidi object
        with a single instrument.
        Parameters
        ----------
        piano_roll : np.ndarray, shape=(128,frames), dtype=int
            Piano roll of one instrument
        fs : int
            Sampling frequency of the columns, i.e. each column is spaced apart
            by ``1./fs`` seconds.
        program : int
            The program number of the instrument.
        Returns
        -------
        midi_object : pretty_midi.PrettyMIDI
            A pretty_midi.PrettyMIDI class instance describing
            the piano roll.
        '''
        notes, frames = piano_roll.shape
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=program)

        # pad 1 column of zeros so we can acknowledge inital and ending events
        piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

        # use changes in velocities to find note on / note off events
        velocity_changes = np.nonzero(np.diff(piano_roll).T)

        # keep track on velocities and note on times
        prev_velocities = np.zeros(notes, dtype=int)
        note_on_time = np.zeros(notes)

        for time, note in zip(*velocity_changes):
            # use time + 1 because of padding above
            velocity = piano_roll[note, time + 1]
            time = time / fs
            if velocity > 0:
                if prev_velocities[note] == 0:
                    note_on_time[note] = time
                    prev_velocities[note] = velocity
            else:
                pm_note = pretty_midi.Note(
                    velocity=prev_velocities[note],
                    pitch=note,
                    start=note_on_time[note],
                    end=time)
                instrument.notes.append(pm_note)
                prev_velocities[note] = 0
        pm.instruments.append(instrument)
        return pm


# midi = Midi("small_midi")

# midi.prepare_data()
