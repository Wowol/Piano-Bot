import pretty_midi
import os
from tqdm import tqdm
import numpy as np
import sys
import argparse
import numpy as np
import pretty_midi


class Midi:

    def __init__(self, directory, batch_size=50):
        self.tokens = {}
        self.tokens[()] = 0
        self.number_of_unique = 1
        self.files = Midi._list_of_files_in_directory(directory)
        self.prepare_data(batch_size)

    def prepare_data(self, batch_size):
        music = []
        next_music = []

        for file_name in tqdm(self.files):
            piano_roll = Midi._get_piano_roll(file_name)
            dictionary = Midi._convert_to_dictionary(piano_roll)
            self.create_tokens(dictionary)

        for file_name in tqdm(self.files):
            piano_roll = Midi._get_piano_roll(file_name)
            dictionary = Midi._convert_to_dictionary(piano_roll)
            new = np.zeros((len(piano_roll[0]), len(self.tokens)), dtype=bool)

            last_key = max(dictionary)
            for k in range(last_key):
                q = self.tokens[tuple(sorted(dictionary[k]))
                                ] if k in dictionary else 0
                new[k][q] = True

            for i in range(0, len(new) - batch_size, 1):
                music.append(new[i: i + batch_size])
                next_music.append(new[i + batch_size])

        self.inputs = np.array(music)
        self.outputs = np.array(next_music)

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
        piano_roll = piano.get_piano_roll(fs=30)
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
        return inputs, outputs

    def create_tokens(self, dictionary):
        for k in dictionary.values():
            if tuple(sorted(k)) not in self.tokens:
                self.tokens[tuple(sorted(k))] = self.number_of_unique
                self.number_of_unique += 1

    def convert_input(self, input, token):
        for a in input:
            for k in a:
                pass


def piano_roll_to_pretty_midi(piano_roll, fs=60, program=0):
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
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
