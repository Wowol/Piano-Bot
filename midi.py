import pretty_midi
import os
from tqdm import tqdm


class Midi:
    def __init__(self, directory):
        self.tokens = {}
        self.number_of_unique = 1
        self.files = Midi._list_of_files_in_directory(directory)

    def prepare_data(self):
        for file_name in tqdm(self.files):
            piano_roll = Midi._get_piano_roll(file_name)
            dictionary = Midi._convert_to_dictionary(piano_roll)
            self.create_tokens(dictionary)
            inputs, outputs = self.generate_inputs_and_outputs_to_neural_network(
                dictionary)
            print(len(inputs), len(outputs))

            return inputs, outputs, self.number_of_unique

    def _list_of_files_in_directory(directory):
        result = []
        for path, subdirs, files in os.walk(directory):
            for name in files:
                result.append(os.path.join(path, name))
        return result

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
        # for key in result.keys():
        #     result[key] = ','.join(str(e) for e in result[key])
        return result

    def generate_inputs_and_outputs_to_neural_network(self, dictionary, size=50):
        length = len(dictionary)
        # inputs = [[] for i in range((len(dictionary)))]
        inputs = []
        outputs = []

        def get_first_time_note():
            for i in range(len(dictionary)):
                if i in dictionary:
                    return i

        start = get_first_time_note()

        inputs.append(list(0 for _ in range(size)))
        for k in dictionary[start]:
            inputs[k] = True
        inputs[0].append(dictionary[start])

        last_key = max(dictionary)
        for begin in range(1, last_key):
            inputs.append(inputs[begin-1][:])
            inputs[begin].pop(0)
            inputs[begin].append(
                dictionary[begin+start] if begin+start in dictionary else 0)
            outputs.append(dictionary[begin+start - 1]
                           if begin+start - 1 in dictionary else 0)

        outputs.append(0)
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
            if k not in self.tokens:
                self.tokens[k] = self.number_of_unique
                self.number_of_unique += 1
        # return tokens#, {val: key for (key, val)in tokens.items()}

    def convert_input(self, input, token):
        for a in input:
            for k in a:
                pass


# midi = Midi("small_midi")

# midi.prepare_data()
