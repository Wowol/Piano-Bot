import pretty_midi

def load(filename):
    midi_object = pretty_midi.PrettyMIDI('music.midi')
    piano = midi_object.instruments[0]
    piano_roll = piano.get_piano_roll()

def convert_to_dictionary(piano_roll):
    length = len(piano_roll[0])
    result = {k: [] for k in range(length)}

    for note in range(len(piano_roll)):
        for i in range(length):
            if piano_roll[note][i] != 0:
                if i not in result:
                    result[i] = []
                result[i].append(note)
    for key in result.keys():
        result[key] = ','.join(str(e) for e in result[key])
    return result


def generate_inputs_and_outputs_to_neural_network(dictionary, token, size=50):
    length = len(dictionary)
    inputs = [[] for i in range((len(dictionary)))]
    outputs = []
    for begin in range(length - size):
        if begin > size:
            outputs.append(token[dictionary[begin]])
        for s in range(begin, begin + size):
            inputs[begin].append(token[dictionary[s]])  # n - no note
    return inputs, outputs

    # if i in result:
    #     result[i] = ','.join(str(result[i]))


def create_tokens(dictionary):
    tokens = {}
    i = 0
    for k in dictionary.values():
        if k not in tokens:
            tokens[k] = i
            i += 1
    return tokens, {val: key for (key, val)in tokens.items()}


def convert_input(input, token):
    for a in input:
        for k in a:
            pass