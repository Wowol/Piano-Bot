import numpy as np
from midi import piano_roll_to_pretty_midi
from random import randint
from tqdm import tqdm


class Song:

    def __init__(self, notes_length, unique_notes):
        self.notes_length = notes_length
        self.unique_notes = unique_notes

    def create(self, model, tokens, name, length=1000, first_notes=None, volume=127, fs=30, first_random=False):
        arr = np.zeros((1, self.notes_length, self.unique_notes), dtype=bool)

        if first_random:
            for k in range(self.notes_length):
                arr[0][k][randint(0, self.unique_notes-1)] = True

        elif first_notes is not None:
            arr = np.array([first_notes])

        notes = []
        for note_index in tqdm(range(length)):
            predicted = self._predict_next(model, arr)
            notes.append(predicted)
            arr = self._create_new_arr(predicted, arr)

        midi = self._create_piano_roll(notes, tokens, volume, fs)
        midi.write(name + ".mid")

    def _create_piano_roll(self, notes, tokens, volume, fs):
        music = np.zeros((128, len(notes)), dtype=float)

        inverse_tokens = {v: k for k, v in tokens.items()}
        for k in tqdm(range(len(notes))):
            a = notes[k]
            for q in inverse_tokens[notes[k]]:
                music[q][k] = volume

        return piano_roll_to_pretty_midi(music, fs=fs)

    def _predict_next(self, model, arr):
        predicted = model.predict(arr)[0]
        argmax = Song._sample(predicted)
        return argmax

    def _sample(preds, scale=0.8):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / scale
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def _create_new_arr(self, new_note, arr):
        bool_predicted = np.zeros(self.unique_notes, dtype=bool)
        bool_predicted[new_note] = True
        k = arr[0].tolist()
        k = k[1:]
        k.append(bool_predicted)
        k = [k]
        return np.array(k)
