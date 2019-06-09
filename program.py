from utils import print_important
from random import randint
from song import Song
from midi import Midi
from model import Model
import argparse


NOTES_LENGTH = 100


parser = argparse.ArgumentParser(
    description='Piano bot')

parser.add_argument('-l', '--learn', action='store_true',
                    default=False, help='Learn new model')

parser.add_argument('-e', '--epocs', type=int, default=2,
                    help='Epocs of the new learned model')

parser.add_argument('-b', '--batch_size', type=int, default=256,
                    help='Batch size')

parser.add_argument('-sn', '--savename', type=str, default="model.h5",
                    help='Name of file in which new generated model will be saved')

parser.add_argument('-ln', '--loadname', type=str, default="trained_model.h5",
                    help='Load model from this file')

parser.add_argument('-mn', '--midi_name', type=str, default="song",
                    help='Name of the generated midi file')

parser.add_argument('-sb', '--song_begin', type=str, default="random",
                    help="How to begin new song (empty, random or from_existing)")

parser.add_argument('-m', '--learn_midi_directory', type=str, default="learn_midi",
                    help='Learn from this directory')

parser.add_argument('--length', type=int, default="1000",
                    help='Length of the new song')

parser.add_argument('-o', '--output_midi_directory', type=str, default="generated",
                    help='Generate to this dictionary')

args = parser.parse_args()


print_important(
    f"Loading midi files from directory {args.learn_midi_directory}...")
midi = Midi(args.learn_midi_directory, NOTES_LENGTH)


print_important(
    f"Files loaded!")


def get_model():
    model = Model()
    if args.learn:
        print_important("Creating model...")

        model.create(NOTES_LENGTH, midi.number_of_unique)
        model.learn(midi.inputs, midi.outputs, epochs=args.epocs,
                    batch_size=args.batch_size)
        model.save_to_file(args.savename)

        print_important(f"Model created and saved under {args.savename}")
    else:
        print_important(f"Loading model from file {args.loadname}")
        model.load_from_file(args.loadname)
    return model


def create_song(model):
    print_important("Creating song...")

    song = Song(NOTES_LENGTH, midi.number_of_unique)
    if args.song_begin == "random":
        song.create(model, midi.tokens, args.output_midi_directory +
                    "/" + args.midi_name, first_random=True, length=args.length)
    elif args.song_begin == "from_existing":
        song.create(model, midi.tokens, args.output_midi_directory +
                    "/" + args.midi_name, first_notes=midi.inputs[randint(0, len(midi.inputs - 3*NOTES_LENGTH))], length=args.length)
    else:
        song.create(model, midi.tokens, args.output_midi_directory +
                    "/" + args.midi_name, first_random=False, length=args.length)

    print_important(
        f"Song created: {args.output_midi_directory}/{args.midi_name}.mid")


model = get_model()
create_song(model)
