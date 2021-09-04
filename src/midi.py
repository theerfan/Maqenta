from music21 import converter, instrument, note, chord
from maq_utils.keras import to_categorical
import glob, pickle
import numpy as np
from pathlib import Path

notes_dir = "data/notes.pk"


class Midi:
    def __init__(self):
        if Path(notes_dir).is_file():
            self.notes = pickle.load(open(notes_dir, "rb"))
        else:
            self.notes = self.get_notes()
            pickle.dump(self.notes, open(notes_dir, "wb"))

        self.network_input, self.network_output = self.prepare_sequences(self.notes)
        print(f"Input shape: {self.network_input.shape}")
        print(f"Output shape: {self.network_output.shape}")

    def get_notes(self):
        """Get all the notes and chords from the midi files in the ./midi_songs directory"""
        # This is assuming that every interval between notes is the same (0.5)
        notes = []

        for file in glob.glob("midi_songs/*.mid"):
            midi = converter.parse(file)

            print("Parsing %s" % file)

            notes_to_parse = None

            try:  # file has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except:  # file has notes in a flat structure
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))

        with open(notes_dir, "wb") as filepath:
            pickle.dump(notes, filepath)

        return notes

    def prepare_sequences(self, notes, n_vocab: int = None):
        """Prepare the sequences used by the Neural Network"""
        if not n_vocab:
            n_vocab = len(set(notes))

        sequence_length = 100

        # get all pitch names
        pitchnames = sorted(set(item for item in notes))

        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

        network_input = []
        network_output = []

        # create input sequences and the corresponding outputs
        for i in range(0, len(self.notes) - sequence_length, 1):
            sequence_in = self.notes[i : i + sequence_length]
            sequence_out = self.notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)

        # reshape the input into a format compatible with LSTM layers
        network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab)

        network_output = to_categorical(network_output)

        return (network_input, network_output)
