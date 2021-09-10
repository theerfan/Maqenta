from midi import Midi
from QLTSM.LSTMusic import LSTMusic
import torch

from pathlib import Path

from pennylane import numpy as np

import matplotlib.pyplot as plt

seq_length = 25
n_epochs = 10
cutoff = 20
n_qubits = 4

model_name = f"lstm-seq{seq_length}-cut{cutoff}-epcs{n_epochs}-qu{n_qubits}"
model_str = f"saved_models/{model_name}.pt"

print("Initialized Midi")
midi = Midi(seq_length)

print("Initialized LSTM")
lstm = LSTMusic(hidden_dim=midi.n_vocab, n_qubits=n_qubits)

if Path(model_str).is_file():
    print("Loading model")
    lstm.load_state_dict(torch.load(model_str))
    lstm.eval()
    # lstm = torch.load(model_str)
else:
    print("Training LSTM")
    train_history = lstm.train(
        True, midi.network_input, midi.network_output, n_epochs=n_epochs, cutoff=cutoff
    )
    torch.save(lstm.state_dict(), model_str)

print("Generating notes")
notes = lstm.generate_notes(
    midi.network_input, midi.int_to_note, midi.n_vocab, n_notes=20
)

print("Saving as MIDI file.")
midi.create_midi_from_model(notes, f"generated_songs/{model_name}_generated.mid")


# train()
# lstm_q = QLSTM(1, 1, n_qubits=10)

# history = lstm_q.train(midi.network_input, midi.network_output, n_epochs=20)

# def plot_history(history_quantum):
#     # loss_c = history_classical['loss']
#     loss_q = history_quantum['loss']
#     n_epochs = len(loss_q)
#     x_epochs = [i for i in range(n_epochs)]

#     plt.figure(figsize=(6, 4))
#     # plt.plot(loss_c, label="Classical LSTM")
#     plt.plot(loss_q, label="Quantum LSTM")
#     plt.title("POS Tagger Training")
#     plt.ylabel("Loss")
#     plt.xlabel("Epoch")
#     plt.ylim(0., 1.5)
#     plt.legend(loc="upper right")

# plot_history(history)
