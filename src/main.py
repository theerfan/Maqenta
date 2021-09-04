from midi import Midi
from QLTSM.lstm_q import LSTM_Q
import matplotlib.pyplot as plt

midi = Midi(25)

lstm_q = LSTM_Q(1, 1, n_qubits=10)

history = lstm_q.train(midi.network_input, midi.network_output, n_epochs=20)

def plot_history(history_quantum):
    # loss_c = history_classical['loss']
    loss_q = history_quantum['loss']
    n_epochs = len(loss_q)
    x_epochs = [i for i in range(n_epochs)]
    
    plt.figure(figsize=(6, 4))
    # plt.plot(loss_c, label="Classical LSTM")
    plt.plot(loss_q, label="Quantum LSTM")
    plt.title("POS Tagger Training")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.ylim(0., 1.5)
    plt.legend(loc="upper right")

plot_history(history)