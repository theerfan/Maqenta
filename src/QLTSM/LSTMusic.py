import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from QLTSM.qlstm import QLSTM

import numpy as np


class LSTMusic(nn.Module):
    def __init__(
        self,
        n_vocab=None,
        input_dim=1,
        hidden_dim=512,
        n_qubits=4,
        backend="default.qubit",
        device="cpu",
    ):
        super(LSTMusic, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if n_qubits > 0:
            print(f"Generator will use Quantum LSTM running on backend {backend}")

            # self.model = nn.Sequential(
            #     QLSTM(
            #         input_size=input_dim,
            #         hidden_size=hidden_dim,
            #         n_qubits=n_qubits,
            #         backend=backend,
            #     ),
            #     nn.Dropout(0.3),
            #     QLSTM(
            #         input_size=input_dim,
            #         hidden_size=hidden_dim,
            #         n_qubits=n_qubits,
            #         backend=backend,
            #     ),
            #     nn.Dropout(0.3),
            #     QLSTM(
            #         input_size=input_dim,
            #         hidden_size=hidden_dim,
            #         n_qubits=n_qubits,
            #         backend=backend,
            #     ),
            #     nn.Linear(hidden_dim, hidden_dim),
            #     nn.Dropout(0.3),
            #     nn.Linear(hidden_dim, n_vocab),
            # )

            self.model = QLSTM(
                input_dim,
                hidden_dim,
                n_qubits=n_qubits,
                backend=backend,
                return_state=True,
                device=device,
            ).to(device)
        else:
            print("Generator will use Classical LSTM")
            self.model = nn.LSTM(input_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space

    def forward(self, note_sequence):
        (h_t, c_t) = self.model(note_sequence)
        scores = F.log_softmax(c_t, dim=1)
        return scores

        # embeds = self.word_embeddings(sentence)
        # lstm_out, _ = self.model(embeds.view(len(sentence), 1, -1))
        # tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # tag_scores = F.log_softmax(tag_logits, dim=1)
        # return tag_scores

    def train(
        self,
        mode=True,
        inputs=None,
        outputs=None,
        n_epochs=None,
        cutoff: int = None,
        learning_rate=0.1,
    ):
        # Same as categorical cross entropy, who would've thought?!
        if mode == False:
            return
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        if cutoff:
            inputs = inputs[:cutoff]
            outputs = outputs[:cutoff]

        history = {"loss": []}

        for epoch in range(n_epochs):
            counter = 0
            losses = []
            for note_series, next_note in zip(inputs, outputs):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.zero_grad()

                # Step 2. Run our forward pass.
                c_t = self(note_series)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(c_t, next_note.reshape(1).long())
                loss.backward()
                optimizer.step()
                losses.append(float(loss))

                if counter % 100 == 0:
                    print(f"On datapoint #{counter} out of {cutoff}")
                counter += 1

            avg_loss = np.mean(losses)
            history["loss"].append(avg_loss)
            print("Epoch {} / {}: Loss = {:.3f}".format(epoch + 1, n_epochs, avg_loss))
        return history

    def generate_notes(self, network_input, int_to_note, n_vocab, n_notes):
        """Generate notes from the neural network based on a sequence of notes"""
        # pick a random sequence from the input as a starting point for the prediction
        with torch.no_grad():
            start = np.random.randint(0, len(network_input) - 1)

            pattern = network_input[start]
            prediction_output = []

            # generate 500 notes
            for _ in range(n_notes):
                prediction_input = pattern.clone().detach().reshape(1, len(pattern), 1)
                # prediction_input = prediction_input / float(n_vocab)

                (h_t, prediction) = self.model.predict(prediction_input)

                index = prediction.argmax()
                result = int_to_note[int(index)]
                prediction_output.append(result)

                added_index = (index / n_vocab).reshape(1, 1)

                pattern = torch.cat((pattern, added_index), 0)
                # pattern.append(index)
                pattern = pattern[1 : len(pattern)]

            return prediction_output
