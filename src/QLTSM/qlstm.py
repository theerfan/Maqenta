import torch
import torch.nn as nn

import pennylane as qml
from pennylane.templates import embeddings as emb
from pennylane.templates import layers as lay

from typing import Union


Embedding = Union[emb.AngleEmbedding, emb.AmplitudeEmbedding, emb.BasisEmbedding]
Layer = Union[
    lay.BasicEntanglerLayers,
    lay.ParticleConservingU1,
    lay.ParticleConservingU2,
    lay.RandomLayers,
    lay.StronglyEntanglingLayers,
]


class QLSTM(nn.Module):
    def quantum_op(
        self,
        wires,
        embedding: Embedding = emb.AngleEmbedding,
        layer: Layer = lay.BasicEntanglerLayers,
    ):
        def circuit_part(inputs, weights):
            embedding(inputs, wires=wires)
            layer(weights, wires=wires)
            return [qml.expval(qml.PauliZ(wires=w)) for w in wires]

        return circuit_part

    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits=4,
        n_qlayers=1,
        batch_first=True,
        return_sequences=False,
        return_state=False,
        backend="default.qubit",
        device="cpu"
    ):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"
        self.device = device # "cpu", "cuda"

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        # self.dev = qml.device("default.qubit", wires=self.n_qubits)
        # self.dev = qml.device('qiskit.basicaer', wires=self.n_qubits)
        # self.dev = qml.device('qiskit.ibm', wires=self.n_qubits)
        # use 'qiskit.ibmq' instead to run on hardware

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        self.qlayer_forget = qml.QNode(
            self.quantum_op(self.wires_forget), self.dev_forget, interface="torch"
        )

        self.qlayer_input = qml.QNode(
            self.quantum_op(self.wires_input), self.dev_input, interface="torch"
        )

        self.qlayer_update = qml.QNode(
            self.quantum_op(self.wires_update), self.dev_update, interface="torch"
        )

        self.qlayer_output = qml.QNode(
            self.quantum_op(self.wires_output), self.dev_output, interface="torch"
        )

        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = {
            "forget": qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes).to(device),
            "input": qml.qnn.TorchLayer(self.qlayer_input, weight_shapes).to(device),
            "update": qml.qnn.TorchLayer(self.qlayer_update, weight_shapes).to(device),
            "output": qml.qnn.TorchLayer(self.qlayer_output, weight_shapes).to(device),
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)
        # self.clayer_out = [torch.nn.Linear(n_qubits, self.hidden_size) for _ in range(4)]

    def forward(self, x, init_states=None):
        """
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        """
        # Automatically assumes single batch
        if len(x.shape) == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=self.device) # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size, device=self.device) # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]

            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1).float()

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(
                self.clayer_out(self.VQC["forget"](y_t).to(self.device))
            )  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.VQC["input"](y_t).to(self.device)))  # input block
            g_t = torch.tanh(self.clayer_out(self.VQC["update"](y_t).to(self.device)))  # update block
            o_t = torch.sigmoid(
                self.clayer_out(self.VQC["output"](y_t).to(self.device))
            )  # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        # Wow, such pseudo-keras!
        h_t, c_t = h_t.float(), c_t.float()

        if self.return_state:
            if self.return_sequences:
                return hidden_seq, (h_t, c_t)
            else:
                return (h_t, c_t)
        else:
            if self.return_sequences:
                return hidden_seq
            else:
                return h_t
    
    def predict(self, x, init_states=None):
        return self.forward(x, init_states)