import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, activation="tanh"):
        super().__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        #######################################################################
        # TODO: Build a simple one layer RNN with an activation with the      #
        # attributes defined above and a forward function below. Use the      #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h as 0 if these values are not given.                     #
        #######################################################################
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        self.linear_x = nn.Linear(self.input_size, self.hidden_size)
        self.linear_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################

        (seq_len, batch_size, _) = x.shape
        if h is None:
            h = torch.zeros((1, batch_size, self.hidden_size))
        h_seq = torch.zeros((seq_len, batch_size, self.hidden_size))

        for i in range(seq_len):
            h = self.activation(self.linear_h(h) + self.linear_x(x[i]))
            h_seq[i] = h

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, h


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=20):
        super().__init__()
        #######################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes #
        # defined above and a forward function below. Use the                 #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h and c as 0 if these values are not given.               #
        #######################################################################
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_gate_fc1 = nn.Linear(self.input_size, self.hidden_size).to(device)
        self.input_gate_fc2 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.input_gate_act = nn.Sigmoid().to(device)

        self.forgot_gate_fc1 = nn.Linear(self.input_size, self.hidden_size).to(device)
        self.forgot_gate_fc2 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.forgot_gate_act = nn.Sigmoid().to(device)

        self.output_gate_fc1 = nn.Linear(self.input_size, self.hidden_size).to(device)
        self.output_gate_fc2 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.output_gate_act = nn.Sigmoid().to(device)

        self.memory_cell_fc1 = nn.Linear(self.input_size, self.hidden_size).to(device)
        self.memory_cell_fc2 = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.memory_cell_act = nn.Tanh().to(device)

        self.c_act = nn.Tanh().to(device)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None, c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = None
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        (seq_len, batch_size, _) = x.shape

        if h is None:
            h = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        
        if c is None:
            c = torch.zeros((1, batch_size, self.hidden_size)).to(device)

        h_seq = torch.zeros((seq_len, batch_size, self.hidden_size)).to(device)
        c_seq = torch.zeros((seq_len, batch_size, self.hidden_size)).to(device)

        for i in range(seq_len):
            f_t = self.forgot_gate_act(self.forgot_gate_fc1(x[i]) + self.forgot_gate_fc2(h))
            i_t = self.input_gate_act(self.input_gate_fc1(x[i]) + self.input_gate_fc2(h))
            c_tilda_t = self.memory_cell_act(self.memory_cell_fc1(x[i]) + self.memory_cell_fc2(h))

            c = c * f_t + i_t * c_tilda_t
            c_seq[i] = c

            o_t = self.output_gate_act(self.output_gate_fc1(x[i]) + self.output_gate_fc2(h))

            h = o_t * self.c_act(c)
            h_seq[i] = h


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, (h, c)


class RNN_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128,
                 activation="relu"):
        super(RNN_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a RNN classifier                                       #
        #######################################################################

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = RNN(input_size,hidden_size)
        self.fc = nn.Linear(hidden_size, classes)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[-1])
        return out

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class LSTM_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[-1])
        return out

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
