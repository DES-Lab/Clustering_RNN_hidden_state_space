from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import nn
from torch.nn.functional import softmax

rnn_types = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}


class RecurrentNN(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, layer_dim, output_dim, nonlinearity,
                 dropout_prob, data_handler, device=None):
        super(RecurrentNN, self).__init__()
        assert model_type in rnn_types.keys()
        self.model_type = model_type
        self.data_handler = data_handler
        self.activation_fun = f'_{nonlinearity}' if model_type == 'rnn' else ''

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # Defining the device
        if device is None:
            self.device = torch.device('cuda:0' if device != 'cpu' and torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # RNN
        if model_type == 'rnn':
            self.rnn = rnn_types[model_type](input_dim, hidden_dim, layer_dim, nonlinearity=nonlinearity,
                                             batch_first=True, dropout=dropout_prob).to(self.device)
        else:
            self.rnn = rnn_types[model_type](input_dim, hidden_dim, layer_dim,
                                             batch_first=True, dropout=dropout_prob).to(self.device)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim).to(self.device)
        # Hidden and Cell state
        self.hs, self.cs = None, None
        # Model Name consist of experiment name and training configuration
        self.model_name = None

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c0 = None
        if self.model_type == 'lstm':
            # Initializing cell state for first input with zeros
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # Forward propagation by passing in the input and hidden state into the model
        if self.model_type == 'lstm':
            out, (hs, hc) = self.rnn(x, (h0.detach(), c0.detach()))
        else:
            out, hs = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out

    def reset_hidden_state(self):
        self.hs = torch.zeros(self.layer_dim, 1, self.hidden_dim).to(self.device)
        if self.model_type == 'lstm':
            self.cs = torch.zeros(self.layer_dim, 1, self.hidden_dim).to(self.device)

    def step(self, inp, return_hidden=False):
        if inp is not None:
            inp = tuple([inp])
            inp = self.data_handler.input_tensor(inp, len(inp))
            inp.unsqueeze_(1)

            if self.model_type == 'lstm':
                out, (self.hs, self.cs) = self.rnn(inp, (self.hs, self.cs))
            else:
                out, self.hs = self.rnn(inp, self.hs)
        else:
            out = self.hs[-1]

        out = self.fc(out)
        out.squeeze_()
        p = softmax(out, dim=0).data
        ind = torch.argmax(p).item()

        if return_hidden:
            if self.model_type == 'lstm':
                return self.data_handler.index_to_class[ind], (self.hs, self.cs)
            else:
                return self.data_handler.index_to_class[ind], self.hs
        else:
            return self.data_handler.index_to_class[ind]

    def get_model_name(self, exp_name=None):
        if self.model_name:
            return self.model_name
        else:
            assert exp_name is not None
            self.model_name = f'{self.model_type}{self.activation_fun}_l{self.layer_dim}' \
                              f'_d{self.hidden_dim}_{exp_name}'
            return exp_name


class Optimization:
    def __init__(self, model, optimizer, device=None):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.accuracy = []
        if device is None:
            self.device = torch.device('cuda:0' if device != 'cpu' and torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def train_step(self, x, y):
        self.optimizer.zero_grad()
        # Makes predictions
        yhat = self.model(x)

        # squeeze only one dimension -> squeeze_() would squeeze two if batch size is equal to 1
        # TODO check this wrt batch size
        if y.shape[0] == 1:
            y = y.squeeze(0)
        else:
            y.squeeze_()

        # Computes loss
        loss = self.loss_fn(yhat, y)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_set, val_set, n_epochs=50, early_stop=False, exp_name='exp', save_location='',
              save_interval=0, verbose=True, save=True, load=True):

        self.model.get_model_name(exp_name)
        model_path = f'{save_location}{self.model.model_name}'
        # model_path = self.model.model_name

        if load and self.load(model_path):
            if verbose:
                print('Model loaded.')
            return

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            self.model.train()
            for x_batch, y_batch in train_set:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                self.model.eval()
                batch_val_losses = []

                total, correct = 0, 0
                for x_val, y_val in val_set:
                    y_val = y_val.to(self.device)
                    # squeeze only one dimension -> squeeze_() would squeeze two if batch size is equal to 1
                    if y_val.shape[0] == 1:
                        y_val = y_val.squeeze(0)
                    else:
                        y_val.squeeze_()

                    predictions = self.model(x_val)
                    val_loss = self.loss_fn(predictions, y_val).item()

                    _, predicted = predictions.max(1)
                    total += predictions.size(0)
                    correct += predicted.eq(y_val).sum().item()

                    y_val.unsqueeze_(1)
                    batch_val_losses.append(val_loss)

                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

                validation_set_accuracy = 100. * correct / total
                self.accuracy.append(validation_set_accuracy)

            if (epoch <= 10) | (epoch % 5 == 0) and verbose:
                print(
                    f"[{epoch}/{n_epochs}] "
                    f"Training loss: {training_loss:.4f}\t "
                    f"Validation loss: {validation_loss:.4f}\t "
                    f"Accuracy: {validation_set_accuracy:.2f}%")

            if save_interval > 0 and epoch % save_interval == 0:
                torch.save(self.model.state_dict(), f'{model_path}_epoch_{epoch}')

            if early_stop:
                if mean(self.train_losses[-3:]) < 0.00001 and mean(self.val_losses[-3:]) < 0.00001 or mean(self.accuracy[-3:]) == 100.0:
                    if verbose:
                        print('Early Stopping based on Patience')
                        break

        if verbose and 1.0 in self.accuracy:
            print(f'Model: {self.model.model_name}')
            print(f'Perfect accuracy reached in epoch {self.accuracy.index(1.0)}')

        if save:
            torch.save(self.model.state_dict(), model_path)

        self.model.eval()

    def load(self, path):
        file = Path(path)
        if file.is_file():
            self.model.load_state_dict(torch.load(path,  map_location=self.device))
            self.model.eval()
            return True
        return False

    def save(self, path):
        torch.save(self.model.state_dict(), path)


def get_model(model, model_params):
    return RecurrentNN(model, **model_params)
