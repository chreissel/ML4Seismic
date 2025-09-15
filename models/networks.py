from torch import nn, optim
import torch
import torch.nn.functional as F
from .s4d import S4D

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, activation='relu', output_activation=None, input_activation=None):
        super().__init__()
        layers = []
        if input_activation is not None:
            layers.append(input_activation())
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activations[activation])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        if output_activation is not None:
            layers.append(activations[output_activation])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


dropout_fn = nn.Dropout2d
class S4DModel(nn.Module):
    def __init__(self, d_input, d_output, d_model=256, n_layers=4, dropout=0.2, prenorm=False, maxpool=False):
        super().__init__()

        self.d_output = d_output
        self.prenorm = prenorm
        self.maxpool = maxpool

        self.encoder = nn.Linear(d_input, d_model)
        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)
            # Dropout on the output of the S4 block
            z = dropout(z)
            # Residual connection
            x = z + x
            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        if self.maxpool:
            # Pooling: average pooling over the sequence length
            x = x.mean(dim=1)
        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        return x

# model taken from here: https://git.ligo.org/tri.nguyen/deepclean-prod/-/blob/master/deepclean_prod/nn/net.py?ref_type=heads
class DeepClean(nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(in_channels),
            nn.Tanh()
        )
        
        self.downsampler = nn.Sequential()
        self.downsampler.add_module('CONV_1', nn.Conv1d(in_channels, 8, kernel_size=7, stride=2, padding=3))
        self.downsampler.add_module('BN_1', nn.BatchNorm1d(8))
        self.downsampler.add_module('TANH_1', nn.Tanh())
        self.downsampler.add_module('CONV_2', nn.Conv1d(8, 16, kernel_size=7, stride=2, padding=3))
        self.downsampler.add_module('BN_2', nn.BatchNorm1d(16))
        self.downsampler.add_module('TANH_2', nn.Tanh())
        self.downsampler.add_module('CONV_3', nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3))
        self.downsampler.add_module('BN_3', nn.BatchNorm1d(32))
        self.downsampler.add_module('TANH_3', nn.Tanh())
        self.downsampler.add_module('CONV_4', nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3))
        self.downsampler.add_module('BN_4', nn.BatchNorm1d(64))
        self.downsampler.add_module('TANH_4', nn.Tanh())
                                      
        self.upsampler = nn.Sequential()
        self.upsampler.add_module(
            'CONVTRANS_1', nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_1', nn.BatchNorm1d(32))
        self.upsampler.add_module('TANH_1', nn.Tanh())
        self.upsampler.add_module(
            'CONVTRANS_2', nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_2', nn.BatchNorm1d(16))
        self.upsampler.add_module('TANH_2', nn.Tanh())
        self.upsampler.add_module(
            'CONVTRANS_3', nn.ConvTranspose1d(16, 8, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_3', nn.BatchNorm1d(8))
        self.upsampler.add_module('TANH_3', nn.Tanh())
        self.upsampler.add_module(
            'CONVTRANS_4', nn.ConvTranspose1d(8,in_channels, kernel_size=7, stride=2, padding=3, output_padding=1))
        self.upsampler.add_module('BN_4', nn.BatchNorm1d(in_channels))
        self.upsampler.add_module('TANH_4', nn.Tanh())
        
        self.output_conv = nn.Conv1d(in_channels, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.downsampler(x)
        x = self.upsampler(x)
        x = self.output_conv(x)
        x = x.transpose(-1, -2)
        return x

