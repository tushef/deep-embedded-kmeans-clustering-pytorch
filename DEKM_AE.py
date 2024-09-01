import numpy as np
import torch
from clustpy.data import unflatten_images, flatten_images
from clustpy.deep.autoencoders._abstract_autoencoder import _AbstractAutoencoder
from numpy import ndarray
from torch import nn


class DEKM_AE(_AbstractAutoencoder):
    def __init__(self, input_shape, layers, embedding_size):
        super().__init__()
        self.input_shape = input_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[2], layers[0], 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(layers[0], layers[1], 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(layers[1], layers[2], 3, 2, 0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 3 * layers[2], embedding_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, 3 * 3 * layers[2]),
            Unflatten((layers[2], 3, 3)),
            nn.ReLU(),
            nn.ConvTranspose2d(layers[2], layers[1], 3, 2, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(layers[1], layers[0], 5, 2, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(layers[0], input_shape[2], 5, 2, 2, 1)
        )
        self.fitted = False

    def forward(self, x):
        x_encoded = self.encoder(x)
        gen = self.decoder(x_encoded)
        return x_encoded, gen

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], self.get_channels(), self.get_image_height(), self.get_image_width())
        x = self.encoder(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        x = x.reshape(x.shape[0], self.get_channels() * self.get_image_height() * self.get_image_width())
        return x

    def get_image_width(self):
        return self.input_shape[0]

    def get_image_height(self):
        return self.input_shape[1]

    def get_channels(self):
        return self.input_shape[2]


class Unflatten(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)