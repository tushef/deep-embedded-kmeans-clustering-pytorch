"""
Deep Embedded K-Means Autoencoder Module

This module implements the autoencoder architecture used in the Deep Embedded K-Means
clustering algorithm. The autoencoder consists of a convolutional encoder and decoder
with configurable layer sizes.

"""

import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Optional


class ConvAutoencoder(nn.Module):
    """
    Deep Embedded K-Means Autoencoder
    
    A convolutional autoencoder designed for deep embedded clustering.
    The architecture consists of:
    - Encoder: Conv2d layers with ReLU activation and downsampling
    - Decoder: ConvTranspose2d layers with ReLU activation and upsampling
    
    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        Input image shape as (height, width, channels)
    embedding_size : int
        Size of the embedding layer (bottleneck)
    layers : List[int], default=[32, 64, 128]
        Number of filters in each convolutional layer
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], 
                 embedding_size: int, 
                 layers: List[int] = [32, 64, 128]):
        super().__init__()
        
        # Validate inputs
        if len(input_shape) != 3:
            raise ValueError("input_shape must be a tuple of (height, width, channels)")
        if len(layers) != 3:
            raise ValueError("layers must be a list of exactly 3 integers")
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        
        self.input_shape = input_shape
        self.embedding_size = embedding_size
        self.layers = layers
        self.fitted = False
        
        # Calculate dimensions for the linear layer
        self._calculate_dimensions()
        
        # Build encoder and decoder
        self._build_encoder()
        self._build_decoder()
    
    def _calculate_dimensions(self):
        """Calculate the dimensions for the linear layer in the encoder."""
        # Simulate the encoder operations to get the final spatial dimensions
        height, width = self.input_shape[0], self.input_shape[1]
        
        # First conv: kernel=5, stride=2, padding=2
        height = (height + 2 * 2 - 5) // 2 + 1
        width = (width + 2 * 2 - 5) // 2 + 1
        
        # Second conv: kernel=5, stride=2, padding=2
        height = (height + 2 * 2 - 5) // 2 + 1
        width = (width + 2 * 2 - 5) // 2 + 1
        
        # Third conv: kernel=3, stride=2, padding=0
        height = (height - 3) // 2 + 1
        width = (width - 3) // 2 + 1
        
        self.flatten_height = height
        self.flatten_width = width
        self.lin_features_len = height * width * self.layers[2]
    
    def _build_encoder(self):
        """Build the encoder network."""
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_shape[2], self.layers[0], kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.layers[0], self.layers[1], kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.layers[1], self.layers[2], kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.lin_features_len, self.embedding_size)
        )
    
    def _build_decoder(self):
        """Build the decoder network."""
        # Calculate output padding for proper upsampling
        out_pad_1 = 1 if self.input_shape[0] // 2 // 2 % 2 == 0 else 0
        out_pad_2 = 1 if self.input_shape[0] // 2 % 2 == 0 else 0
        out_pad_3 = 1 if self.input_shape[0] % 2 == 0 else 0
        
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.lin_features_len),
            Unflatten((self.layers[2], self.flatten_height, self.flatten_width)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.layers[2], self.layers[1], kernel_size=3, stride=2, 
                              padding=0, output_padding=out_pad_1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.layers[1], self.layers[0], kernel_size=5, stride=2, 
                              padding=2, output_padding=out_pad_2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.layers[0], self.input_shape[2], kernel_size=5, stride=2, 
                              padding=2, output_padding=out_pad_3)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing (encoded_features, reconstructed_input)
        """
        x_encoded = self.encoder(x)
        reconstructed = self.decoder(x_encoded)
        return x_encoded, reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input data to embedding space.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)
            
        Returns
        -------
        torch.Tensor
            Encoded features of shape (batch_size, embedding_size)
        """
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode embedding back to original space.
        
        Parameters
        ----------
        x : torch.Tensor
            Encoded features of shape (batch_size, embedding_size)
            
        Returns
        -------
        torch.Tensor
            Reconstructed input of shape (batch_size, channels, height, width)
        """
        return self.decoder(x)
    
    def get_embedding_size(self) -> int:
        """Get the size of the embedding layer."""
        return self.embedding_size
    
    def get_image_shape(self) -> Tuple[int, int, int]:
        """Get the input image shape."""
        return self.input_shape
    
    def get_image_height(self) -> int:
        """Get the input image height."""
        return self.input_shape[0]

    def get_image_width(self) -> int:
        """Get the input image width."""
        return self.input_shape[1]

    def get_channels(self) -> int:
        """Get the number of input channels."""
        return self.input_shape[2]
    
    def get_model_info(self) -> dict:
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_shape': self.input_shape,
            'embedding_size': self.embedding_size,
            'layers': self.layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'flatten_dimensions': (self.flatten_height, self.flatten_width),
            'linear_features': self.lin_features_len
        }


class Unflatten(nn.Module):
    """
    Custom module to reshape flattened tensors back to multi-dimensional tensors.
    
    This is used in the decoder to reshape the linear layer output back to
    the appropriate spatial dimensions for transposed convolutions.
    
    Parameters
    ----------
    shape : Tuple[int, ...]
        The target shape to reshape to (excluding batch dimension)
    """
    
    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape input tensor to the specified shape.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor to reshape
            
        Returns
        -------
        torch.Tensor
            Reshaped tensor with shape (batch_size, *self.shape)
        """
        return x.view(-1, *self.shape)