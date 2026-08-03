"""Linear probe module for mid-layer representation decoding."""

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Linear probe head for decoding entity information from hidden states.
    
    This module takes hidden states from a specific layer and projects them
    to the candidate entity space for classification.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_entities: int,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize the linear probe.
        
        Args:
            hidden_size: Dimension of the hidden state (residual stream)
            num_entities: Number of entities in the candidate set |A|
            bias: Whether to use bias in the linear layer
            dropout: Dropout probability before projection
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_entities = num_entities
        
        # Dropout layer (optional)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Linear projection from hidden size to entity vocabulary
        self.projection = nn.Linear(hidden_size, num_entities, bias=bias)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize probe weights with Xavier initialization."""
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the probe.
        
        Args:
            hidden_states: Tensor of shape (batch_size, hidden_size)
                          containing hidden states at the anchor position
            
        Returns:
            Logits tensor of shape (batch_size, num_entities)
        """
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Project to entity space
        logits = self.projection(hidden_states)
        
        return logits
    
    @property
    def weight(self) -> torch.Tensor:
        """Get the probe weight matrix W_p."""
        return self.projection.weight
    
    @property
    def bias(self) -> torch.Tensor:
        """Get the probe bias vector b_p."""
        return self.projection.bias
    
    def get_probe_matrix(self) -> torch.Tensor:
        """
        Get the full probe matrix (weights + bias if present).
        
        Returns:
            Dictionary containing weight and bias tensors
        """
        result = {"weight": self.projection.weight.detach().cpu()}
        if self.projection.bias is not None:
            result["bias"] = self.projection.bias.detach().cpu()
        return result


def create_probe(
    hidden_size: int,
    num_entities: int,
    bias: bool = True,
    dropout: float = 0.0,
) -> LinearProbe:
    """
    Factory function to create a LinearProbe instance.
    
    Args:
        hidden_size: Hidden state dimension
        num_entities: Number of entities in candidate set
        bias: Whether to use bias
        dropout: Dropout probability
        
    Returns:
        LinearProbe instance
    """
    return LinearProbe(
        hidden_size=hidden_size,
        num_entities=num_entities,
        bias=bias,
        dropout=dropout,
    )
