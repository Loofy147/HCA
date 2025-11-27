import torch
import torch.nn as nn

class CyclicController(nn.Module):
    """
    The "brain" of the cyclic feedback loop.

    This module takes a pre-computed global context vector and processes it to
    generate a feedback signal.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.Tanh()
        )

    def forward(self, global_context: torch.Tensor) -> torch.Tensor:
        """
        Generates the feedback signal from the global context.

        Args:
            global_context (torch.Tensor): The global context vector, shape (B, D).

        Returns:
            torch.Tensor: The processed feedback signal, shape (B, D).
        """
        return self.net(global_context)


class SparsityPredictor(nn.Module):
    """
    Predicts the number of tokens (k) to attend to for adaptive sparsity.

    This network takes a global context vector and outputs a scalar value `k`
    for each item in the batch.
    """
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.max_k = max_seq_len
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, global_context: torch.Tensor) -> torch.Tensor:
        """
        Predicts k based on the global context.

        Args:
            global_context (torch.Tensor): The aggregated context vector, shape (B, D).

        Returns:
            torch.Tensor: A tensor of k values, shape (B, 1).
        """
        k_scale = self.net(global_context)
        predicted_k = 1 + (k_scale * (self.max_k - 1))
        return predicted_k
