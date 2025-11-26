import torch
import torch.nn as nn

class CyclicController(nn.Module):
    """
    The "brain" of the cyclic feedback loop.

    This module takes the final hidden state of the transformer, pools it to create a
    global context vector, and then processes this vector through a small neural network
    to generate a feedback signal. This signal is then injected back into the first
    attention layer to guide the "second pass" of reasoning.
    """
    def __init__(self, dim: int):
        """
        Initializes the CyclicController.

        Args:
            dim (int): The embedding dimension, which is the size of the feedback signal.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.Tanh()  # Bound signal to avoid exploding gradients
        )

    def forward(self, final_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Generates the feedback signal from the final hidden state.

        Args:
            final_hidden_state (torch.Tensor): The output of the top transformer layer,
                of shape (Batch, SeqLen, Dim).

        Returns:
            torch.Tensor: The processed feedback signal of shape (Batch, Dim).
        """
        # Pool the sequence to get a summary vector. For a causal model, the last
        # token's hidden state contains the accumulated history of the sequence.
        global_context = final_hidden_state[:, -1, :]
        return self.net(global_context)
