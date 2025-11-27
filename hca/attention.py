import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class HCAAttention(nn.Module):
    """
    The core attention mechanism for HCA, implementing Soft-Tree Constraints and Feedback Injection.

    This module performs multi-head scaled dot-product attention with two key modifications
    based on HCA theory:
    1.  **Soft-Tree Constraint:** A penalty is applied to attention scores for distant tokens,
        encouraging heads to initially focus on local relationships.
    2.  **Cyclic Feedback Injection:** An optional `feedback_signal` can be added to the queries,
        allowing a global context to override the local bias and focus on distant tokens.
    """
    def __init__(self, dim: int, num_heads: int, tree_penalty: float = 5.0):
        """
        Initializes the HCAAttention module.

        Args:
            dim (int): The embedding dimension of the input.
            num_heads (int): The number of attention heads.
            tree_penalty (float): The negative bias applied to distant tokens.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.tree_penalty = tree_penalty  # The cost of looking far away (Soft Constraint)

        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def create_soft_tree_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates a soft bias matrix to encourage local attention.

        The mask applies a `tree_penalty` to all token interactions except for the
        current and immediately preceding token, which receive no penalty.

        Args:
            seq_len (int): The length of the sequence.
            device (torch.device): The device to create the tensor on.

        Returns:
            torch.Tensor: A (seq_len, seq_len) tensor with penalties for distant tokens.
        """
        mask = torch.ones((seq_len, seq_len), device=device) * -self.tree_penalty

        # Allow local window (Current + Previous) without penalty
        # This forces the "Sapling" behavior
        for i in range(seq_len):
            mask[i, i] = 0.0          # Self
            if i > 0: mask[i, i-1] = 0.0  # Previous

        return mask

    def forward(self, x: torch.Tensor, feedback_signal: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the attention mechanism.

        Args:
            x (torch.Tensor): The input tensor of shape (Batch, SeqLen, Dim).
            feedback_signal (Optional[torch.Tensor]): The global context signal to be
                injected into the queries, shape (Batch, Dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The output tensor of shape (Batch, SeqLen, Dim).
                - The attention weights of shape (Batch, NumHeads, SeqLen, SeqLen).
                - The value tensor of shape (Batch, NumHeads, SeqLen, HeadDim).
        """
        B, L, D = x.shape

        # 1. Project Q, K, V
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # --- PILLAR 3: CYCLIC FEEDBACK INJECTION ---
        # If "Sunlight" (Feedback) is present, modify the Queries.
        # This allows the Global Context to bias what the head looks for.
        if feedback_signal is not None:
            # feedback_signal: [B, D] -> Reshape to [B, 1, H, D_h] -> Transpose to [B, H, 1, D_h]
            fb = feedback_signal.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            q = q + fb  # Injection: Q_new = Q_old + Global_Context

        # 2. Compute Raw Scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # --- PILLAR 2: TREE CONSTRAINT (Arborization) ---
        # Apply Soft Mask (Penalty for non-local), then Hard Causal Mask
        soft_mask = self.create_soft_tree_mask(L, x.device)
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        scores = scores + soft_mask # Apply Energy Barrier
        scores.masked_fill_(causal_mask, float('-inf')) # Strict Causality

        # 3. Attention Weights
        attn_weights = F.softmax(scores, dim=-1)

        # 4. Output
        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out), attn_weights, v
