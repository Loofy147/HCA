import torch
import torch.nn as nn
from .attention import HCAAttention
from .controller import CyclicController
from typing import List, Tuple, Optional

class HCATransformer(nn.Module):
    """
    The main HCA Transformer model, orchestrating the 2-pass system.

    This model processes input sequences in two distinct phases:
    1.  **Fast Pass (System 1):** A standard forward pass where attention heads are
        constrained by a soft-tree mask, forcing them to learn local syntax and phrases.
    2.  **Refinement Pass (System 2):** The global context from the first pass is used
        to generate a feedback signal, which is injected back into the first layer. This
        allows the model to re-attend to the input, overriding the local constraints to
        resolve long-range dependencies and refine semantic understanding.
    """
    def __init__(self, vocab_size: int, dim: int, num_heads: int, num_layers: int, max_seq_len: int = 1024):
        """
        Initializes the HCATransformer.

        Args:
            vocab_size (int): The size of the vocabulary.
            dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of transformer layers.
            max_seq_len (int): The maximum sequence length for the learned positional encoding.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, dim))

        # Layers
        self.layers = nn.ModuleList([
            HCAAttention(dim, num_heads) for _ in range(num_layers)
        ])

        # Feed Forward Networks (Standard per layer)
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.ReLU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])

        # The Feedback Brain
        self.feedback_controller = CyclicController(dim)

        # Final Head
        self.lm_head = nn.Linear(dim, vocab_size)

    def _forward_pass(self, x: torch.Tensor, feedback: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Executes a single forward pass through the transformer layers.

        Args:
            x (torch.Tensor): The input tensor for the pass.
            feedback (Optional[torch.Tensor]): The feedback signal to inject into the first layer.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]: A tuple containing:
                - The final hidden state of the pass.
                - A list of attention weights from each layer.
                - A list of value tensors from each layer.
        """
        attentions = []
        values = []
        for i, (attn_layer, ffn, norm) in enumerate(zip(self.layers, self.ffns, self.norms)):
            # Inject feedback only into Layer 0 (The Roots)
            current_feedback = feedback if i == 0 else None

            # Attention Sub-layer
            residual = x
            x, attn_w, v = attn_layer(x, feedback_signal=current_feedback)
            x = norm(x + residual)
            attentions.append(attn_w)
            values.append(v)

            # FFN Sub-layer
            residual = x
            x = ffn(x)
            x = norm(x + residual)

        return x, attentions, values

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        The main forward logic implementing the 2-pass system.

        Args:
            x (torch.Tensor): The input token IDs, shape (Batch, SeqLen).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: A tuple containing:
                - The final logits for prediction, shape (Batch, SeqLen, VocabSize).
                - The attention weights from the first (local) pass.
                - The attention weights from the second (refined) pass.
                - The value tensors from the second (refined) pass.
        """
        B, L = x.shape
        x_emb = self.embedding(x) + self.pos_enc[:, :L, :]

        # --- PHASE 1: THE LOCAL PASS (Tree Construction) ---
        # Heads are constrained. They build local phrases.
        hidden_state_1, attns_1, _ = self._forward_pass(x_emb, feedback=None) # values from pass 1 are ignored

        # --- PHASE 2: THE CYCLIC STEP (Reasoning) ---
        # 1. Extract Global Context
        feedback_vector = self.feedback_controller(hidden_state_1)

        # 2. Re-run with "Sunlight"
        # The feedback vector enters Layer 0, modifying Queries to break local constraints.
        # Note: We re-use the original embedding `x_emb` as the input to the second pass.
        hidden_state_2, attns_2, values_2 = self._forward_pass(x_emb, feedback=feedback_vector)

        # Output prediction based on the Refined State
        logits = self.lm_head(hidden_state_2)

        return logits, attns_1, attns_2, values_2
