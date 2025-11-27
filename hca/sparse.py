import torch
from typing import Union

def top_k_masking(scores: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Applies a top-k mask to each item in the batch dimension of the scores tensor.

    This function handles a batch of k values, applying the corresponding k
    to mask the attention scores of each item in the batch.

    Args:
        scores (torch.Tensor): The attention scores, shape (B, H, L, L).
        k (torch.Tensor): The k values for each batch item, shape (B, 1).

    Returns:
        torch.Tensor: The scores tensor with the top-k mask applied per item.
    """
    batch_size = scores.size(0)
    seq_len = scores.size(-1)

    # Final tensor will be built here
    masked_scores = torch.full_like(scores, float('-inf'))

    # Clamp and convert k to a list of integers
    k_values = torch.clamp(k.round().long(), 1, seq_len).squeeze(-1).tolist()

    # Iterate over each item in the batch
    for i in range(batch_size):
        item_scores = scores[i]       # Shape (H, L, L)
        item_k = k_values[i]          # Scalar int

        # Find top values and indices for this specific item
        top_k_values, top_k_indices = item_scores.topk(k=item_k, dim=-1)

        # Use scatter to place the top values into the correct positions
        masked_scores[i].scatter_(dim=-1, index=top_k_indices, src=top_k_values)

    return masked_scores
