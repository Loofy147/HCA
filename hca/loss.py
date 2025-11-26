import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class HCALoss(nn.Module):
    """
    The custom loss function for the HCA model, incorporating a Divergence Penalty.

    This loss function combines a standard task-based loss (Cross-Entropy for language
    modeling) with a "Lateral Repulsion Loss" designed to encourage attention head
    specialization. The divergence penalty is calculated as the mean cosine similarity
    between the attention maps of all head pairs in the first layer, encouraging them
    to learn orthogonal (different) patterns.
    """
    def __init__(self, divergence_weight: float = 0.1):
        """
        Initializes the HCALoss.

        Args:
            divergence_weight (float): The coefficient (lambda) to scale the divergence
                penalty before adding it to the task loss.
        """
        super().__init__()
        self.task_criterion = nn.CrossEntropyLoss()
        self.div_weight = divergence_weight

    def _divergence_penalty(self, attns: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculates the divergence penalty (Semantic Orthogonality).

        This penalty aims to minimize the cosine similarity between the attention maps
        of different heads in the first layer, forcing them to specialize.

        Args:
            attns (List[torch.Tensor]): A list of attention weight tensors from the
                transformer, where each element has shape (Batch, Heads, Seq, Seq).

        Returns:
            torch.Tensor: A scalar tensor representing the total divergence penalty.
        """
        # We only apply the penalty to the first layer, as it's the most critical
        # for foundational pattern divergence.
        layer0_attn = attns[0]
        B, H, S, _ = layer0_attn.shape

        # Flatten the attention maps to vectors for cosine similarity calculation
        flat_maps = layer0_attn.view(B, H, -1)

        # In batches, calculate cosine similarity between all pairs of heads (i, j) where i < j
        head_indices = torch.arange(H, device=flat_maps.device)
        pairs = torch.combinations(head_indices, r=2)

        head1 = flat_maps[:, pairs[:, 0], :] # Shape: [B, num_pairs, S*S]
        head2 = flat_maps[:, pairs[:, 1], :] # Shape: [B, num_pairs, S*S]

        # Cosine similarity is calculated along the last dimension (the flattened map)
        # The result has shape [B, num_pairs], we then take the mean over all pairs and batches.
        sim = F.cosine_similarity(head1, head2, dim=-1)

        # The penalty is the mean similarity. We want to minimize this.
        return torch.mean(sim)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, attns_final: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the total loss for the HCA model.

        Args:
            logits (torch.Tensor): The model's output logits, shape (Batch, SeqLen, VocabSize).
            targets (torch.Tensor): The ground truth token IDs, shape (Batch, SeqLen).
            attns_final (List[torch.Tensor]): The attention weights from the final pass,
                used to calculate the divergence penalty.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The total combined loss.
                - The task-specific loss.
                - The divergence penalty loss.
        """
        # 1. Standard Prediction Loss
        # Reshape for CrossEntropyLoss which expects (N, C) and (N)
        task_loss = self.task_criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        # 2. HCA Divergence Penalty
        div_loss = self._divergence_penalty(attns_final)

        # 3. Total Loss
        total_loss = task_loss + (self.div_weight * div_loss)
        return total_loss, task_loss, div_loss
