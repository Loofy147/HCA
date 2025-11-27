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

    def _divergence_penalty(self, attns: List[torch.Tensor], values: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculates the enhanced divergence penalty.

        This penalty combines two orthogonality constraints:
        1.  **Attention Patterns**: Minimizes cosine similarity between attention maps.
        2.  **Value Spaces**: Minimizes cosine similarity between value head outputs.

        Args:
            attns (List[torch.Tensor]): List of attention weights from the transformer.
            values (List[torch.Tensor]): List of value tensors from the transformer.

        Returns:
            torch.Tensor: A scalar tensor representing the total divergence penalty.
        """
        # We only apply the penalty to the first layer.
        layer0_attn = attns[0]
        layer0_values = values[0]
        B, H, S, _ = layer0_attn.shape

        # --- 1. Attention Pattern Loss ---
        flat_maps = layer0_attn.view(B, H, -1)
        head_indices = torch.arange(H, device=flat_maps.device)
        pairs = torch.combinations(head_indices, r=2)

        attn_head1 = flat_maps[:, pairs[:, 0], :]
        attn_head2 = flat_maps[:, pairs[:, 1], :]
        pattern_sim = F.cosine_similarity(attn_head1, attn_head2, dim=-1)
        pattern_loss = torch.mean(pattern_sim)

        # --- 2. Value Space Loss ---
        # Reshape values to [B, H, L*D_h]
        flat_values = layer0_values.reshape(B, H, -1)

        value_head1 = flat_values[:, pairs[:, 0], :]
        value_head2 = flat_values[:, pairs[:, 1], :]
        value_sim = F.cosine_similarity(value_head1, value_head2, dim=-1)
        value_loss = torch.mean(value_sim)

        # Combine the two losses. The roadmap implies a 1:1 weighting for now.
        return pattern_loss + value_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, attns_final: List[torch.Tensor], values_final: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the total loss for the HCA model.

        Args:
            logits (torch.Tensor): The model's output logits, shape (Batch, SeqLen, VocabSize).
            targets (torch.Tensor): The ground truth token IDs, shape (Batch, SeqLen).
            attns_final (List[torch.Tensor]): The attention weights from the final pass.
            values_final (List[torch.Tensor]): The value tensors from the final pass.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The total combined loss.
                - The task-specific loss.
                - The divergence penalty loss.
        """
        # 1. Standard Prediction Loss
        task_loss = self.task_criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        # 2. HCA Divergence Penalty
        div_loss = self._divergence_penalty(attns_final, values_final)

        # 3. Total Loss
        total_loss = task_loss + (self.div_weight * div_loss)
        return total_loss, task_loss, div_loss
