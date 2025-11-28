import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class HCALoss(nn.Module):
    """
    The custom loss for HCA, combining task loss, divergence, and sparsity penalties.
    """
    def __init__(self, divergence_weight: float = 0.1, sparsity_weight: float = 0.01):
        """
        Initializes the HCALoss.

        Args:
            divergence_weight (float): Coefficient for the head divergence penalty.
            sparsity_weight (float): Coefficient for the sparsity regularization penalty.
        """
        super().__init__()
        self.task_criterion = nn.CrossEntropyLoss()
        self.div_weight = divergence_weight
        self.sparsity_weight = sparsity_weight

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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, attns_final: List[torch.Tensor], values_final: List[torch.Tensor], predicted_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the total loss for the HCA model.

        Args:
            logits (torch.Tensor): The model's output logits.
            targets (torch.Tensor): The ground truth token IDs.
            attns_final (List[torch.Tensor]): The attention weights from the final pass.
            values_final (List[torch.Tensor]): The value tensors from the final pass.
            predicted_k (torch.Tensor): The k values predicted by the sparsity predictor.

        Returns:
            A tuple containing: (total_loss, task_loss, div_loss, sparsity_loss)
        """
        # 1. Task Loss
        task_loss = self.task_criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        # 2. Divergence Penalty
        div_loss = self._divergence_penalty(attns_final, values_final)

        # 3. Sparsity Regularization
        # We penalize the model for using a large k. The mean of k across the
        # batch is used as the penalty.
        sparsity_loss = torch.mean(predicted_k)

        # 4. Total Loss
        total_loss = task_loss + (self.div_weight * div_loss) + (self.sparsity_weight * sparsity_loss)
        return total_loss, task_loss, div_loss, sparsity_loss
