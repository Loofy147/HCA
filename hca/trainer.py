import torch
from dataclasses import dataclass
from .transformer import HCATransformer
from .loss import HCALoss

@dataclass
class TrainingConfig:
    """
    Configuration for the HCA training process.
    """
    vocab_size: int = 1000
    dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    seq_len: int = 20
    max_seq_len: int = 512
    batch_size: int = 32
    learning_rate: float = 1e-3
    divergence_weight: float = 0.5
    sparsity_weight: float = 0.01

class Trainer:
    """
    The Trainer class encapsulates the training logic for the HCATransformer.
    """
    def __init__(self, config: TrainingConfig):
        """
        Initializes the Trainer.

        Args:
            config (TrainingConfig): The training configuration.
        """
        self.config = config
        self.model = HCATransformer(
            vocab_size=config.vocab_size,
            dim=config.dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_seq_len=config.max_seq_len
        )
        self.loss_fn = HCALoss(
            divergence_weight=config.divergence_weight,
            sparsity_weight=config.sparsity_weight
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )

    def train(self, epochs: int):
        """
        Runs the training loop for a specified number of epochs.

        Args:
            epochs (int): The number of epochs to train for.
        """
        for epoch in range(epochs):
            # Dummy Data
            inputs = torch.randint(0, self.config.vocab_size, (self.config.batch_size, self.config.seq_len))
            targets = torch.randint(0, self.config.vocab_size, (self.config.batch_size, self.config.seq_len))

            # Training Step
            self.optimizer.zero_grad()
            logits, attn_pass1, attn_pass2, values_pass2, pred_k = self.model(inputs)

            # Calculate Loss
            loss, task_l, div_l, sparsity_l = self.loss_fn(logits, targets, attn_pass2, values_pass2, pred_k)

            loss.backward()
            self.optimizer.step()

            avg_k = pred_k.mean().item()
            print(f"Epoch {epoch+1}/{epochs} | Task Loss: {task_l.item():.4f} | Div Loss: {div_l.item():.4f} | Sparsity Loss: {sparsity_l.item():.4f} | Avg k: {avg_k:.2f}")
