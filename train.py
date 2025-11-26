import torch
from hca.transformer import HCATransformer
from hca.loss import HCALoss

# Configuration
vocab_size = 1000
dim = 128
num_heads = 4
layers = 2
seq_len = 20
max_seq_len = 512 # Set a max sequence length for the model

# Initialize
model = HCATransformer(
    vocab_size=vocab_size,
    dim=dim,
    num_heads=num_heads,
    num_layers=layers,
    max_seq_len=max_seq_len
)
loss_fn = HCALoss(divergence_weight=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Dummy Data
inputs = torch.randint(0, vocab_size, (32, seq_len))
targets = torch.randint(0, vocab_size, (32, seq_len))

# Training Step
optimizer.zero_grad()
logits, attn_pass1, attn_pass2 = model(inputs)

# Calculate Loss (using only the finalized attention for divergence)
loss, task_l, div_l = loss_fn(logits, targets, attn_pass2)

loss.backward()
optimizer.step()

print(f"Task Loss: {task_l.item():.4f} | Divergence Loss: {div_l.item():.4f}")
