# Hierarchical Cyclic Arborization (HCA)

This repository contains the PyTorch implementation of the Hierarchical Cyclic Arborization (HCA) theory, a novel AI architecture designed as an alternative to the standard Transformer model.

HCA is founded on the idea that intelligence is not a "Dense Correlation Engine" but rather a "Dynamic Tree" that is Hierarchical, Cyclic, and Arborizing. It is built upon three core pillars:

1.  **Divergent Evolution:** A mechanism that forces attention heads to specialize by penalizing similarity between them, promoting a diverse set of feature detectors. This is implemented via a "Lateral Repulsion Loss."
2.  **Soft-Tree Constraint:** A bias that encourages attention heads to focus on local information in their initial pass. This helps the model build up an understanding of local syntax and structure before attempting to resolve long-range dependencies.
3.  **Cyclic Feedback:** A two-pass system where the global context from a first, locally-focused pass is injected back into the early layers for a second, refinement pass. This allows the model to use high-level understanding to re-examine and reinterpret low-level details, effectively solving long-range dependencies.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd hca
pip install -r requirements.txt
```

## Usage

### Training

The training process is encapsulated within a `Trainer` class. To start training the model with a default configuration and dummy data, run:

```bash
python3 train.py
```

The training script will print the loss and other metrics for each epoch. Checkpoints will be saved to the `checkpoints/` directory by default.

### Generation

To generate text from a trained model, use the `generate.py` script:

```bash
python3 generate.py --prompt "Your prompt here" --num_chars 200
```

This will load the latest checkpoint and generate 200 characters of new text based on your prompt.

### Testing

The project includes a suite of unit tests to verify the core components of the HCA model. To run the tests, execute:

```bash
python3 test_hca.py
```
