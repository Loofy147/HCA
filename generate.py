import torch
import argparse
from hca.trainer import Trainer, TrainingConfig
from hca.data import CharDataset

def generate(prompt: str, num_chars: int):
    """
    Generates text using a trained HCA model.

    Args:
        prompt (str): The initial text to seed the generation.
        num_chars (int): The number of characters to generate.
    """
    config = TrainingConfig()
    dataset = CharDataset("data.txt", config.seq_len)
    trainer = Trainer(config, dataset)
    trainer.load_checkpoint()

    model = trainer.model
    model.eval()

    print(f"Generating {num_chars} characters from prompt: '{prompt}'")

    # Encode the prompt
    encoded_prompt = [dataset.stoi[char] for char in prompt]
    input_tensor = torch.tensor(encoded_prompt, dtype=torch.long).unsqueeze(0)

    generated_text = prompt
    for _ in range(num_chars):
        # Get the model's prediction
        with torch.no_grad():
            logits, _, _, _, _ = model(input_tensor)

        # Get the last token's logits and apply softmax to get probabilities
        last_logits = logits[:, -1, :]
        probs = torch.softmax(last_logits, dim=-1)

        # Sample from the distribution
        next_char_idx = torch.multinomial(probs, num_samples=1).item()
        next_char = dataset.itos[next_char_idx]

        generated_text += next_char

        # Update the input tensor
        next_input = torch.tensor([[next_char_idx]], dtype=torch.long)
        input_tensor = torch.cat([input_tensor, next_input], dim=1)

    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with a trained HCA model.")
    parser.add_argument("--prompt", type=str, default="Hello", help="The prompt to start generation from.")
    parser.add_argument("--num_chars", type=int, default=100, help="The number of characters to generate.")
    args = parser.parse_args()

    generate(args.prompt, args.num_chars)
