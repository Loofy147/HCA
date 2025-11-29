from hca.trainer import Trainer, TrainingConfig
from hca.data import CharDataset

# Configuration
epochs = 10
data_file = "data.txt"

# Initialize dataset and trainer
config = TrainingConfig()
dataset = CharDataset(data_file, config.seq_len)
trainer = Trainer(config, dataset)

# Run the training
trainer.train(epochs)
