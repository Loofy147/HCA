from hca.trainer import Trainer, TrainingConfig

# Configuration
epochs = 10

# Initialize and run the trainer
config = TrainingConfig()
trainer = Trainer(config)
trainer.train(epochs)
