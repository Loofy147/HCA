import unittest
import torch
from hca.transformer import HCATransformer

class TestHCATransformer(unittest.TestCase):

    def setUp(self):
        """
        Set up a common HCATransformer model for testing.
        """
        self.vocab_size = 1000
        self.dim = 128
        self.num_heads = 4
        self.layers = 2
        self.max_seq_len = 512
        self.seq_len = 20
        self.batch_size = 32

        self.model = HCATransformer(
            vocab_size=self.vocab_size,
            dim=self.dim,
            num_heads=self.num_heads,
            num_layers=self.layers,
            max_seq_len=self.max_seq_len
        )

    def test_model_initialization(self):
        """
        Tests if the HCATransformer model initializes without errors.
        """
        self.assertIsInstance(self.model, HCATransformer)

    def test_forward_pass(self):
        """
        Tests the forward pass of the HCATransformer model.
        """
        inputs = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        # The model returns: logits, attns_1, attns_2, values_2, predicted_k
        # We only need to check the logits shape for this test.
        logits, _, _, _, _ = self.model(inputs)

        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, self.vocab_size))

if __name__ == '__main__':
    unittest.main()
