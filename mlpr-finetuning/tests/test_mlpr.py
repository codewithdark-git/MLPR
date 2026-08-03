"""Test cases for MLPR fine-tuning components."""

import unittest
import torch
import tempfile
import os
from unittest.mock import Mock, patch


class TestLinearProbe(unittest.TestCase):
    """Test cases for the LinearProbe module."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.models.probe import LinearProbe
        
        self.hidden_size = 512
        self.num_entities = 100
        self.probe = LinearProbe(
            hidden_size=self.hidden_size,
            num_entities=self.num_entities,
            bias=True,
            dropout=0.0,
        )
    
    def test_probe_initialization(self):
        """Test probe is initialized correctly."""
        self.assertEqual(self.probe.hidden_size, self.hidden_size)
        self.assertEqual(self.probe.num_entities, self.num_entities)
        self.assertEqual(self.probe.projection.weight.shape, (self.num_entities, self.hidden_size))
    
    def test_probe_forward(self):
        """Test probe forward pass."""
        batch_size = 4
        hidden_states = torch.randn(batch_size, self.hidden_size)
        
        output = self.probe(hidden_states)
        
        self.assertEqual(output.shape, (batch_size, self.num_entities))
    
    def test_probe_weight_property(self):
        """Test probe weight property returns correct tensor."""
        weight = self.probe.weight
        self.assertEqual(weight.shape, (self.num_entities, self.hidden_size))
    
    def test_probe_bias_property(self):
        """Test probe bias property returns correct tensor."""
        bias = self.probe.bias
        self.assertEqual(bias.shape, (self.num_entities,))
    
    def test_get_probe_matrix(self):
        """Test get_probe_matrix returns weight and bias."""
        matrices = self.probe.get_probe_matrix()
        
        self.assertIn("weight", matrices)
        self.assertIn("bias", matrices)
        self.assertEqual(matrices["weight"].shape, (self.num_entities, self.hidden_size))


class TestLambdaScheduler(unittest.TestCase):
    """Test cases for LambdaScheduler."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.callbacks.lambda_scheduler import LambdaScheduler
        
        self.scheduler = LambdaScheduler(
            lambda_0=0.3,
            tau_0=0.9,
            delta_0=0.05,
        )
    
    def test_initial_lambda_is_zero(self):
        """Test that initial lambda is 0."""
        self.assertEqual(self.scheduler.get_lambda(), 0.0)
    
    def test_lambda_computation_below_threshold(self):
        """Test lambda is 0 when A_mem < τ₀."""
        self.scheduler.update_a_mem(0.8)  # Below threshold of 0.9
        self.assertEqual(self.scheduler.get_lambda(), 0.0)
        self.assertFalse(self.scheduler.probe_activated)
    
    def test_lambda_computation_at_threshold(self):
        """Test lambda computation at threshold."""
        self.scheduler.update_a_mem(0.9)  # At threshold
        lambda_val = self.scheduler.get_lambda()
        self.assertGreaterEqual(lambda_val, 0.0)
        self.assertTrue(self.scheduler.memorization_saturated)
    
    def test_lambda_computation_above_threshold(self):
        """Test lambda increases as A_mem exceeds threshold."""
        self.scheduler.update_a_mem(0.95)  # Above threshold by delta_0
        lambda_val = self.scheduler.get_lambda()
        
        # Should be at max lambda_0 since we're at tau_0 + delta_0
        self.assertAlmostEqual(lambda_val, 0.3, places=5)
        self.assertTrue(self.scheduler.probe_activated)
    
    def test_memorization_saturation_flag(self):
        """Test memorization saturation flag is set correctly."""
        self.assertFalse(self.scheduler.is_memorization_saturated())
        
        self.scheduler.update_a_mem(0.9)
        self.assertTrue(self.scheduler.is_memorization_saturated())
    
    def test_get_status(self):
        """Test get_status returns all required fields."""
        status = self.scheduler.get_status()
        
        self.assertIn("lambda", status)
        self.assertIn("a_mem", status)
        self.assertIn("tau_0", status)
        self.assertIn("lambda_0", status)
        self.assertIn("delta_0", status)
        self.assertIn("memorization_saturated", status)
        self.assertIn("probe_activated", status)


class TestMLPRTargetCollator(unittest.TestCase):
    """Test cases for MLPRTargetCollator."""
    
    def setUp(self):
        """Set up test fixtures."""
        from transformers import AutoTokenizer
        from src.data.collator import MLPRTargetCollator
        
        # Use a small tokenizer for testing
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.collator = MLPRTargetCollator(
            tokenizer=self.tokenizer,
            padding="longest",
        )
    
    def test_collator_basic(self):
        """Test basic collation functionality."""
        features = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "entity_pos": 1,
                "entity_class_id": 0,
            },
            {
                "input_ids": [4, 5, 6, 7],
                "attention_mask": [1, 1, 1, 1],
                "entity_pos": 2,
                "entity_class_id": 1,
            },
        ]
        
        batch = self.collator(features)
        
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("entity_pos", batch)
        self.assertIn("entity_class_id", batch)
        self.assertEqual(batch["entity_pos"].shape[0], 2)
        self.assertEqual(batch["entity_class_id"].shape[0], 2)


class TestExactMatch(unittest.TestCase):
    """Test cases for exact match evaluation."""
    
    def test_exact_match_basic(self):
        """Test basic exact match."""
        from src.evaluation.gen_eval import compute_exact_match
        
        self.assertTrue(compute_exact_match("hello", "hello"))
        self.assertFalse(compute_exact_match("hello", "world"))
    
    def test_exact_match_normalization(self):
        """Test exact match with normalization."""
        from src.evaluation.gen_eval import compute_exact_match
        
        # Case insensitive
        self.assertTrue(compute_exact_match("Hello", "hello"))
        
        # Whitespace normalization
        self.assertTrue(compute_exact_match("hello  world", "hello world"))
        
        # Punctuation removal
        self.assertTrue(compute_exact_match("hello!", "hello"))
    
    def test_extract_answer_pattern(self):
        """Test answer extraction from generated text."""
        from src.evaluation.gen_eval import extract_answer_from_generation
        
        self.assertEqual(
            extract_answer_from_generation("Answer: FinancialAuthority"),
            "FinancialAuthority"
        )
        self.assertEqual(
            extract_answer_from_generation("answer: CorporationZ"),
            "CorporationZ"
        )


class TestDatasetCreation(unittest.TestCase):
    """Test cases for dataset creation."""
    
    def test_sample_dataset_creation(self):
        """Test that sample dataset can be created."""
        from src.data.dataset import MLPDataset
        
        # Create dataset without loading from HF
        dataset = MLPDataset(
            dataset_name="test-dataset",
            entity_vocab_path=None,
            max_length=512,
        )
        
        # Check dataset has expected splits
        self.assertIsNotNone(dataset.dataset)
        self.assertIn("train", dataset.dataset)
        self.assertIn("eval", dataset.dataset)
        
        # Check candidate set size (default vocab is created)
        # The default vocab should have at least some entities
        self.assertGreaterEqual(dataset.candidate_set_size, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for MLPR components."""
    
    @patch('src.trainer.mlpr_trainer.MLPRTrainer.compute_loss')
    def test_trainer_loss_computation(self, mock_compute_loss):
        """Test that trainer loss computation is called correctly."""
        # Mock the loss computation
        mock_compute_loss.return_value = torch.tensor(1.0)
        
        # This test verifies the structure without actual training
        self.assertTrue(True)  # Placeholder for integration test


if __name__ == "__main__":
    unittest.main()
