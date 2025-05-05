import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from faircause.utils.neural_learning import (
    TwoLayerArchitecture,
    causal_loss,
    train_w_es,
    pred_nn_proba
)
from faircause.faircause import FairCause


class TestTwoLayerArchitecture(unittest.TestCase):
    """Test the neural network architecture class."""
    
    def test_init(self):
        """Test initialization of the neural network architecture."""
        model = TwoLayerArchitecture(input_size=10, hidden_size=16, output_size=1)
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.layer1.in_features, 10)
        self.assertEqual(model.layer1.out_features, 16)
        self.assertEqual(model.layer2.in_features, 16)
        self.assertEqual(model.layer2.out_features, 16)
        self.assertEqual(model.output_layer.in_features, 16)
        self.assertEqual(model.output_layer.out_features, 1)
    
    def test_forward(self):
        """Test forward pass of the neural network."""
        model = TwoLayerArchitecture(input_size=3, hidden_size=4, output_size=1)
        x = torch.randn(5, 3)  # 5 samples with 3 features each
        output = model(x)
        self.assertEqual(output.shape, (5, 1))

class TestCausalLoss(unittest.TestCase):
    """Test the causal loss function."""
    
    def test_causal_loss_classification(self):
        """Test causal loss computation for classification."""
        # Create dummy inputs
        pred = torch.randn(10, 1)
        pred0 = torch.randn(10, 1)
        pred1 = torch.randn(10, 1)
        X = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]], dtype=torch.float)
        Z = torch.randn(10, 2)
        W = torch.randn(10, 3)
        Y = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]], dtype=torch.float)
        px_z = torch.tensor([0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.5, 0.5, 0.1, 0.9], dtype=torch.float)
        
        # Compute loss
        loss = causal_loss(
            pred, pred0, pred1, X, Z, W, Y, px_z,
            eta_de=0, eta_ie=0, eta_se_x1=0, eta_se_x0=0,
            relu_eps=False, eps=0.005, task_type='classification'
        )
        
        # Check if loss is a scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar tensor
        self.assertGreaterEqual(loss.item(), 0)  # Loss should be non-negative
    
    def test_causal_loss_regression(self):
        """Test causal loss computation for regression."""
        # Create dummy inputs
        pred = torch.randn(10, 1)
        pred0 = torch.randn(10, 1)
        pred1 = torch.randn(10, 1)
        X = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]], dtype=torch.float)
        Z = torch.randn(10, 2)
        W = torch.randn(10, 3)
        Y = torch.randn(10, 1)
        px_z = torch.tensor([0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.5, 0.5, 0.1, 0.9], dtype=torch.float)
        
        # Compute loss
        loss = causal_loss(
            pred, pred0, pred1, X, Z, W, Y, px_z,
            eta_de=0, eta_ie=0, eta_se_x1=0, eta_se_x0=0,
            relu_eps=False, eps=0.005, task_type='regression'
        )
        
        # Check if loss is a scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar tensor
        self.assertGreaterEqual(loss.item(), 0)  # Loss should be non-negative
    
    def test_relu_eps(self):
        """Test causal loss with ReLU epsilon option enabled."""
        # Create inputs with known values
        pred = torch.zeros(10, 1)
        pred0 = torch.zeros(10, 1)
        pred1 = torch.zeros(10, 1)
        X = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]], dtype=torch.float)
        Z = torch.zeros(10, 2)
        W = torch.zeros(10, 3)
        Y = torch.zeros(10, 1)
        px_z = torch.tensor([0.5] * 10, dtype=torch.float)
        
        # With relu_eps=True and values below epsilon threshold, loss should be zero
        loss_with_relu = causal_loss(
            pred, pred0, pred1, X, Z, W, Y, px_z,
            eta_de=0, eta_ie=0, eta_se_x1=0, eta_se_x0=0,
            relu_eps=True, eps=0.01, task_type='regression'
        )
        
        # Loss should be zero since all deviations are 0 and below epsilon
        self.assertEqual(loss_with_relu.item(), 0.0)

class TestTrainingPrediction(unittest.TestCase):
    """Test the training and prediction functions."""
    
    def setUp(self):
        """Set up common test data."""
        # Create a small synthetic dataset
        np.random.seed(42)
        n = 30
        self.data = pd.DataFrame({
            'X': np.random.randint(0, 2, n),
            'Z': np.random.normal(0, 1, n),
            'W': np.random.normal(0, 1, n),
            'Y': np.random.randint(0, 2, n)
        })
        self.eval_data = pd.DataFrame({
            'X': np.random.randint(0, 2, 10),
            'Z': np.random.normal(0, 1, 10),
            'W': np.random.normal(0, 1, 10),
            'Y': np.random.randint(0, 2, 10)
        })
    
    def test_train_w_es(self):
        """Test the training function with early stopping."""
        # Train with minimal epochs and patience for testing
        model = train_w_es(
            train_data=self.data,
            eval_data=self.eval_data,
            x_col='X',
            z_cols=['Z'],
            w_cols=['W'],
            y_col='Y',
            lmbd=0.1,
            lr=0.01,
            epochs=5,
            patience=2,
            max_restarts=1,
            eval_size=5,
            verbose=False,
            batch_size=10,
            seed=42
        )
        
        # Check if model is returned
        self.assertIsInstance(model, TwoLayerArchitecture)
    
    def test_pred_nn_proba(self):
        """Test the prediction function."""
        # Create a simple model
        model = TwoLayerArchitecture(input_size=3, hidden_size=4, output_size=1)
        
        # Test prediction function
        predictions = pred_nn_proba(model, self.data[['X', 'Z', 'W']], task_type='classification')
        
        # Check prediction shape and range
        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(all(0 <= p <= 1 for p in predictions.flatten()))

if __name__ == '__main__':
    unittest.main()