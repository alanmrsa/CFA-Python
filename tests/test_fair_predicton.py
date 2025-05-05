import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from faircause.fairprediction import FairPredict
import torch

class TestFairPredict(unittest.TestCase):
    """Test suite for the FairPredict class."""
    
    def setUp(self):
        """Set up common test data."""
        # Create a synthetic dataset
        np.random.seed(42)
        n = 50
        self.data = pd.DataFrame({
            'X': np.random.randint(0, 2, n),
            'Z1': np.random.normal(0, 1, n),
            'Z2': np.random.normal(0, 1, n),
            'W1': np.random.normal(0, 1, n),
            'W2': np.random.normal(0, 1, n),
            'Y': np.random.randint(0, 2, n)
        })
        
        # Parameters for FairPredict
        self.X = 'X'
        self.Z = ['Z1', 'Z2']
        self.W = ['W1', 'W2']
        self.Y = 'Y'
        self.x0 = 0
        self.x1 = 1
        self.BN = 'DE'  # Direct effect in Bayesian Network
        
        # Create a smaller lambda sequence for faster testing
        self.lmbd_seq = [0.1, 1.0]
    
    def test_init(self):
        """Test initialization of FairPredict."""
        # Initialize FairPredict with minimal configuration
        fp = FairPredict(
            data=self.data,
            X=self.X,
            Z=self.Z,
            W=self.W,
            Y=self.Y,
            x0=self.x0,
            x1=self.x1,
            BN=self.BN,
            lmbd_seq=self.lmbd_seq,
            patience=5  # Small value for testing
        )
        
        # Check instance attributes
        self.assertEqual(fp.X, self.X)
        self.assertEqual(fp.Z, self.Z)
        self.assertEqual(fp.W, self.W)
        self.assertEqual(fp.Y, self.Y)
        self.assertEqual(fp.x0, self.x0)
        self.assertEqual(fp.x1, self.x1)
        self.assertEqual(fp.BN, self.BN)
        self.assertEqual(fp.lmbd_seq, self.lmbd_seq)
        self.assertIsNone(fp.y_meas)
        self.assertIsNone(fp.yhat_meas)
        self.assertIsNone(fp.nn_mod)
    
    def test_verify_numeric_input(self):
        """Test input validation function."""
        # Valid input (all numeric)
        fp = FairPredict(
            data=self.data,
            X=self.X,
            Z=self.Z,
            W=self.W,
            Y=self.Y,
            x0=self.x0,
            x1=self.x1,
            lmbd_seq=self.lmbd_seq
        )
        
        # Should not raise exception
        fp.verify_numeric_input(self.data)
        
        # Invalid input with non-numeric column
        invalid_data = self.data.copy()
        invalid_data['X'] = invalid_data['X'].astype(str)
        
        with self.assertRaises(ValueError):
            fp.verify_numeric_input(invalid_data)
        
        # Invalid input (not a DataFrame)
        with self.assertRaises(ValueError):
            fp.verify_numeric_input([1, 2, 3])
    
    @patch('faircause.fairprediction.train_w_es')
    @patch('faircause.fairprediction.pred_nn_proba')
    @patch('faircause.fairprediction.lambda_performance')
    def test_train(self, mock_lambda_performance, mock_pred_nn_proba, mock_train_w_es):
        """Test training method with mocked dependencies."""
        # Setup mocks
        mock_model = MagicMock()
        mock_pred_nn_proba.return_value = torch.tensor([0.7]*13)
        mock_train_w_es.return_value = mock_model
        
        # Mock lambda_performance to return valid DataFrame
        mock_lambda_performance.return_value = pd.DataFrame({
            'measure': ['nde', 'nie', 'expse_x1', 'expse_x0'],
            'value': [0.1, 0.2, 0.3, 0.4],
            'sd': [0.01, 0.02, 0.03, 0.04],
            'bce': [0.5, 0.5, 0.5, 0.5],
            'bce_sd': [0.05, 0.05, 0.05, 0.05],
            'acc': [0.8, 0.8, 0.8, 0.8],
            'acc_sd': [0.02, 0.02, 0.02, 0.02],
            'auc': [0.75, 0.75, 0.75, 0.75],
            'auc_sd': [0.03, 0.03, 0.03, 0.03],
            'lmbd': [0.1, 0.1, 0.1, 0.1]
        })
        
        # Create FairPredict instance with small lambda sequence
        fp = FairPredict(
            data=self.data,
            X=self.X,
            Z=self.Z,
            W=self.W,
            Y=self.Y,
            x0=self.x0,
            x1=self.x1,
            BN=self.BN,
            lmbd_seq=[0.1],  # Only one lambda for testing
            patience=2
        )
        
        # Mock the FairCause.summary method
        fp.y_fcb.summary = MagicMock(return_value=pd.DataFrame({
            'measure': ['nde', 'nie', 'expse_x1', 'expse_x0'],
            'value': [0.1, 0.2, 0.3, 0.4],
            'sd': [0.01, 0.02, 0.03, 0.04]
        }))
        
        # Call train method
        fp.train()
        
        # Verify function calls
        mock_train_w_es.assert_called_once()
        mock_lambda_performance.assert_called_once()
        
        # Check attributes are set
        self.assertIsNotNone(fp.y_meas)
        self.assertIsNotNone(fp.yhat_meas)
        self.assertIsNotNone(fp.nn_mod)
    
    @patch('faircause.fairprediction.autoplot_fair_prediction')
    def test_plot(self, mock_autoplot):
        """Test plot method."""
        # Create FairPredict instance
        fp = FairPredict(
            data=self.data,
            X=self.X,
            Z=self.Z,
            W=self.W,
            Y=self.Y,
            x0=self.x0,
            x1=self.x1,
            lmbd_seq=self.lmbd_seq
        )
        
        # Set required attributes that would normally be populated by train()
        fp.y_meas = pd.DataFrame({
            'measure': ['nde', 'nie', 'expse_x1', 'expse_x0'],
            'value': [0.1, 0.2, 0.3, 0.4],
            'sd': [0.01, 0.02, 0.03, 0.04]
        })
        
        fp.yhat_meas = pd.DataFrame({
            'measure': ['nde', 'nie', 'expse_x1', 'expse_x0'],
            'value': [0.1, 0.2, 0.3, 0.4],
            'sd': [0.01, 0.02, 0.03, 0.04],
            'bce': [0.5, 0.5, 0.5, 0.5],
            'bce_sd': [0.05, 0.05, 0.05, 0.05],
            'acc': [0.8, 0.8, 0.8, 0.8],
            'acc_sd': [0.02, 0.02, 0.02, 0.02],
            'auc': [0.75, 0.75, 0.75, 0.75],
            'auc_sd': [0.03, 0.03, 0.03, 0.03],
            'lmbd': [0.1, 0.1, 0.1, 0.1]
        })
        
        # Call plot method
        fp.plot(type="causal")
        
        # Verify autoplot_fair_prediction was called with correct arguments
        mock_autoplot.assert_called_once_with(fp.yhat_meas, fp.y_meas, fp.BN, "causal")
    
    @patch('faircause.fairprediction.pred_nn_proba')
    def test_predict(self, mock_pred_nn_proba):
        """Test prediction method with mocked dependencies."""
        
        # Setup a more comprehensive mock
        tensor_mock = MagicMock()
        numpy_mock = MagicMock()
        flatten_mock = MagicMock()
        
        # Configure the chain of mocks
        flatten_mock.return_value = np.array([0.7, 0.3, 0.8, 0.2, 0.6])
        numpy_mock.flatten = MagicMock(return_value=flatten_mock.return_value)
        tensor_mock.numpy = MagicMock(return_value=numpy_mock)
        
        # Have torch.sigmoid return our configured mock
        with patch('torch.sigmoid', return_value=tensor_mock):
            # Mock pred_nn_proba to return anything (it won't matter now)
            mock_pred_nn_proba.return_value = "This doesn't matter"
            
            # Create test data
            test_data = pd.DataFrame({
                'X': [0, 1, 0, 1, 0],
                'Z1': [0.1, 0.2, 0.3, 0.4, 0.5],
                'Z2': [0.5, 0.4, 0.3, 0.2, 0.1],
                'W1': [0.1, 0.2, 0.3, 0.4, 0.5],
                'W2': [0.5, 0.4, 0.3, 0.2, 0.1],
                'Y': [1, 0, 1, 0, 1]
            })
            
            # Create FairPredict instance
            fp = FairPredict(
                data=self.data,
                X=self.X,
                Z=self.Z,
                W=self.W,
                Y=self.Y,
                x0=self.x0,
                x1=self.x1,
                lmbd_seq=[0.1]  # Only one lambda for testing
            )
            
            # Setup necessary attributes that would be set by train()
            fp.y_meas = pd.DataFrame({
                'measure': ['nde', 'nie', 'expse_x1', 'expse_x0'],
                'value': [0.1, 0.2, 0.3, 0.4],
                'sd': [0.01, 0.02, 0.03, 0.04]
            })
            
            mock_model = MagicMock()
            fp.nn_mod = {'lmbd': mock_model}
            
            # Mock methods on FairCause
            with patch('faircause.fairprediction.FairCause') as mock_faircause:
                mock_faircause_instance = MagicMock()
                mock_faircause.return_value = mock_faircause_instance
                
                # Mock summary method to return a DataFrame
                mock_faircause_instance.summary.return_value = pd.DataFrame({
                    'measure': ['nde', 'nie', 'expse_x1', 'expse_x0'],
                    'value': [0.1, 0.2, 0.3, 0.4],
                    'sd': [0.01, 0.02, 0.03, 0.04]
                })
                
                # Call predict method
                result = fp.predict(test_data)
                
                # Verify FairCause was initialized correctly
                mock_faircause.assert_called_once()
                
                # Check result structure
                self.assertIsInstance(result, dict)
                self.assertIn('predictions', result)
                self.assertIn('test_meas', result)
                self.assertIn('y_meas', result)
                self.assertIn('BN', result)
                
                # Check predictions
                self.assertIsInstance(result['predictions'], dict)
                
                # Check if FairCause.estimate_effects was called
                mock_faircause_instance.estimate_effects.assert_called_once()


if __name__ == '__main__':
    unittest.main()