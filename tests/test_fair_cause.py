import unittest
from faircause.faircause import FairCause
import pandas as pd
import numpy as np
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from faircause.estimation.one_step_debiased import *
       
class TestFairCause(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset
        self.data = pd.DataFrame({
            'X': np.random.randint(0, 2, 500),
            'Y': np.random.randn(500),
            'Z1': np.random.randn(500),
            'Z2': np.random.randn(500),
            'W1': np.random.randn(500),
            'W2': np.random.randn(500)
        })
        
        # Create binary outcome data
        self.binary_data = pd.DataFrame({
            'X': np.random.randint(0, 2, 500),
            'Y': np.random.randint(0, 2, 500),
            'Z1': np.random.randn(500),
            'Z2': np.random.randn(500),
            'W1': np.random.randn(500),
            'W2': np.random.randn(500)
        })
        
        # Create categorical data for preprocessing tests
        self.categorical_data = pd.DataFrame({
            'X': ['A', 'B', 'A', 'B', 'A'] * 100,
            'Y': np.random.randint(0, 2, 500),
            'Z1': ['cat1', 'cat2', 'cat1', 'cat2', 'cat1'] * 100,
            'W1': ['high', 'medium', 'low'] * (33*5) + ['high'] *5,
            'W2': np.random.randn(500)
        })

    def test_init(self):
        # Create a sample dataset
        data = pd.DataFrame({
            'X': [1, 2, 3],
            'Y': [4, 5, 6],
            'Z': [7, 8, 9],
            'W': [10, 11, 12]
        })

        # Create a FairCause instance
        fair_cause = FairCause(data, 'X', ['Z'], ['W'], 'Y', 1, 2)

        # Verify that the instance was created correctly
        self.assertIsInstance(fair_cause, FairCause)
        self.assertEqual(fair_cause.X, 'X')
        self.assertEqual(fair_cause.Z, ['Z'])
        self.assertEqual(fair_cause.W, ['W'])
        self.assertEqual(fair_cause.Y, 'Y')
        self.assertEqual(fair_cause.x0, 1)
        self.assertEqual(fair_cause.x1, 2)

    def test_estimate_effects(self):
        # Create a sample dataset
        data = pd.DataFrame({
            'X': [1, 0, 0, 1, 0, 1, 0, 1],
            'Y': [7, 8, 9, 10, 11, 12, 13, 14],
            'Z': [13, 14, 15, 16, 17, 18, 19, 20],
            'W': [19, 20, 21, 22, 23, 24, 25, 26]
        })

        # Create a FairCause instance
        fair_cause = FairCause(data, 'X', ['Z'], ['W'], 'Y', 1, 0, method='medDML')

        # Call the estimate_effects method
        fair_cause.estimate_effects()

        # Verify that the method returned the expected result
        self.assertIsInstance(fair_cause.res, list)
        self.assertEqual(len(fair_cause.res), 1)

    def test_estimate_effects_classification(self): 
        # Create a sample dataset
        data = pd.DataFrame({
            'X': [1, 0, 0, 1, 0, 1, 0, 1],
            'Y': [1,1, 0, 1, 1, 1, 0, 0],
            'Z': [13, 14, 15, 16, 17, 18, 19, 20],
            'W': [19, 20, 21, 22, 23, 24, 25, 26]
        })

        # Create a FairCause instance
        fair_cause = FairCause(data, 'X', ['Z'], ['W'], 'Y', 1, 0, method='medDML')

        # Call the estimate_effects method
        fair_cause.estimate_effects()

        # Verify that the method returned the expected result
        self.assertIsInstance(fair_cause.res, list)
        self.assertEqual(len(fair_cause.res), 1)

    def test_estimate_effects_linear(self): 
        data = pd.DataFrame({
            'X': [1, 0, 0, 1, 0, 1, 0, 1],
            'Y': [7, 8, 9, 10, 11, 12, 13, 14],
            'Z': [13, 14, 15, 16, 17, 18, 19, 20],
            'W': [19, 20, 21, 22, 23, 24, 25, 26]
        })

        # Create a FairCause instance
        fair_cause = FairCause(data, 'X', ['Z'], ['W'], 'Y', 1, 0, model='linear', method='medDML')

        # Call the estimate_effects method
        fair_cause.estimate_effects()

        # Verify that the method returned the expected result
        self.assertIsInstance(fair_cause.res, list)
        self.assertEqual(len(fair_cause.res), 1)

    def test_estimate_effects_tune(self): 
        # Create a sample dataset
        data = pd.DataFrame({
            'X': np.random.randint(0, 2, 100),
            'Y': np.random.randint(0, 100, 100),
            'Z': np.random.randint(0, 100, 100),
            'W': np.random.randint(0, 100, 100)
        })

        # Create a FairCause instance
        fair_cause = FairCause(data, 'X', ['Z'], ['W'], 'Y', 1, 0, tune_params=True, method='medDML')

        # Call the estimate_effects method
        fair_cause.estimate_effects()

        # Verify that the method returned the expected result
        self.assertIsInstance(fair_cause.res, list)
        self.assertEqual(len(fair_cause.res), 1)
    
    def test_estimate_effects_classification_tune(self): 
        # Create a sample dataset
        data = pd.DataFrame({
            'X': np.random.randint(0, 2, 100),
            'Y': np.random.randint(0, 2, 100),
            'Z': np.random.randint(0, 100, 100),
            'W': np.random.randint(0, 100, 100)
        })

        # Create a FairCause instance
        fair_cause = FairCause(data, 'X', ['Z'], ['W'], 'Y', 1, 0, tune_params=True, method='medDML')

        # Call the estimate_effects method
        fair_cause.estimate_effects()

        # Verify that the method returned the expected result
        self.assertIsInstance(fair_cause.res, list)
        self.assertEqual(len(fair_cause.res), 1)

    def test_validate_inputs(self):
        # Create a sample dataset
        data = pd.DataFrame({
            'X': [1, 2, 3],
            'Y': [4, 5, 6],
            'Z': [7, 8, 9],
            'W': [10, 11, 12]
        })

        # Create a FairCause instance
        fair_cause = FairCause(data, 'X', ['Z'], ['W'], 'Y', 1, 2)

        # Call the _validate_inputs method
        fair_cause._validate_inputs(data)

        # Verify that no errors were raised
        self.assertTrue(True)

    def test_preprocess_data(self):
        # Create a sample dataset
        data = pd.DataFrame({
            'X': [1, 2, 3],
            'Y': [4, 5, 6],
            'Z': [7, 8, 9],
            'W': [10, 11, 12]
        })

        # Create a FairCause instance
        fair_cause = FairCause(data, 'X', ['Z'], ['W'], 'Y', 1, 2)

        # Call the _preprocess_data method
        preprocessed_data = fair_cause._preprocess_data(data)

        # Verify that the method returned the expected result
        self.assertIsInstance(preprocessed_data, pd.DataFrame)
        self.assertEqual(preprocessed_data.shape, data.shape)

    def test_preproc_data_basic(self):
        """Test preprocessing of basic numeric data"""
        data_proc, sfm = preproc_data(self.data, 'X', ['Z1', 'Z2'], ['W1', 'W2'], 'Y')
        
        # Check that data is returned as DataFrame
        self.assertIsInstance(data_proc, pd.DataFrame)
        
        # Check that SFM structure is correct
        self.assertEqual(sfm['X'], 'X')
        self.assertEqual(sfm['Z'], ['Z1', 'Z2'])
        self.assertEqual(sfm['W'], ['W1', 'W2'])
        self.assertEqual(sfm['Y'], 'Y')
        
    def test_preproc_data_categorical(self):
        """Test preprocessing of categorical data"""
        data_proc, sfm = preproc_data(self.categorical_data, 'X', ['Z1'], ['W1', 'W2'], 'Y')
        
        # Check that categorical variables are properly encoded
        self.assertTrue(pd.api.types.is_numeric_dtype(data_proc['X']))
        
        # Check multi-level categorical handling (W1 should be one-hot encoded)
        self.assertNotIn('W1', data_proc.columns)
        self.assertTrue(any('W1' in col for col in data_proc.columns))
        
    def test_cv_xgb_regression(self):
        """Test cross-validated XGBoost for regression"""
        X = self.data[['Z1', 'Z2', 'W1', 'W2']]
        y = self.data['Y']
        
        result = cv_xgb(X, y)
        
        # Check return structure
        self.assertIsInstance(result, dict)
        self.assertIn('model', result)
        self.assertIn('is_binary', result)
        self.assertFalse(result['is_binary'])
        
    def test_cv_xgb_classification(self):
        """Test cross-validated XGBoost for classification"""
        X = self.binary_data[['Z1', 'Z2', 'W1', 'W2']]
        y = self.binary_data['Y']
        
        result = cv_xgb(X, y)
        
        # Check return structure
        self.assertIsInstance(result, dict)
        self.assertIn('model', result)
        self.assertIn('is_binary', result)
        self.assertTrue(result['is_binary'])
        
    def test_pred_xgb(self):
        """Test prediction from trained XGBoost model"""
        # Train a model first
        X_train = self.data[['Z1', 'Z2', 'W1', 'W2']]
        y_train = self.data['Y']
        model_result = cv_xgb(X_train, y_train)
        
        # Test prediction
        X_test = pd.DataFrame({
            'Z1': np.random.randn(10),
            'Z2': np.random.randn(10),
            'W1': np.random.randn(10),
            'W2': np.random.randn(10)
        })
        
        predictions = pred_xgb(model_result, X_test)
        
        # Check predictions
        self.assertEqual(len(predictions), len(X_test))
        self.assertTrue(np.all(np.isfinite(predictions)))
        
    def test_pred_xgb_intervention(self):
        """Test prediction with intervention"""
        # Train a model first
        data_with_x = self.data.copy()
        X_train = data_with_x[['X', 'Z1', 'Z2', 'W1', 'W2']]
        y_train = data_with_x['Y']
        model_result = cv_xgb(X_train, y_train)
        
        # Test prediction with intervention
        X_test = pd.DataFrame({
            'X': np.random.randint(0, 2, 10),
            'Z1': np.random.randn(10),
            'Z2': np.random.randn(10),
            'W1': np.random.randn(10),
            'W2': np.random.randn(10)
        })
        
        predictions = pred_xgb(model_result, X_test, intervention=1, X="X")
        
        # Check predictions
        self.assertEqual(len(predictions), len(X_test))
        self.assertTrue(np.all(np.isfinite(predictions)))
        
    def test_measure_spec(self):
        """Test measure specification function"""
        measures = measure_spec()
        
        # Check that default measures are returned
        self.assertIsInstance(measures, dict)
        self.assertIn('tv', measures)
        self.assertIn('ctfde', measures)
        self.assertIn('ctfie', measures)
        self.assertIn('ctfse', measures)
        self.assertIn('ett', measures)
        
        # Check structure of each measure
        for measure_name, measure_info in measures.items():
            self.assertIn('sgn', measure_info)
            self.assertIn('spc', measure_info)
            self.assertIn('ia', measure_info)
            
    def test_one_step_debias_basic(self):
        """Test one-step debiasing procedure"""
        # First preprocess data
        data_proc, sfm = preproc_data(self.data, 'X', ['Z1', 'Z2'], ['W1', 'W2'], 'Y')
        
        # Run one-step debiasing
        result = one_step_debias(
            data_proc, 
            sfm['X'], 
            sfm['Z'], 
            sfm['W'], 
            sfm['Y'],
            nested_mean='refit',
            log_risk=False
        )
        
        # Check return structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('measure', result.columns)
        self.assertIn('value', result.columns)
        self.assertIn('sd', result.columns)
        self.assertIn('scale', result.columns)
        
        # Check measures are correct
        expected_measures = ['tv', 'ett', 'ctfde', 'ctfie', 'ctfse']
        self.assertTrue(all(m in result['measure'].values for m in expected_measures))
        
    def test_one_step_debias_with_log_risk(self):
        """Test one-step debiasing with log-risk scale"""
        # First preprocess data
        data_proc, sfm = preproc_data(self.binary_data, 'X', ['Z1', 'Z2'], ['W1', 'W2'], 'Y')
        
        # Run one-step debiasing
        result = one_step_debias(
            data_proc, 
            sfm['X'], 
            sfm['Z'], 
            sfm['W'], 
            sfm['Y'],
            nested_mean='refit',
            log_risk=True
        )
        
        # Check return structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result['scale'].iloc[0], 'log-risk')
        
    def test_one_step_debias_invalid_nested_mean(self):
        """Test error handling for invalid nested_mean parameter"""
        # First preprocess data
        data_proc, sfm = preproc_data(self.data, 'X', ['Z1', 'Z2'], ['W1', 'W2'], 'Y')
        
        # Test invalid nested_mean parameter
        with self.assertRaises(ValueError):
            one_step_debias(
                data_proc, 
                sfm['X'], 
                sfm['Z'], 
                sfm['W'], 
                sfm['Y'],
                nested_mean='invalid'
            )

    def test_str_method(self):
        """Test the string representation of FairCause object"""
        fair_cause = FairCause(self.data, 'X', ['Z1', 'Z2'], ['W1', 'W2'], 'Y', 0, 1)
        
        result = str(fair_cause)
        self.assertIn('faircause object:', result)
        self.assertIn('Attribute:       X', result)
        self.assertIn('Outcome:         Y', result)
        self.assertIn('Confounders:     W1, W2', result)
        self.assertIn('Mediators:       Z1, Z2', result)
    
    def test_summary_without_effects(self):
        """Test summary method before running estimate_effects"""
        fair_cause = FairCause(self.data, 'X', ['Z1'], ['W1'], 'Y', 0, 1)
        
        with self.assertRaises(ValueError) as context:
            fair_cause.summary()
        
        self.assertIn("No results available", str(context.exception))
    
    def test_summary_with_effects(self):
        """Test summary method after running estimate_effects"""
        fair_cause = FairCause(self.data, 'X', ['Z1'], ['W1'], 'Y', 0, 1, n_boot1=1, n_boot2=10)
        fair_cause.estimate_effects()
        
        summary = fair_cause.summary(decompose="xspec", print_sum=False)
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn('measure', summary.columns)
        self.assertIn('value', summary.columns)
        self.assertIn('sd', summary.columns)
    
    def test_summary_decomposition_options(self):
        """Test different decomposition options in summary"""
        fair_cause = FairCause(self.data, 'X', ['Z1'], ['W1'], 'Y', 0, 1, n_boot1=1, n_boot2=10, method='medDML')
        fair_cause.estimate_effects()
        
        # Test different decomposition options
        for decompose in ["xspec", "general", "both"]:
            summary = fair_cause.summary(decompose=decompose, print_sum=False)
            self.assertIsInstance(summary, pd.DataFrame)
    
    def test_plot_without_effects(self):
        """Test plot method before running estimate_effects"""
        fair_cause = FairCause(self.data, 'X', ['Z1'], ['W1'], 'Y', 0, 1)
        
        with self.assertRaises(ValueError) as context:
            fair_cause.plot()
        
        self.assertIn("No results available", str(context.exception))
    
    def test_plot_with_effects(self):
        """Test plot method after running estimate_effects"""
        fair_cause = FairCause(self.data, 'X', ['Z1'], ['W1'], 'Y', 0, 1, n_boot1=1, n_boot2=10)
        fair_cause.estimate_effects()
        
        fig = fair_cause.plot(decompose="xspec", dataset="Test Dataset", signed=True, var_name="outcome")
        
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)  # Clean up
    
    def test_plot_different_options(self):
        """Test plot method with different options"""
        fair_cause = FairCause(self.data, 'X', ['Z1'], ['W1'], 'Y', 0, 1, n_boot1=1, n_boot2=10, method='medDML')
        fair_cause.estimate_effects()
        
        # Test with signed=False
        fig = fair_cause.plot(decompose="xspec", signed=False)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
        
        # Test with decompose="general"
        fig = fair_cause.plot(decompose="general")
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_debiasing_method(self):
        """Test estimate_effects with debiasing method"""
        fair_cause = FairCause(self.data, 'X', ['Z1'], ['W1'], 'Y', 0, 1, method='debiasing')
        fair_cause.estimate_effects()
        
        self.assertEqual(len(fair_cause.res), 1)
        result_df = fair_cause.res[0]
        self.assertIn('measure', result_df.columns)
        self.assertTrue(any(result_df['measure'] == 'tv'))
    
    def test_input_validation_errors(self):
        """Test various input validation errors"""
        # Test invalid column
        with self.assertRaises(ValueError):
            FairCause(self.data, 'NonExistentColumn', ['Z1'], ['W1'], 'Y', 0, 1)
        
        # Test invalid protected attribute levels
        with self.assertRaises(ValueError):
            FairCause(self.data, 'X', ['Z1'], ['W1'], 'Y', 2, 3)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            FairCause(self.data, 'X', ['Z1'], ['W1'], 'Y', 0, 1, method='invalid_method')
        
        # Test invalid model
        with self.assertRaises(ValueError):
            FairCause(self.data, 'X', ['Z1'], ['W1'], 'Y', 0, 1, model='invalid_model')
        
        # Test linear model with tune_params=True
        with self.assertRaises(ValueError):
            FairCause(self.data, 'X', ['Z1'], ['W1'], 'Y', 0, 1, model='linear', tune_params=True)
    
    def test_preprocessing_categorical_variables(self):
        """Test preprocessing of categorical variables"""
        fair_cause = FairCause(self.categorical_data, 'X', ['Z1'], ['W1'], 'Y', 'A', 'B')
        
        # Check that protected attribute is encoded correctly
        self.assertEqual(fair_cause.data['X'].unique().tolist(), [0, 1])
        
        # Check that categorical outcomes are encoded
        self.assertTrue(pd.api.types.is_numeric_dtype(fair_cause.data['Y']))
    
    def test_empty_mediators_and_confounders(self):
        """Test with empty mediators and confounders"""
        fair_cause = FairCause(self.data, 'X', [], [], 'Y', 0, 1)
        fair_cause.estimate_effects()
        
        self.assertEqual(len(fair_cause.res), 1)
    



if __name__ == '__main__':
    unittest.main()