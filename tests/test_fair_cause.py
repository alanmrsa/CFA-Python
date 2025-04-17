import unittest
from faircause.faircause import FairCause
import pandas as pd

class TestFairCause(unittest.TestCase):

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
        fair_cause = FairCause(data, 'X', ['Z'], ['W'], 'Y', 1, 0)

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

if __name__ == '__main__':
    unittest.main()