import unittest
from faircause.faircause import FairCause

class TestFairCause(unittest.TestCase):
    def test_init(self):
        # Create a FairCause instance with some sample data
        data = pd.DataFrame({'X': [1, 2, 3], 'Y': [4, 5, 6]})
        fair_cause = FairCause(data, 'X', 'Y')

        # Verify that the instance was created correctly
        self.assertIsInstance(fair_cause, FairCause)
        self.assertEqual(fair_cause.X, 'X')
        self.assertEqual(fair_cause.Y, 'Y')

    def test_estimate_effects(self):
        # Create a FairCause instance with some sample data
        data = pd.DataFrame({'X': [1, 2, 3], 'Y': [4, 5, 6]})
        fair_cause = FairCause(data, 'X', 'Y')

        # Call the estimate_effects method
        fair_cause.estimate_effects()

        # Verify that the method returned the expected result
        self.assertIsInstance(fair_cause.res, list)
        self.assertEqual(len(fair_cause.res), 1)

if __name__ == '__main__':
    unittest.main()