import unittest
from cloud import CLOUD
import numpy as np

np.random.seed(0)

def generate_continuous_X_to_Y_data(sample_size: int):
    a = np.random.choice([np.random.uniform(-2, -0.5), np.random.uniform(0.5, 2)])
    b = np.random.choice([np.random.uniform(-2, -0.5), np.random.uniform(0.5, 2)])
    
    x0 = np.random.normal(-5, 2, int(sample_size * 0.6))
    x1 = np.random.normal(0, 1, int(sample_size * 0.2))
    x2 = np.random.normal(5, 2, sample_size - len(x0) - len(x1))
    x = np.concatenate((x0, x1, x2))
    np.random.shuffle(x)

    y = a * x + b * np.sin(x) + np.random.normal(0, 2, sample_size)
    return x, y

class TestCLOUDCausalDirection(unittest.TestCase):
    def test_correct_causal_direction_continuous_X_to_Y(self):
        """Check if CLOUD correctly identifies the causal direction for continuous X to Y."""
        sample_size = 1000
        x, y = generate_continuous_X_to_Y_data(sample_size)
        model = CLOUD(x, y, n_candidates=4, is_X_continuous=True, is_Y_continuous=True, max_exponent=6)
        prediction_results = model.predict(report=True)
        
        self.assertEqual(prediction_results, 'to', "CLOUD failed to identify 'X → Y' as the model with the shortest code length.")

if __name__ == '__main__':
    unittest.main()

