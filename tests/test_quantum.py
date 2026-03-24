"""
Tests for quantum path selection module.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantum.path_selector import choose_path


class TestQuantumPathSelector(unittest.TestCase):
    """Test cases for quantum path selection."""
    
    def test_choose_path_basic(self):
        """Test basic path selection."""
        scores = np.array([0.8, 0.3, 0.5, 0.9])
        result = choose_path(scores, rep=1)
        
        # Result should be an integer index
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, len(scores))
    
    def test_choose_path_single_option(self):
        """Test with only one path option."""
        scores = np.array([1.0])
        result = choose_path(scores, rep=1)
        self.assertEqual(result, 0)
    
    def test_choose_path_two_options(self):
        """Test with two path options."""
        scores = np.array([0.5, 0.8])
        result = choose_path(scores, rep=1)
        self.assertIn(result, [0, 1])


if __name__ == '__main__':
    unittest.main()
