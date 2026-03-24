"""
Tests for classical algorithm modules.
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import parse_time


class TestParseTime(unittest.TestCase):
    """Test cases for time parsing function."""
    
    def test_parse_time_with_minutes(self):
        """Test parsing time with hours and minutes format."""
        self.assertAlmostEqual(parse_time("4'50"), 4.8333, places=2)
        self.assertAlmostEqual(parse_time("1'25"), 1.4167, places=2)
    
    def test_parse_time_float(self):
        """Test parsing float time."""
        self.assertEqual(parse_time("2.5"), 2.5)
        self.assertEqual(parse_time(3.0), 3.0)
    
    def test_parse_time_invalid(self):
        """Test parsing invalid time strings."""
        self.assertIsNone(parse_time("X"))
        self.assertIsNone(parse_time(""))
        self.assertIsNone(parse_time(None))


if __name__ == '__main__':
    unittest.main()
