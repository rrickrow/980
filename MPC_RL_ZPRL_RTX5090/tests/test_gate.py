import unittest
from mpcrl.gate import adaptive_uncertainty_gate
class T(unittest.TestCase):
    def test_monotonic(self): self.assertLess(adaptive_uncertainty_gate(0.0),adaptive_uncertainty_gate(0.1))
if __name__=='__main__': unittest.main()
