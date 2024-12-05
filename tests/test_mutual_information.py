import numpy as np
import pytest
from src.strategy.calmutualinfo import MutualInformationCalculator

def test_mutual_information_calculator_initialization():
    calc = MutualInformationCalculator()
    assert hasattr(calc, 'calculate_mutual_information')


def test_mutual_information_realistic_scenarios():
    calc = MutualInformationCalculator()
    
    # Test case 1: Realistic TLT weekly changes with moderate correlation
    tlt_changes = np.array([-0.014, 0.007, -0.003, 0.021, -0.008, 0.012, -0.005,-0.014, 0.007, -0.003, 0.021, -0.008, 0.012, -0.005])
    straddle_pl = np.array([0.05, -0.02, 0.01, 0.08, 0.03, 0.04, -0.01,0.05, -0.02, 0.01, 0.08, 0.03, 0.04, -0.01])  # Some correlation but not perfect
    mi = calc.calculate_mutual_information(tlt_changes, straddle_pl)
    assert mi > 0 and mi < 1  # Should show moderate mutual information
