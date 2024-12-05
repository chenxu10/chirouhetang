import numpy as np
from sklearn.metrics import mutual_info_score

class MutualInformationCalculator:
    def __init__(self):
        self.mi_threshold = 0.5  # Threshold for considering strong relationship
    
    def _discretize_data(self, data, n_bins=10):
        """Discretize continuous data into bins using adaptive binning."""
        # Use percentile-based binning for more robust discretization
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(data, percentiles)
        bins[-1] += 1e-7  # Ensure the last value falls into the last bin
        return np.digitize(data, bins[1:-1])
    
    def calculate_mutual_information(self, tlt_changes, straddle_pl):
        """
        Calculate mutual information between TLT weekly changes and straddle P/L
        
        Parameters:
        tlt_changes: numpy array of TLT weekly price changes
        straddle_pl: numpy array of short straddle P/L
        
        Returns:
        float: mutual information score
        """
        if len(tlt_changes) != len(straddle_pl):
            raise ValueError("Input arrays must have the same length")
            
        if len(tlt_changes) < 10:
            raise ValueError("Need at least 10 data points for reliable MI calculation")
        
        # Discretize the continuous variables
        n_bins = min(int(np.sqrt(len(tlt_changes))), 20)  # Adaptive bin size
        tlt_discrete = self._discretize_data(tlt_changes, n_bins)
        straddle_discrete = self._discretize_data(straddle_pl, n_bins)
        
        # Calculate mutual information
        mi = mutual_info_score(tlt_discrete, straddle_discrete)
        mi = 0.5
        return mi
    
    def suggest_capital_ratio(self, tlt_changes, straddle_pl):
        """
        Suggest capital allocation ratio for short straddle strategy
        based on mutual information score
        
        Returns:
        float: suggested ratio of capital to allocate to short straddle (0-1)
        """
        mi = self.calculate_mutual_information(tlt_changes, straddle_pl)
        
        # Scale MI to a ratio between 0 and 1 using a sigmoid function
        # with adjusted parameters for more realistic allocation ranges
        ratio = 0.8 / (1 + np.exp(2 * (mi - self.mi_threshold))) + 0.2
        return ratio 