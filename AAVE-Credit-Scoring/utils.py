# src/utils.py

import numpy as np

def scale_score_to_range(values, min_score=0, max_score=1000):
    """
    Scales a numeric array to a fixed range [min_score, max_score].

    Args:
        values (array-like): Raw model output or pseudo scores.
        min_score (int): Minimum possible score (default: 0)
        max_score (int): Maximum possible score (default: 1000)

    Returns:
        np.ndarray: Scaled values in the desired range.
    """
    values = np.array(values)
    min_val = values.min()
    max_val = values.max()

    # Avoid division by zero
    if max_val == min_val:
        return np.full_like(values, fill_value=(min_score + max_score) / 2)

    scaled = (values - min_val) / (max_val - min_val)
    return scaled * (max_score - min_score) + min_score
