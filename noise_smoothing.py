# noise_smoothing.py
import numpy as np

def apply_label_smoothing(labels, factor=0.1):
    """Mitigates clinical diagnosis label noise by regularizing hard margins.

    Reduces penalty overhead caused by late electronic medical charting entry.
    """
    labels = np.array(labels)
    return labels * (1.0 - factor) + 0.5 * factor