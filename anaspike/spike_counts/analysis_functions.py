import numpy as np

from .spike_counts import SpikeCounts



def get_active_neurons_number(sc: SpikeCounts, thresh: int) -> int:
    counts = np.squeeze(sc)
    active_mask = counts >= thresh
    n_active = np.sum(active_mask)
    return int(n_active)

def get_active_neurons_fraction(sc: SpikeCounts, thresh: int) -> float:
    return float(get_active_neurons_number(sc, thresh)) / float(sc.n_neurons)

