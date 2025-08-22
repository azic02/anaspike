# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Analysis Test

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML

# +
from pathlib import Path
from anaspike.dataclasses import SimulationData

test_data_path = Path('./test_data.h5')
simulation_data = SimulationData.load(test_data_path)
populations = simulation_data.populations
spike_recorder = simulation_data.spike_recorder
# -

# ## analysis parameters

# +
from anaspike.dataclasses.interval import Interval
from anaspike.dataclasses.histogram import EquiBins

pop = populations['inh']

extent = 2.
origin = (0., 0.)
t_sim = 10000.

n_spatial_bins = 50

t_bins = EquiBins.from_interval_with_median_values(Interval(100., t_sim), size=500.)
x_bins = EquiBins.from_interval_with_median_values(Interval(-extent / 2., extent / 2.), n_spatial_bins)
y_bins = EquiBins.from_interval_with_median_values(Interval(-extent / 2., extent / 2.), n_spatial_bins)

spatial_bin_size = extent / n_spatial_bins
time_vals = [t_bin.value for t_bin in t_bins]
# -

# ## analysis

# ### instantaneous firing rate

from anaspike.analysis.instantaneous_firing_rate import InstantaneousFiringRates
instantaneous_firing_rates = InstantaneousFiringRates.from_nest(pop, spike_recorder, t_bins)

# +
from anaspike.analysis.instantaneous_firing_rate import temporal_correlation
from anaspike.visualization import plot_firing_rate_temporal_correlation

e = 0.02
center_mask = (-e < pop.x_pos) & (pop.x_pos < e) & (-e < pop.y_pos) & (pop.y_pos < e)
ref_neuron = pop[center_mask][0]
ref_neuron_firing_rate = instantaneous_firing_rates.along_neuron_dim[center_mask][0]

t_corr = temporal_correlation(instantaneous_firing_rates, ref_neuron_firing_rate)

fig, ax = plt.subplots(figsize=(12,10))
plot_firing_rate_temporal_correlation(fig, ax,
                                      (ref_neuron.x_pos, ref_neuron.y_pos),
                                      pop.x_pos, pop.y_pos,
                                      t_corr,
                                      cmap='bwr')
# +
from anaspike.analysis.instantaneous_firing_rate import temporal_correlation_matrix
from anaspike.visualization import plot_pairwise_temporal_correlation_matrix

t_corr_mat = temporal_correlation_matrix(instantaneous_firing_rates)

fig, ax = plt.subplots()
plot_pairwise_temporal_correlation_matrix(fig, ax, t_corr_mat)
# +
from anaspike.analysis.instantaneous_firing_rate import morans_i_evolution
from anaspike.visualization import plot_morans_i_evolution

morans_i = morans_i_evolution(instantaneous_firing_rates, pop.coords, decay_power=4)

fig, ax = plt.subplots()
plot_morans_i_evolution(ax, time_vals, morans_i)
# -


