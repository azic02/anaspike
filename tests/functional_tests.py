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
from anaspike.dataclasses.bins import ContigBins1D, ContigBins2D
from anaspike.dataclasses.grid import RegularGrid1D, RegularGrid2D

pop = populations['inh']

extent = 2.
origin = (0., 0.)
x_extent = Interval( origin[0] - extent / 2., origin[0] + extent / 2.)
y_extent = Interval( origin[1] - extent / 2., origin[1] + extent / 2.)
t_sim = 10000.

t_interval = Interval(2000., t_sim)
t_bin_size = 500.
t_bins = ContigBins1D.with_median_labels(RegularGrid1D.from_interval_given_d(t_interval, t_bin_size))

n_spatial_bins = 50
spatial_bins = ContigBins2D.with_median_labels(RegularGrid2D(x=RegularGrid1D.from_interval_given_n(x_extent, n_spatial_bins, endpoint=True),
                                                             y=RegularGrid1D.from_interval_given_n(x_extent, n_spatial_bins, endpoint=True)))

spatial_bin_size = extent / n_spatial_bins
time_vals = t_bins.labels
# -

# ## analysis

# ### spike trains

from anaspike.spike_trains import SpikeTrains
spike_trains = SpikeTrains.from_nest(pop, spike_recorder)

from anaspike.spike_trains import construct_spike_time_histogram
spike_time_bins = ContigBins1D.with_median_labels(RegularGrid1D.from_interval_given_n(t_interval, n=300, endpoint=True))
spike_time_histogram = construct_spike_time_histogram(spike_trains, spike_time_bins)
fig, ax = plt.subplots(figsize=(15,3))
spike_time_histogram.plot(ax)
plt.show()

from anaspike.spike_trains import construct_interspike_interval_histogram
spike_interval_bins = ContigBins1D.with_median_labels(RegularGrid1D.from_interval_given_n(Interval(0,400), n=100, endpoint=True))
interspike_interval_histogram = construct_interspike_interval_histogram(spike_trains, spike_interval_bins)
fig, ax = plt.subplots(figsize=(15,3))
interspike_interval_histogram.plot(ax)
plt.show()

# ### spike counts

from anaspike.spike_counts import SpikeCounts
spike_counts = SpikeCounts.from_spike_trains(spike_trains, Interval(-np.inf, np.inf))

from anaspike.spike_counts import get_active_neurons_number
active_neurons_number = get_active_neurons_number(spike_counts, thresh=1)
print(active_neurons_number)

from anaspike.spike_counts import get_active_neurons_fraction
active_neurons_fraction = get_active_neurons_fraction(spike_counts, thresh=1)
print(active_neurons_fraction)

# ### time averaged firing rate

# +
from anaspike.firing_rates import FiringRates
firing_ratess = FiringRates.from_spike_trains(spike_trains, t_interval)

fig, ax = plt.subplots()
scat = ax.scatter(x=pop.coords.x, y=pop.coords.y, c=firing_ratess.values, vmin=0., cmap='Greys', s=10)
ax.set_xlabel('cortical space [mm]')
ax.set_ylabel('cortical space [mm]')
fig.colorbar(scat, ax=ax, label='firing rate [Hz]')
plt.show()
# -

from anaspike.firing_rates import calculate_morans_i
morans_i = calculate_morans_i(firing_ratess)
print(morans_i)

from anaspike.firing_rates import mean as tafr_mean
from anaspike.firing_rates import std as tafr_std
print(tafr_mean(firing_ratess), tafr_std(firing_ratess))

from anaspike.firing_rates import construct_histogram
freq_bins = ContigBins1D.with_median_labels(RegularGrid1D.from_interval_given_n(Interval(0., 5.), n=20, endpoint=True))
tafr_histogram = construct_histogram(firing_ratess, freq_bins)
fig, ax = plt.subplots(figsize=(15,3))
tafr_histogram.plot(ax)
plt.show()

# +
from anaspike.firing_rates import bin_spatially
binned_tafr = bin_spatially(firing_ratess, spatial_bins)

fig, ax = plt.subplots()
pcmesh = ax.pcolormesh(binned_tafr.xx, binned_tafr.yy, binned_tafr.elements,
             vmin=0.,
             cmap='Greys')
ax.set_xlabel('cortical space [mm]')
ax.set_ylabel('cortical space [mm]')
fig.colorbar(pcmesh, ax=ax, label='firing rate [Hz]')
plt.show()

# +
from anaspike.firing_rates import calculate_spatial_psd
tafr_spatial_psd = calculate_spatial_psd(binned_tafr)

fig, ax = plt.subplots()
pcmesh = ax.pcolormesh(tafr_spatial_psd.xx, tafr_spatial_psd.yy, tafr_spatial_psd.elements,
             vmin=0.,
             cmap='Greys')
ax.set_xlabel('freq [1/mm]')
ax.set_ylabel('freq [1/mm]')
fig.colorbar(pcmesh, ax=ax, label='PSD')
plt.show()

# +
from anaspike.firing_rates import calculate_spatial_autocorrelation_wiener_khinchin
spatial_ac_wk = calculate_spatial_autocorrelation_wiener_khinchin(binned_tafr)

fig, ax = plt.subplots()
pcmesh = ax.pcolormesh(spatial_ac_wk.xx, spatial_ac_wk.yy, spatial_ac_wk.elements,
                       vmin=-1.,
                       vmax=1.0,
                       cmap='bwr')
ax.set_xlabel('offset [mm]')
ax.set_ylabel('offset [mm]')
fig.colorbar(pcmesh, ax=ax, label='spatial autocorrelation (wiener-khinchin)')
plt.show()
# -

# ### instantaneous firing rate

from anaspike.firing_rates_evolution import FiringRatesEvolution
firing_rates_evolution = FiringRatesEvolution.from_spike_trains(spike_trains, t_bins)

# +
from anaspike.firing_rates_evolution import temporal_correlation
from anaspike.visualization import plot_firing_rate_temporal_correlation

e = 0.02
center_mask = (-e < pop.x_pos) & (pop.x_pos < e) & (-e < pop.y_pos) & (pop.y_pos < e)
ref_neuron = pop[center_mask][0]
ref_neuron_firing_rate = firing_rates_evolution.along_neuron_dim[center_mask][0]

t_corr = temporal_correlation(firing_rates_evolution, ref_neuron_firing_rate)

fig, ax = plt.subplots(figsize=(12,10))
plot_firing_rate_temporal_correlation(fig, ax,
                                      (ref_neuron.x_pos, ref_neuron.y_pos),
                                      pop.x_pos, pop.y_pos,
                                      t_corr,
                                      cmap='bwr')
# +
from anaspike.firing_rates_evolution import temporal_correlation_matrix
from anaspike.visualization import plot_pairwise_temporal_correlation_matrix

t_corr_mat = temporal_correlation_matrix(firing_rates_evolution)

fig, ax = plt.subplots()
plot_pairwise_temporal_correlation_matrix(fig, ax, t_corr_mat)
# +
from anaspike.firing_rates_evolution import morans_i_evolution
from anaspike.visualization import plot_morans_i_evolution

morans_i = morans_i_evolution(firing_rates_evolution, pop.coords, decay_power=4)

fig, ax = plt.subplots()
plot_morans_i_evolution(ax, time_vals, morans_i)
# -


