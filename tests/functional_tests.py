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
from anaspike.dataclasses.contig_bins_2d import ContigBins2D

pop = populations['inh']

extent = 2.
origin = (0., 0.)
t_sim = 10000.

t_interval = Interval(2000., t_sim)
t_bin_size = 500.
t_bins = EquiBins.from_interval_with_median_values(t_interval, size=t_bin_size)

n_spatial_bins = 50
spatial_bins = ContigBins2D(x=EquiBins.from_interval_with_median_values(Interval(-extent / 2., extent / 2.), n_spatial_bins),
                            y=EquiBins.from_interval_with_median_values(Interval(-extent / 2., extent / 2.), n_spatial_bins))

spatial_bin_size = extent / n_spatial_bins
time_vals = [t_bin.value for t_bin in t_bins]
# -

# ## analysis

# ### spike trains

from anaspike.dataclasses.spike_train import SpikeTrainArray
spike_trains = SpikeTrainArray.from_nest(pop, spike_recorder)

from anaspike.analysis.spike_trains import construct_spike_time_histogram
spike_time_histogram = construct_spike_time_histogram(spike_trains, EquiBins.from_interval_with_median_values(t_interval, n=300))
fig, ax = plt.subplots(figsize=(15,3))
spike_time_histogram.plot(ax)
plt.show()

from anaspike.analysis.spike_trains import construct_interspike_interval_histogram
interspike_interval_histogram = construct_interspike_interval_histogram(spike_trains, EquiBins.from_interval_with_median_values(Interval(0,400), n=100))
fig, ax = plt.subplots(figsize=(15,3))
interspike_interval_histogram.plot(ax)
plt.show()

from anaspike.analysis.spike_trains import calculate_active_neuron_fraction
active_neuron_fraction = calculate_active_neuron_fraction(spike_trains, t_interval, thresh=1)
print(active_neuron_fraction)

# ### time averaged firing rate

# +
from anaspike.analysis.time_averaged_firing_rate import TimeAveragedFiringRate
time_averaged_firing_rates = TimeAveragedFiringRate.from_spike_trains(pop.coords, spike_trains, t_interval)

fig, ax = plt.subplots()
scat = ax.scatter(x=pop.coords.x, y=pop.coords.y, c=time_averaged_firing_rates, vmin=0., cmap='Greys', s=10)
ax.set_xlabel('cortical space [mm]')
ax.set_ylabel('cortical space [mm]')
fig.colorbar(scat, ax=ax, label='firing rate [Hz]')
plt.show()
# -

from anaspike.analysis.time_averaged_firing_rate import mean as tafr_mean
from anaspike.analysis.time_averaged_firing_rate import std as tafr_std
print(tafr_mean(time_averaged_firing_rates), tafr_std(time_averaged_firing_rates))

from anaspike.analysis.time_averaged_firing_rate import construct_histogram
freq_bins = EquiBins.from_interval_with_median_values(Interval(0., 5.), n=20)
tafr_histogram = construct_histogram(time_averaged_firing_rates, freq_bins)
fig, ax = plt.subplots(figsize=(15,3))
tafr_histogram.plot(ax)
plt.show()

# +
from anaspike.analysis.time_averaged_firing_rate import bin_spatially
binned_tafr = bin_spatially(time_averaged_firing_rates, spatial_bins)

fig, ax = plt.subplots()
pcmesh = ax.pcolormesh(binned_tafr.xx, binned_tafr.yy, binned_tafr.elements,
             vmin=0.,
             cmap='Greys')
ax.set_xlabel('cortical space [mm]')
ax.set_ylabel('cortical space [mm]')
fig.colorbar(pcmesh, ax=ax, label='firing rate [Hz]')
plt.show()
# -

# ### instantaneous firing rate

from anaspike.analysis.instantaneous_firing_rate import InstantaneousFiringRates
instantaneous_firing_rates = InstantaneousFiringRates.from_spike_trains(spike_trains, t_bins)

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


