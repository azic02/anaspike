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

# ## pre-analysis analysis

# +
from anaspike.visualization import spike_raster_plot

fig, ax = plt.subplots(figsize=(14, 4))
spike_raster_plot(ax, spike_recorder.times, spike_recorder.senders, markersize=1.)

# +
from anaspike.dataclasses.interval import Interval
from anaspike.dataclasses.histogram import EquiBins
from anaspike.analysis import intra_population_average_firing_rate_evolution
from anaspike.visualization import plot_intra_population_average_firing_rate_evolution

t_bins = EquiBins.from_interval_with_median_values(Interval(0., 2000.), size=10.)
names, avgs, stds = intra_population_average_firing_rate_evolution(populations, spike_recorder, t_bins)

fig, ax = plt.subplots(figsize=(14, 4))
plot_intra_population_average_firing_rate_evolution(ax, [t_bin.value for t_bin in t_bins], names, avgs, stds)
# -

# ## analysis parameters

# +
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

# +
from anaspike.analysis import firing_rates_evolution
from anaspike.visualization import animate_firing_rate_evolution

firing_rates_at_times = firing_rates_evolution(pop, spike_recorder, t_bins)

fig, ax = plt.subplots(figsize=(12,10))
ani = animate_firing_rate_evolution(fig, ax,
                                    pop.x_pos,
                                    pop.y_pos,
                                    time_vals,
                                    firing_rates_at_times,
                                    cmap='Greys')
ax.set_xlabel('cortical space [mm]')
ax.set_ylabel('cortical space [mm]')

HTML(ani.to_jshtml())

# +
from anaspike.analysis import time_avg_firing_rates_histogram
from anaspike.visualization import plot_time_avg_firing_rate_histogram

t_avg_interval = Interval(100., t_sim)
bin_vals, bin_counts, bin_edges = time_avg_firing_rates_histogram(pop,
                                                                  spike_recorder,
                                                                  t_avg_interval,
                                                                  nbins=10)

fig, ax = plt.subplots()
plot_time_avg_firing_rate_histogram(ax, bin_vals, bin_counts, bin_widths=np.diff(bin_edges))

# +
from anaspike.analysis import spike_gaps_coefficient_of_variation_histogram
from anaspike.visualization import plot_spike_gaps_coefficient_of_variation_histogram

bin_vals, bin_counts, bin_edges = spike_gaps_coefficient_of_variation_histogram(pop,
                                                                                spike_recorder,
                                                                                nbins=10,
                                                                                min_nspikes=2)

fig, ax = plt.subplots()
plot_spike_gaps_coefficient_of_variation_histogram(ax, bin_vals, bin_counts, bin_widths=np.diff(bin_edges))

# +
from anaspike.analysis import firing_rate_temporal_correlation
from anaspike.visualization import plot_firing_rate_temporal_correlation

e = 0.02
mask = (-e < pop.x_pos) & (pop.x_pos < e) & (-e < pop.y_pos) & (pop.y_pos < e)
center_neurons = pop[mask]

ref_neuron = center_neurons[0]
t_corr = firing_rate_temporal_correlation(ref_neuron,
                                          pop,
                                          spike_recorder,
                                          t_bins)

fig, ax = plt.subplots(figsize=(12,10))
plot_firing_rate_temporal_correlation(fig, ax,
                                      (ref_neuron.x_pos, ref_neuron.y_pos),
                                      pop.x_pos, pop.y_pos,
                                      t_corr,
                                      cmap='bwr')

# +
from anaspike.analysis import pairwise_temporal_correlation_matrix
from anaspike.visualization import plot_pairwise_temporal_correlation_matrix

t_corr_mat = pairwise_temporal_correlation_matrix(pop, spike_recorder, t_bins)

fig, ax = plt.subplots()
plot_pairwise_temporal_correlation_matrix(fig, ax, t_corr_mat)

# +
from anaspike.analysis import hayleighs_spatial_autocorrelation
from anaspike.visualization import animate_hayleighs_spatial_autocorrelation

autocorr_for_each_t = hayleighs_spatial_autocorrelation(pop, spike_recorder, x_bins, y_bins, t_bins)

fig, ax = plt.subplots()
ani = animate_hayleighs_spatial_autocorrelation(fig, ax, autocorr_for_each_t, time_vals, cmap='bwr')
HTML(ani.to_jshtml())

# +
from anaspike.analysis import hayleighs_spatial_autocorrelation_radial_avg
from anaspike.visualization import animate_hayleighs_spatial_autocorrelation_radial_avg

radial_autocorr_for_each_t, distances = hayleighs_spatial_autocorrelation_radial_avg(pop, spike_recorder, x_bins, y_bins, t_bins, spatial_bin_size)

fig, ax = plt.subplots()
ani = animate_hayleighs_spatial_autocorrelation_radial_avg(fig, ax, radial_autocorr_for_each_t, distances, time_vals)
HTML(ani.to_jshtml())

# +
from anaspike.analysis import sigrids_spatial_autocorrelation
from anaspike.visualization import animate_sigrids_spatial_autocorrelation

autocorr_for_each_t = sigrids_spatial_autocorrelation(pop, spike_recorder, x_bins, y_bins, t_bins)

fig, ax = plt.subplots()
ani = animate_sigrids_spatial_autocorrelation(fig, ax, autocorr_for_each_t, time_vals, cmap='bwr')
HTML(ani.to_jshtml())

# +
from anaspike.analysis import sigrids_spatial_autocorrelation_radial_avg
from anaspike.visualization import animate_sigrids_spatial_autocorrelation_radial_avg

radial_autocorr_for_each_t = sigrids_spatial_autocorrelation_radial_avg(pop, spike_recorder, x_bins, y_bins, t_bins)

fig, ax = plt.subplots()
ani = animate_sigrids_spatial_autocorrelation_radial_avg(fig, ax, radial_autocorr_for_each_t, time_vals)
HTML(ani.to_jshtml())

# +
from anaspike.analysis import spike_counts_spatial_autocorrelation
from anaspike.visualization import animate_spike_counts_spatial_autocorrelation

spatial_corr = spike_counts_spatial_autocorrelation(pop, spike_recorder, x_bins, y_bins, t_bins)

fig, ax = plt.subplots(figsize=(12,10))
ani = animate_spike_counts_spatial_autocorrelation(fig, ax, x_bins, y_bins, spatial_corr, time_vals, cmap='bwr')
HTML(ani.to_jshtml())

# +
from anaspike.analysis import spike_counts_spatial_autocorrelation_radial_avg
from anaspike.visualization import animate_spike_counts_spatial_autocorrelation_radial_avg

radial_bins = EquiBins.from_interval_with_median_values(Interval(0., extent / 2), size=x_bins[0].width)
spatial_corr_radial_avg = spike_counts_spatial_autocorrelation_radial_avg(pop, spike_recorder, x_bins, y_bins, t_bins, origin, radial_bins)

fig, ax = plt.subplots()
ani = animate_spike_counts_spatial_autocorrelation_radial_avg(fig, ax, radial_bins, spatial_corr_radial_avg, time_vals)
HTML(ani.to_jshtml())

# +
from anaspike.analysis import morans_i_evolution
from anaspike.visualization import plot_morans_i_evolution

morans_i = morans_i_evolution(pop, spike_recorder, t_bins, decay_power=4)

fig, ax = plt.subplots()
plot_morans_i_evolution(ax, time_vals, morans_i)
