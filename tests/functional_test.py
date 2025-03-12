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

# # Simulation Test

# +
import os

import matplotlib.pyplot as plt
import nest
# -

# ## Parametrisation

# ### neurons

# +
extent = (2., 2.)

neuron_params = {
    'C_m': 1.0,                        # membrane capacity (pF)
    'E_L': 0.,                         # resting membrane potential (mV)
    'I_e': 0.,                         # external input current (pA)
    'V_m': 0.,                         # membrane potential (mV)
    'V_reset': 10.,                    # reset membrane potential after a spike (mV)
    'V_th': 20.,                       # spike threshold (mV)
    't_ref': 2.0,                      # refractory period (ms)
    'tau_m': 20.,                      # membrane time constant (ms)
    }


positions_exc = nest.spatial.grid(
    shape=[120, 120],
    extent=extent,
    edge_wrap=True
    )

positions_inh = nest.spatial.grid(
    shape=[60, 60],
    extent=extent,
    edge_wrap=True
    )
# -

# ### connections

# #### synapses

# +
w_mean_exc = 1.
w_mean_inh = -5.

syn_spec_exc = {
    'synapse_model': 'static_synapse',
    'delay': 1.5,                      # synaptic transmission delay (ms)
    'weight': w_mean_exc
    }
syn_spec_inh = {
    'synapse_model': 'static_synapse',
    'delay': 1.5,                      # synaptic transmission delay (ms)
    'weight': w_mean_inh
    }
# -

# #### connectivity

conn_spec_exc = {
    'rule': 'pairwise_bernoulli',
    'p': nest.spatial_distributions.gaussian(nest.spatial.distance, std=0.2),
    }
conn_spec_inh = {
    'rule': 'pairwise_bernoulli',
    'p': nest.spatial_distributions.gaussian(nest.spatial.distance, std=0.5),
    }

# ### stimulation

poisson_generator_params = {
    'rate': 2000.
    }
poisson_generator_conn_spec = 'all_to_all'
poisson_generator_syn_spec = {'weight': w_mean_exc, 'delay': 1.0}

# ## Simulation

nest.ResetKernel()
nest.rng_seed = 1234
nest.resolution = 0.1
nest.local_num_threads = os.cpu_count()

pop_exc = nest.Create('iaf_psc_exp', params=neuron_params, positions=positions_exc)
pop_inh = nest.Create('iaf_psc_exp', params=neuron_params, positions=positions_inh)

nest.Connect(pop_exc, pop_exc, syn_spec=syn_spec_exc, conn_spec=conn_spec_exc)
nest.Connect(pop_exc, pop_inh, syn_spec=syn_spec_exc, conn_spec=conn_spec_exc)
nest.Connect(pop_inh, pop_exc, syn_spec=syn_spec_inh, conn_spec=conn_spec_inh)
nest.Connect(pop_inh, pop_inh, syn_spec=syn_spec_inh, conn_spec=conn_spec_inh)

nest.PlotTargets(pop_exc[500], pop_exc, probability_parameter=conn_spec_exc['p'], src_size=250, tgt_color='moccasin', tgt_size=20, probability_cmap='Purples')
plt.show()

nest.PlotTargets(pop_exc[500], pop_inh, probability_parameter=conn_spec_exc['p'], src_size=250, tgt_color='moccasin', tgt_size=20, probability_cmap='Purples')
plt.show()

nest.PlotTargets(pop_inh[500], pop_exc, probability_parameter=conn_spec_inh['p'], src_size=250, tgt_color='moccasin', tgt_size=20, probability_cmap='Purples')
plt.show()

nest.PlotTargets(pop_inh[500], pop_inh, probability_parameter=conn_spec_inh['p'], src_size=250, tgt_color='moccasin', tgt_size=20, probability_cmap='Purples')
plt.show()

poisson_generator = nest.Create('poisson_generator', params=poisson_generator_params)
nest.Connect(poisson_generator, pop_exc, poisson_generator_conn_spec, poisson_generator_syn_spec)
nest.Connect(poisson_generator, pop_inh, poisson_generator_conn_spec, poisson_generator_syn_spec)

spike_recorder = nest.Create('spike_recorder')
nest.Connect(pop_exc, spike_recorder)
nest.Connect(pop_inh, spike_recorder)

t_sim = 10000.
nest.Simulate(t_sim)

# # Analysis Test

# +
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML

from anaspike.dataclasses.interval import Interval

# +
from pathlib import Path
from anaspike.dataclasses import SimulationData
from anaspike.dataclasses.nest_devices import PopulationData, SpikeRecorderData

simulation_data = SimulationData(
    populations={'exc': PopulationData.from_pynest(pop_exc), 'inh': PopulationData.from_pynest(pop_inh)},
    spike_recorder=SpikeRecorderData.from_pynest(spike_recorder)
)

populations = simulation_data.populations
spike_recorder = simulation_data.spike_recorder
# -

# ## pre-analysis analysis

# +
from anaspike.visualization import spike_raster_plot

fig, ax = plt.subplots(figsize=(14, 4))
spike_raster_plot(ax, spike_recorder.times, spike_recorder.senders, markersize=1.)

# +
from anaspike.analysis import intra_population_average_firing_rate_evolution
from anaspike.visualization import plot_intra_population_average_firing_rate_evolution

t_bins = Interval(0., 2000.).bin(size=10.)
names, avgs, stds = intra_population_average_firing_rate_evolution(populations, spike_recorder, t_bins)

fig, ax = plt.subplots(figsize=(14, 4))
plot_intra_population_average_firing_rate_evolution(ax, [t_bin.value for t_bin in t_bins], names, avgs, stds)
# -

# ## analysis parameters

# +
pop = populations['inh']

extent = 2.
t_sim = 10000.

n_spatial_bins = 50

t_bins = Interval(100., t_sim).bin(size=500.)
x_bins = Interval(-extent / 2., extent / 2.).bin(n_spatial_bins)
y_bins = Interval(-extent / 2., extent / 2.).bin(n_spatial_bins)

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

max_offset_x = len(x_bins) - 20
max_offset_y = len(y_bins) - 20
spatial_corr = spike_counts_spatial_autocorrelation(pop,
                                                    spike_recorder,
                                                    x_bins,
                                                    y_bins,
                                                    t_bins,
                                                    max_offset_x,
                                                    max_offset_y)

fig, ax = plt.subplots(figsize=(12,10))
ani = animate_spike_counts_spatial_autocorrelation(fig,
                                                   ax,
                                                   spatial_corr,
                                                   max_offset_x,
                                                   max_offset_y,
                                                   time_vals,
                                                   cmap='bwr')
HTML(ani.to_jshtml())
