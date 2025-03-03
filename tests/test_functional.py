import unittest
from pathlib import Path
import sys
import importlib.util

import numpy as np
import matplotlib.pyplot as plt

from anaspike.dataclasses.interval import Interval
from anaspike.dataclasses import SimulationData
from anaspike.dataclasses.nest_devices import PopulationData, SpikeRecorderData
from anaspike.analysis import (firing_rates_evolution,
                               time_avg_firing_rates_histogram,
                               intra_population_average_firing_rate_evolution,
                               spike_gaps_coefficient_of_variation_histogram,
                               firing_rate_temporal_correlation,
                               pairwise_temporal_correlation_matrix,
                               spike_counts_spatial_autocorrelation,
                               hayleighs_spatial_autocorrelation
                               )
from anaspike.visualization import (spike_raster_plot,
                                    animate_firing_rate_evolution,
                                    plot_time_avg_firing_rate_histogram,
                                    plot_intra_population_average_firing_rate_evolution,
                                    plot_spike_gaps_coefficient_of_variation_histogram,
                                    plot_firing_rate_temporal_correlation,
                                    plot_pairwise_temporal_correlation_matrix,
                                    animate_spike_counts_spatial_autocorrelation,
                                    animate_hayleighs_spatial_autocorrelation
                                    )

import nest



def load_parameters(parameters_path: Path) -> 'parameters':
    spec = importlib.util.spec_from_file_location('parameters', parameters_path)
    parameters = importlib.util.module_from_spec(spec)
    sys.modules['parameters'] = parameters
    spec.loader.exec_module(parameters)
    return parameters


def run_simulation(parameters_path: Path, target_plots_path: Path) -> SimulationData:

    # load parameters
    parameters = load_parameters(parameters_path)

    # configure kernel
    nest.ResetKernel()

    # create populations
    populations = {}
    for name, params in parameters.populations.items():
        populations[name] = nest.Create('iaf_psc_delta', **params['neuronal'])

    # connect populations
    for src, src_params in parameters.populations.items():
        for trg, _ in parameters.populations.items():
            nest.Connect(populations[src], populations[trg], **src_params['synaptic'])

    # plot connectivity
    if target_plots_path is not None:
        for src, src_params in parameters.populations.items():
            for trg, _ in parameters.populations.items():
                fig = plot_targets(populations[src][50], populations[trg], src_params['synaptic']['conn_spec'])
                fig.savefig(target_plots_path / f'conn-{src}-{trg}.png')

    # create stimulation device
    poisson_generator = nest.Create('poisson_generator', params=parameters.poisson_generator['params'])

    # connect stimulation device
    for n in populations:
        nest.Connect(poisson_generator, populations[n], syn_spec=parameters.poisson_generator['syn_spec'])

    # create recording device
    spike_recorder = nest.Create('spike_recorder')

    # connect recording device
    for n in populations:
        nest.Connect(populations[n], spike_recorder)

    # simulate
    nest.Simulate(parameters.simulation['duration'])

    return SimulationData(
        populations={name: PopulationData.from_pynest(pop) for name, pop in populations.items()},
        spike_recorder=SpikeRecorderData.from_pynest(spike_recorder),
        )


def plot_targets(src_neuron: nest.NodeCollection, trg_population: nest.NodeCollection, conn_spec: dict, nodecolor=None):
    fig = nest.PlotLayer(trg_population, nodecolor=nodecolor, nodesize=40)

    prob_param = conn_spec.get('p', None)
    if type(prob_param) != nest.Parameter:
        prob_param = None
    mask = conn_spec.get('mask', None)

    nest.PlotTargets(src_neuron, trg_population, probability_parameter=prob_param, mask=mask, fig=fig, src_size=250, tgt_color='moccasin', tgt_size=20, probability_cmap='Purples')
    if prob_param is not None:
        fig.axes[1].set_ylabel('connection probability')

    return fig


class TestFunctional(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters_path = Path('./tests/test_simulation_parameters.py')
        cls.plots_path = Path('./tests/plots')
        cls.simulation_data = run_simulation(cls.parameters_path, cls.plots_path)
        cls.parameters = load_parameters(cls.parameters_path)
        cls.spike_recorder = cls.simulation_data.spike_recorder
        cls.populations = cls.simulation_data.populations

    def test_spike_raster_plot(self):
        fig, ax = plt.subplots()
        spike_raster_plot(ax, self.spike_recorder.times, self.spike_recorder.senders, markersize=0.05)
        fig.savefig(self.plots_path / 'spike_raster.png')

    def test_firing_rates_evolution(self):
        t_bins = Interval(0., self.parameters.simulation['duration']).bin(100)
        times = np.array([t.value for t in t_bins])
        for n, p in self.populations.items():
            firing_rates_at_times = firing_rates_evolution(p, self.spike_recorder, t_bins)
            fig, ax = plt.subplots()
            ani = animate_firing_rate_evolution(fig, ax, p.x_pos, p.y_pos, times, firing_rates_at_times)
            ani.save(self.plots_path / f'firing_rates_evolution_{n}.gif', fps=20)

    def test_time_avg_firing_rates_histogram(self):
        t_interval = Interval(0., self.parameters.simulation['duration'])
        for n, p in self.populations.items():
            bin_vals, bin_counts, bin_edges = time_avg_firing_rates_histogram(p, self.spike_recorder, t_interval, 10)
            fig, ax = plt.subplots()
            plot_time_avg_firing_rate_histogram(ax, bin_vals, bin_counts, np.diff(bin_edges))
            fig.savefig(self.plots_path / f'time_avg_firing_rates_histogram_{n}.png')

    def test_intra_population_average_firing_rate_evolution(self):
        ts = Interval(0., self.parameters.simulation['duration']).bin(100)
        times = np.array([t.value for t in ts])
        names, avgs, stds = intra_population_average_firing_rate_evolution(self.populations, self.spike_recorder, ts)
        fig, ax = plt.subplots()
        plot_intra_population_average_firing_rate_evolution(ax, times, names, avgs, stds)
        fig.savefig(self.plots_path / 'inter_population_average_firing_rate_evolution.png')

    def test_spike_gaps_coefficient_of_variation_histogram(self):
        for n, p in self.populations.items():
            bin_vals, bin_counts, bin_edges = spike_gaps_coefficient_of_variation_histogram(p, self.spike_recorder, 10)
            fig, ax = plt.subplots()
            plot_spike_gaps_coefficient_of_variation_histogram(ax, bin_vals, bin_counts, np.diff(bin_edges))
            fig.savefig(self.plots_path / f'spike_gaps_coefficient_of_variation_histogram_{n}.png')

    def test_firing_rate_temporal_correlation(self):
        t_bins = Interval(0., self.parameters.simulation['duration']).bin(100)
        for n, p in self.populations.items():
            ref_neuron = p[50]
            correlation = firing_rate_temporal_correlation(ref_neuron, p, self.spike_recorder, t_bins)
            fig, ax = plt.subplots()
            plot_firing_rate_temporal_correlation(fig, ax, (ref_neuron.x_pos, ref_neuron.y_pos), p.x_pos, p.y_pos, correlation)
            fig.savefig(self.plots_path / f'firing_rate_temporal_correlation_{n}.png')

    def test_pairwise_temporal_correlation_matrix(self):
        t_bins = Interval(0., self.parameters.simulation['duration']).bin(100)
        for n, p in self.populations.items():
            correlation_matrix = pairwise_temporal_correlation_matrix(p, self.spike_recorder, t_bins)
            fig, ax = plt.subplots()
            plot_pairwise_temporal_correlation_matrix(fig, ax, correlation_matrix)
            fig.savefig(self.plots_path / f'pairwise_temporal_correlation_matrix_{n}.png')

    def test_spike_counts_spatial_autocorrelation(self):
        n_spatial_bins = 5
        x_bins = Interval(-self.parameters.spatial['extent']['x'] / 2.,
                          +self.parameters.spatial['extent']['x'] / 2.).bin(n_spatial_bins)
        y_bins = Interval(-self.parameters.spatial['extent']['y'] / 2.,
                          +self.parameters.spatial['extent']['y'] / 2.).bin(n_spatial_bins)
        t_bins = Interval(0., self.parameters.simulation['duration']).bin(100)
        max_offset = n_spatial_bins - 2

        for n, p in self.populations.items():
            spatial_autocorrelation = spike_counts_spatial_autocorrelation(p, self.spike_recorder, x_bins, y_bins, t_bins, max_offset, max_offset)
            fig, ax = plt.subplots()
            ani = animate_spike_counts_spatial_autocorrelation(fig, ax, spatial_autocorrelation, max_offset, max_offset, np.array([t.value for t in t_bins]))
            ani.save(self.plots_path / f'spike_counts_spatial_autocorrelation_{n}.gif', fps=20)

    def test_hayleighs_spatial_autocorrelation(self):
        n_spatial_bins = 5
        x_bins = Interval(-self.parameters.spatial['extent']['x'] / 2.,
                          +self.parameters.spatial['extent']['x'] / 2.).bin(n_spatial_bins)
        y_bins = Interval(-self.parameters.spatial['extent']['y'] / 2.,
                          +self.parameters.spatial['extent']['y'] / 2.).bin(n_spatial_bins)
        t_bins = Interval(0., self.parameters.simulation['duration']).bin(100)

        for n, p in self.populations.items():
            spatial_autocorrelation = hayleighs_spatial_autocorrelation(p, self.spike_recorder, x_bins, y_bins, t_bins)
            fig, ax = plt.subplots()
            ani = animate_hayleighs_spatial_autocorrelation(fig, ax, spatial_autocorrelation, np.array([t.value for t in t_bins]))
            ani.save(self.plots_path / f'hayleighs_spatial_autocorrelation_{n}.gif', fps=20)


if __name__ == '__main__':
    unittest.main(verbosity=2)

