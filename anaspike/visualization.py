from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray



def spike_raster_plot(ax: Axes, times: NDArray[np.float64], senders: NDArray[np.int64], markersize: float) -> None:
    ax.plot(times, senders, 'k.', markersize=markersize)
    ax.set_xlim(0, max(times))
    ax.set_ylim(0, max(senders))
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('neuron id')


def animate_firing_rate_evolution(fig: Figure, ax: Axes, x_pos: NDArray[np.float64], y_pos: NDArray[np.float64], times: NDArray[np.float64], firing_rates_at_times: NDArray[np.float64], **kwargs) -> animation.FuncAnimation:
    def update_plot(i, data, scat, time):
        scat.set_array(data[i])
        plt.title(f'elapsed time: {int(time[i])} ms')
        return scat,

    scat = ax.scatter(x=x_pos, y=y_pos, c=firing_rates_at_times[0], vmin=0., **kwargs)
    ani = animation.FuncAnimation(fig, update_plot, frames=len(times), fargs=(firing_rates_at_times, scat, times))
    fig.colorbar(scat, ax=ax, label='firing rate [Hz]')
    return ani


def plot_time_avg_firing_rate_histogram(ax: Axes, bin_vals: NDArray[np.float64], bin_counts: NDArray[np.int64], bin_widths: NDArray[np.float64]) -> None:
    ax.bar(bin_vals, bin_counts, width=bin_widths, align='center')
    ax.set_xlabel("Time-averaged firing rate (Hz)")
    ax.set_ylabel("Number of neurons")
    ax.set_title("Time-averaged firing rate distribution")


def plot_intra_population_average_firing_rate_evolution(ax: Axes, times: NDArray[np.float64], names: Iterable[str], avgs: Iterable[NDArray[np.float64]], stds: Iterable[NDArray[np.float64]]) -> None:
    for name, avg, std in zip(names, avgs, stds):
        ax.plot(times, avg, label=name)
        ax.fill_between(times, avg - std, avg + std, alpha=0.3)
    ax.legend()
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('average firing rate (Hz)')


def plot_spike_gaps_coefficient_of_variation_histogram(ax: Axes, bin_vals: NDArray[np.float64], bin_counts: NDArray[np.int64], bin_widths: NDArray) -> None:
    ax.bar(bin_vals, bin_counts, width=bin_widths, align='center')
    ax.set_xlabel('Coefficient of variation')
    ax.set_ylabel('Number of neurons')
    ax.set_title('Coefficient of variation of spike gaps distribution across neurons')


def plot_firing_rate_temporal_correlation(fig: Figure, ax: Axes, ref_pos: Tuple[float,float], x_pos: NDArray[np.float64], y_pos: NDArray[np.float64], correlation: NDArray[np.float64], **kwargs) -> None:
    scat = ax.scatter(x=x_pos, y=y_pos, c=correlation, vmin=-1, vmax=1, **kwargs)
    ax.scatter(ref_pos[0], ref_pos[1], facecolors='none', edgecolors='green', s=30, linewidth=5)
    ax.set_xlabel('cortical space x (mm)')
    ax.set_ylabel('cortical space y (mm)')
    fig.colorbar(scat, ax=ax, label="Pearson's correlation coefficient")


def plot_pairwise_temporal_correlation_matrix(fig: Figure, ax: Axes, correlation_matrix: NDArray[np.float64]) -> None:
    img = ax.matshow(correlation_matrix, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_title('Pairwise Firing Rate Temporal Correlation Matrix')
    fig.colorbar(img, ax=ax, label="pearson's correlation coefficient")


def animate_spike_counts_spatial_autocorrelation(fig: Figure, ax: Axes, spatial_autocorrelation, max_x_offset: int, max_y_offset, times: NDArray[np.float64], **kwargs) -> animation.FuncAnimation:
    def update_plot(i, data, scat, time):
        mesh.set_array(data[i])
        plt.title(f'elapsed time: {int(time[i])} ms')
        return scat,

    x_offsets = np.arange(-max_x_offset, max_x_offset + 1)
    y_offsets = np.arange(-max_y_offset, max_y_offset + 1)
    mesh = ax.pcolormesh(x_offsets, y_offsets, spatial_autocorrelation[0], vmin=-1, vmax=1, **kwargs)
    ani = animation.FuncAnimation(fig, update_plot, frames=len(times), fargs=(spatial_autocorrelation, mesh, times))

    fig.colorbar(mesh, ax=ax, label="Pearson's correlation coefficient")

    return ani


def animate_hayleighs_spatial_autocorrelation(fig: Figure, ax: Axes, spatial_autocorrelation: NDArray[np.float64], times: NDArray[np.float64], **kwargs) -> animation.FuncAnimation:
    def update_plot(i, data, img, time):
        img.set_array(data[i])
        plt.title(f'elapsed time: {int(time[i])} ms')
        return img,

    img = ax.imshow(spatial_autocorrelation[0], vmin=-1, vmax=1, **kwargs)
    ani = animation.FuncAnimation(fig, update_plot, frames=len(times), fargs=(spatial_autocorrelation, img, times))

    fig.colorbar(img, ax=ax, label='correlation coefficient')

    return ani


def animate_sigrids_spatial_autocorrelation(fig: Figure, ax: Axes, spatial_autocorrelation: NDArray[np.float64], times: NDArray[np.float64], **kwargs) -> animation.FuncAnimation:
    def update_plot(i, data, img, time):
        img.set_array(data[i])
        plt.title(f'elapsed time: {int(time[i])} ms')
        return img,

    img = ax.imshow(spatial_autocorrelation[0], vmin=-1, vmax=1, **kwargs)
    ani = animation.FuncAnimation(fig, update_plot, frames=len(times), fargs=(spatial_autocorrelation, img, times))

    fig.colorbar(img, ax=ax, label='correlation coefficient')

    return ani

