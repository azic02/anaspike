from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray

from anaspike.dataclasses.interval import Interval, Bin
from anaspike.functions._helpers import construct_offsets



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


def animate_spike_counts_spatial_autocorrelation(fig: Figure, ax: Axes, xs: Sequence[Interval], ys: Sequence[Interval], spatial_autocorrelation: NDArray[np.float64], times: NDArray[np.float64], **kwargs) -> animation.FuncAnimation:
    def update_plot(i, data, scat, time):
        mesh.set_array(data[i])
        plt.title(f'elapsed time: {int(time[i])} ms')
        return scat,

    margin = 20
    x_offsets, y_offsets = construct_offsets(len(xs), margin), construct_offsets(len(ys), margin)
    correlation_formatted = np.transpose([[spatial_autocorrelation[:,i * len(y_offsets) + j] for j in range(len(y_offsets))] for i in range(len(x_offsets))])

    mesh = ax.pcolormesh(x_offsets, y_offsets, correlation_formatted[0], vmin=-1, vmax=1, **kwargs)
    ani = animation.FuncAnimation(fig, update_plot, frames=len(times), fargs=(correlation_formatted, mesh, times))

    fig.colorbar(mesh, ax=ax, label="Pearson's correlation coefficient")

    return ani


def animate_spike_counts_spatial_autocorrelation_radial_avg(fig: Figure, ax: Axes, radial_bins: Sequence[Bin], spatial_autocorr_radial_avg: NDArray[np.float64], times: NDArray[np.float64], **kwargs) -> animation.FuncAnimation:
    def update_plot(i, data, bars, time):
        for bar, h in zip(bars, data[i]):
            bar.set_height(h)
        plt.title(f'elapsed time: {int(time[i])} ms')
        return bars,

    bin_vals = [bin_.value for bin_ in radial_bins]
    bin_widths = [bin_.width for bin_ in radial_bins]
    bars = ax.bar(bin_vals, spatial_autocorr_radial_avg[0], width=bin_widths, align='center', **kwargs)
    ani = animation.FuncAnimation(fig, update_plot, frames=len(times), fargs=(spatial_autocorr_radial_avg, bars, times))

    ax.set_xlabel('distance (mm)')
    ax.set_ylabel('correlation coefficient')
    ax.set_ylim(-1, 1)


    return ani


def plot_morans_i_evolution(ax: Axes, times: NDArray[np.float64], morans_i: NDArray[np.float64], **kwargs) -> None:
    ax.plot(times, morans_i, **kwargs)
    ax.set_xlabel('time (ms)')
    ax.set_ylabel("Moran's I")
    ax.set_ylim(-1, 1)
    ax.set_title("Moran's I Evolution")
    ax.scatter(times, morans_i, **kwargs)


def animate_hayleighs_spatial_autocorrelation(fig: Figure, ax: Axes, spatial_autocorrelation: NDArray[np.float64], times: NDArray[np.float64], **kwargs) -> animation.FuncAnimation:
    def update_plot(i, data, img, time):
        img.set_array(data[i])
        plt.title(f'elapsed time: {int(time[i])} ms')
        return img,

    img = ax.imshow(spatial_autocorrelation[0], vmin=-1, vmax=1, **kwargs)
    ani = animation.FuncAnimation(fig, update_plot, frames=len(times), fargs=(spatial_autocorrelation, img, times))

    fig.colorbar(img, ax=ax, label='correlation coefficient')

    return ani


def animate_hayleighs_spatial_autocorrelation_radial_avg(fig: Figure, ax: Axes, spatial_autocorr_radial_avg: NDArray[np.float64], distances: NDArray[np.float64], times: NDArray[np.float64], **kwargs) -> animation.FuncAnimation:
    def update_plot(i, data, line, time):
        line.set_ydata(data[i])
        plt.title(f'elapsed time: {int(time[i])} ms')
        return line,

    line, = ax.plot(distances, spatial_autocorr_radial_avg[0], **kwargs)
    ani = animation.FuncAnimation(fig, update_plot, frames=len(times), fargs=(spatial_autocorr_radial_avg, line, times))

    ax.set_xlabel('distance (mm)')
    ax.set_ylabel('correlation coefficient')
    ax.set_xlim(0, max(distances))
    ax.set_ylim(-1, 1)

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


def animate_sigrids_spatial_autocorrelation_radial_avg(fig: Figure, ax: Axes, spatial_autocorr_radial_avg: NDArray[np.float64], times: NDArray[np.float64], **kwargs) -> animation.FuncAnimation:
    def update_plot(i, data, line, time):
        line.set_ydata(data[i])
        plt.title(f'elapsed time: {int(time[i])} ms')
        return line,

    line, = ax.plot(spatial_autocorr_radial_avg[0], **kwargs)
    ani = animation.FuncAnimation(fig, update_plot, frames=len(times), fargs=(spatial_autocorr_radial_avg, line, times))

    ax.set_ylabel('correlation coefficient')
    ax.set_xlim(0, len(spatial_autocorr_radial_avg[0]))
    ax.set_ylim(-1, 1)

    return ani

