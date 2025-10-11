import numpy as np
from numpy.typing import NDArray

from .firing_rates_evolution import FiringRatesEvolution, FiringRateEvolution
from ..dataclasses.scalar_spatio_temporal_map import temporal_correlation as temporal_correlation_stm
from ..dataclasses.scalar_spatio_temporal_map import temporal_correlation_matrix as temporal_correlation_matrix_stm
from ..dataclasses.scalar_spatio_temporal_map import morans_i as morans_i_stm
from ..dataclasses.scalar_temporal_map import ScalarTemporalMap
from ..dataclasses.scalar_spatial_map import ScalarSpatialMap
from anaspike.dataclasses.coords2d import Coords2D



def temporal_correlation(fr: FiringRatesEvolution,
                         ref: FiringRateEvolution) -> ScalarSpatialMap[Coords2D]:
    return temporal_correlation_stm(fr, ref)


def temporal_correlation_matrix(firing_rates: FiringRatesEvolution) -> NDArray[np.float64]:
    return temporal_correlation_matrix_stm(firing_rates)


def morans_i_evolution(firing_rates: FiringRatesEvolution) -> ScalarTemporalMap:
    return morans_i_stm(firing_rates)

