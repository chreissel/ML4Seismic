"""Coherence analysis of the microseismic sensor array.

Implements the coherence study of "Microseismic Noise Mitigation with Machine
Learning for Advanced LIGO" (arXiv:2511.19682): magnitude-squared coherence
between the target (GS13) channel and the witness sensors, band-averaged
pairwise coherence, and the time dependence of the coherence in the
microseismic (0.1-0.3 Hz) band.
"""

from .coherence import (
    pair_coherence,
    band_average,
    coherence_matrix,
    coherency_matrices,
    time_dependent_coherence,
)

__all__ = [
    "pair_coherence",
    "band_average",
    "coherence_matrix",
    "coherency_matrices",
    "time_dependent_coherence",
]
