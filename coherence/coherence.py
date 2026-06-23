"""Magnitude-squared coherence utilities.

The magnitude-squared coherence between two signals ``x`` and ``y`` is

    C_xy(f) = |P_xy(f)|^2 / (P_xx(f) P_yy(f))

with ``P`` the (cross-)power spectral densities (Welch estimate). It measures the
fraction of the power in ``y`` that is linearly predictable from ``x`` at each
frequency, and therefore bounds how well any *linear* filter can subtract the
witness contribution -- the link between this analysis and the linear baseline.
"""

import numpy as np
from scipy.signal import coherence as _scipy_coherence


def pair_coherence(x, y, fs, nperseg=None):
    """Magnitude-squared coherence between ``x`` and ``y``.

    Returns ``(freqs, Cxy)``.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if nperseg is None:
        nperseg = min(len(x), 256)
    f, cxy = _scipy_coherence(x, y, fs=fs, nperseg=nperseg)
    return f, cxy


def band_average(freqs, cxy, fmin, fmax):
    """Average a coherence spectrum over the band ``[fmin, fmax]``."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(cxy[mask]))


def coherence_matrix(data, fs, fmin, fmax, nperseg=None):
    """Band-averaged pairwise coherence matrix for all channels.

    Parameters
    ----------
    data : ndarray, shape (C, N)
    fs : float
    fmin, fmax : float
        Band over which the coherence is averaged.

    Returns
    -------
    M : ndarray, shape (C, C)
        Symmetric matrix of band-averaged coherence (diagonal = 1).
    """
    C = data.shape[0]
    M = np.eye(C)
    for i in range(C):
        for j in range(i + 1, C):
            f, cxy = pair_coherence(data[i], data[j], fs, nperseg)
            val = band_average(f, cxy, fmin, fmax)
            M[i, j] = M[j, i] = val
    return M


def time_dependent_coherence(data, fs, bands, n_segments, target_idx,
                             nperseg=None):
    """Average coherence of the target against all witnesses over time.

    The record is split into ``n_segments`` consecutive chunks; within each chunk
    the target-vs-witness coherence is band-averaged and then averaged across the
    witnesses, for every band in ``bands``.

    Parameters
    ----------
    data : ndarray, shape (C, N)
    fs : float
    bands : list[tuple[float, float]]
        Frequency bands ``(fmin, fmax)`` to track.
    n_segments : int
    target_idx : int

    Returns
    -------
    seg_times : ndarray, shape (n_segments,)
        Centre time [s] of each segment.
    series : dict[(fmin, fmax) -> ndarray (n_segments,)]
        Mean target-vs-witness coherence per band over time.
    """
    C, N = data.shape
    seg_len = N // n_segments
    if seg_len < (nperseg or 0) or seg_len < 8:
        raise ValueError(
            f"segment length {seg_len} too short for nperseg={nperseg}; "
            "reduce --n-segments")

    witness_idx = [c for c in range(C) if c != target_idx]
    seg_times = np.empty(n_segments)
    series = {b: np.empty(n_segments) for b in bands}

    for s in range(n_segments):
        lo = s * seg_len
        hi = lo + seg_len
        seg = data[:, lo:hi]
        seg_times[s] = (lo + hi) / 2.0 / fs
        per_band = {b: [] for b in bands}
        for w in witness_idx:
            f, cxy = pair_coherence(seg[target_idx], seg[w], fs, nperseg)
            for b in bands:
                per_band[b].append(band_average(f, cxy, b[0], b[1]))
        for b in bands:
            series[b][s] = np.nanmean(per_band[b])

    return seg_times, series
