"""Multichannel FIR Wiener filter for linear noise subtraction.

The target channel ``y[t]`` (the GS13 platform-motion residual) is modelled as a
linear, time-invariant combination of *past* samples of the witness channels
``x_c[t]``::

    y_hat[t] = b + sum_c sum_{k=0}^{T-1} w_{c,k} * x_c[t - k]

with ``T = n_taps`` filter taps per channel. The taps ``w_{c,k}`` and bias ``b``
are found by (Tikhonov-regularised) least squares on the training segment, which
is exactly the discrete-time Wiener filter / linear regression referred to in the
paper. The filter is *causal* by default so that it can in principle be deployed
on the LIGO control computers, but an acausal (centered) variant is available for
offline analysis.
"""

import numpy as np
from scipy.signal import welch


def _shift(x, k):
    """Return ``out`` with ``out[t] = x[t - k]``, zero padded at the edges.

    ``k > 0`` introduces a delay (uses past samples), ``k < 0`` an advance.
    """
    out = np.zeros_like(x)
    if k == 0:
        out[:] = x
    elif k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out


def build_design_matrix(witnesses, n_taps, causal=True):
    """Build the lagged design matrix for a multichannel FIR filter.

    Parameters
    ----------
    witnesses : ndarray, shape (C, N)
        Witness channels (already standardised by the caller).
    n_taps : int
        Number of FIR taps per channel.
    causal : bool
        If True use lags ``0 .. n_taps-1`` (past samples only). If False use a
        symmetric, centered window of lags.

    Returns
    -------
    X : ndarray, shape (N, C * n_taps)
    """
    C = witnesses.shape[0]
    if causal:
        lags = range(0, n_taps)
    else:
        half = n_taps // 2
        lags = range(-half, n_taps - half)

    cols = []
    for c in range(C):
        for k in lags:
            cols.append(_shift(witnesses[c], k))
    return np.stack(cols, axis=1)


class LinearSubtractor:
    """Fit and apply a multichannel FIR Wiener filter.

    Parameters
    ----------
    n_taps : int
        Number of FIR taps per witness channel.
    alpha : float
        Tikhonov (ridge) regularisation strength on the standardised problem.
    causal : bool
        Whether the filter only uses past witness samples.
    """

    def __init__(self, n_taps=64, alpha=1e-3, causal=True):
        self.n_taps = int(n_taps)
        self.alpha = float(alpha)
        self.causal = bool(causal)
        # learned parameters
        self.w = None              # (C * n_taps,) filter taps
        self.bias = 0.0
        self.x_mean = None         # (C, 1) witness standardisation
        self.x_std = None
        self.y_mean = 0.0

    def _standardise(self, witnesses):
        return (witnesses - self.x_mean) / self.x_std

    def fit(self, witnesses, target):
        """Fit the filter on a (C, N) witness array and (N,) target.

        Returns ``self``.
        """
        witnesses = np.asarray(witnesses, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64).reshape(-1)

        # per-channel standardisation keeps the ridge penalty well scaled
        self.x_mean = witnesses.mean(axis=1, keepdims=True)
        self.x_std = witnesses.std(axis=1, keepdims=True)
        self.x_std[self.x_std == 0] = 1.0
        self.y_mean = float(target.mean())

        Xw = self._standardise(witnesses)
        X = build_design_matrix(Xw, self.n_taps, self.causal)
        y = target - self.y_mean

        # normal equations with ridge: (X^T X + alpha I) w = X^T y
        A = X.T @ X
        A += self.alpha * np.trace(A) / A.shape[0] * np.eye(A.shape[0])
        b = X.T @ y
        self.w = np.linalg.solve(A, b)
        self.bias = self.y_mean
        return self

    def predict(self, witnesses):
        """Predict the target from a (C, N) witness array. Returns (N,)."""
        if self.w is None:
            raise RuntimeError("LinearSubtractor must be fit before predict().")
        witnesses = np.asarray(witnesses, dtype=np.float64)
        Xw = self._standardise(witnesses)
        X = build_design_matrix(Xw, self.n_taps, self.causal)
        return X @ self.w + self.bias

    def clean(self, witnesses, target):
        """Return ``(prediction, residual)`` where ``residual = target - pred``."""
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        pred = self.predict(witnesses)
        return pred, target - pred


def asd(x, fs, nperseg=None):
    """Amplitude spectral density via Welch. Returns ``(freqs, asd)``."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if nperseg is None:
        nperseg = min(len(x), 256)
    f, pxx = welch(x, fs=fs, nperseg=nperseg)
    return f, np.sqrt(pxx)


def band_rms(x, fs, fmin, fmax, nperseg=None):
    """Band-limited RMS of ``x`` computed by integrating the PSD over [fmin, fmax]."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if nperseg is None:
        nperseg = min(len(x), 256)
    f, pxx = welch(x, fs=fs, nperseg=nperseg)
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.trapezoid(pxx[mask], f[mask])))
