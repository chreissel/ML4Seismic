"""Linear (Wiener / FIR least-squares) microseismic noise subtraction.

This package implements the *linear* baseline used in
"Microseismic Noise Mitigation with Machine Learning for Advanced LIGO"
(arXiv:2511.19682). A multichannel FIR Wiener filter is fit by least squares
to predict the target platform-motion channel (GS13) from the auxiliary
witness sensors. Subtracting the prediction yields the cleaned residual.
"""

from .wiener import LinearSubtractor, band_rms, asd

__all__ = ["LinearSubtractor", "band_rms", "asd"]
