# Linear noise subtraction (Wiener / FIR baseline)

This folder implements the **linear** microseismic-noise subtraction baseline
from *"Microseismic Noise Mitigation with Machine Learning for Advanced LIGO"*
([arXiv:2511.19682](https://arxiv.org/abs/2511.19682)). It is the conventional
linear-filtering reference against which the ML models (DeepClean / S4D) in the
parent repo are compared.

## Method

The target channel `y[t]` (the GS13 platform-motion residual, `target_idx=9`) is
modelled as a causal, linear, time-invariant combination of *past* samples of
the witness channels `x_c[t]`:

```
y_hat[t] = b + sum_c sum_{k=0}^{T-1} w_{c,k} * x_c[t - k]
```

with `T = n_taps` FIR taps per channel. The taps are found by Tikhonov-regularised
least squares (the discrete-time Wiener filter) on the **training** segment, and
the subtraction `residual = target - y_hat` is evaluated on the held-out **test**
segment. The filter is causal by default so it could be deployed on the LIGO
control computers; use `--acausal` for an offline centered filter.

## Usage

```bash
# from the repo root
python -m linear.run_linear                       # defaults (n_taps=64, ridge alpha=1e-3)
python -m linear.run_linear --n-taps 128 --alpha 1e-2
python -m linear.run_linear --acausal             # non-causal (offline) filter
python -m linear.run_linear --witness-idx 0 1 2 3 4 5 6 7 8   # use all witnesses
```

### Witness channel selection

By default the filter uses **GND + CPS in the targeted (X) direction**
(`--witness-idx 0 3` → `GND_STS_ITMY_X`, `HAM5_CPS_X`), matching the witness-only
LSTM (`configs/config_LSTM.yaml`); the non-targeted Y/Z directions are dropped.
The channel indices follow `data_prep.py`:

| idx | channel | idx | channel |
|-----|---------|-----|---------|
| 0 | GND_STS_ITMY_X | 5 | HAM5_CPS_Z |
| 1 | GND_STS_ITMY_Y | 6 | HAM5_CPSRX |
| 2 | GND_STS_ITMY_Z | 7 | HAM5_CPSRY |
| 3 | HAM5_CPS_X | 8 | HAM5_CPSRZ |
| 4 | HAM5_CPS_Y | 9 | HAM5_GS13X (target) |

Pass `--witness-idx` to change the set (e.g. `--witness-idx 0 1 2 3 4 5 6 7 8`
for all witnesses).

Key options: `--time`, `--data-dir`, `--target-idx`, `--witness-idx`, `--fs`,
`--n-taps`, `--alpha`, `--acausal`, `--fmin/--fmax` (microseismic band), `--outdir`.

## Outputs (`linear/results/`)

| file | contents |
|------|----------|
| `metrics.txt`   | band-limited RMS before/after + reduction factor (microseismic & broadband) |
| `asd.png`       | target amplitude spectral density before vs after subtraction |
| `timeseries.png`| target / linear prediction / residual time series |

The reduction factor in the 0.1–0.3 Hz band quantifies the linear performance;
the ML models aim to exceed it where nonlinear cross-couplings dominate.

## Programmatic API

```python
from linear import LinearSubtractor, band_rms
model = LinearSubtractor(n_taps=64, alpha=1e-3, causal=True).fit(witnesses, target)
pred, residual = model.clean(test_witnesses, test_target)
```
