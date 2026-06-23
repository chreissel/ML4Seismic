# Coherence analysis

This folder implements the coherence study from *"Microseismic Noise Mitigation
with Machine Learning for Advanced LIGO"*
([arXiv:2511.19682](https://arxiv.org/abs/2511.19682)). The magnitude-squared
coherence between the witness sensors and the target (GS13) channel quantifies
how much of the target motion is *linearly* predictable from the witnesses, and
its time dependence motivates the move from linear filters to ML models.

## Method

The magnitude-squared coherence (Welch estimate) is

```
C_xy(f) = |P_xy(f)|^2 / (P_xx(f) * P_yy(f))
```

The analysis produces three views:

1. **Coherence spectra** — `C(f)` between the target and each witness.
2. **Pairwise coherence matrix** — band-averaged coherence between every pair of
   channels in the microseismic band.
3. **Time-dependent coherence** — the record is split into consecutive segments
   and the mean target-vs-witness coherence is tracked per segment, comparing the
   broad available band with the 0.1–0.3 Hz microseismic band. This reproduces
   the paper's observation of *time-dependent cross couplings* in the
   microseismic band.

## Usage

```bash
# from the repo root
python -m coherence.run_coherence                      # train split, defaults
python -m coherence.run_coherence --split test --n-segments 4
```

Key options: `--time`, `--data-dir`, `--split`, `--target-idx`, `--fs`,
`--nperseg`, `--micro-fmin/--micro-fmax`, `--broad-fmin/--broad-fmax`,
`--n-segments`, `--outdir`.

### Frequency-range note

The provided `.npy` data is resampled to 4 Hz and bandpassed to 0.1–0.3 Hz, so
the default "broad" band here is the available 0.1–2 Hz. For the paper's full
0.1–100 Hz vs 0.1–0.3 Hz comparison, point `--data-dir`/`--fs` at broadband
(e.g. 128 Hz) data and raise `--broad-fmax` accordingly.

## Outputs (`coherence/results/`)

| file | contents |
|------|----------|
| `coherence_spectra.png` | target-vs-witness coherence vs frequency |
| `coherence_matrix.png`  | band-averaged pairwise coherence heatmap |
| `coherence_time.png`    | mean coherence over time, broad vs microseismic |
| `summary.txt`           | band-averaged coherence values + time series |

## Paper figure

`reproduce_cov.py` builds the 3-panel coherence-matrix figure (broad band, narrow
band at t1, narrow band at t2). See [../FIGURES.md](../FIGURES.md).

## Programmatic API

```python
from coherence import pair_coherence, band_average, coherence_matrix, time_dependent_coherence
f, cxy = pair_coherence(target, witness, fs=4.0)
mean_micro = band_average(f, cxy, 0.1, 0.3)
```
