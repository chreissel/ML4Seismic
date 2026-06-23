# Reproducing the linear-analysis figures

This guide recreates the three figures of the "linear baseline" section of
*Microseismic Noise Mitigation with Machine Learning for Advanced LIGO*
([arXiv:2511.19682](https://arxiv.org/abs/2511.19682)):

| paper figure | script | output |
|---|---|---|
| Fig. `performance` (3-panel ASD: top-hat / f² / causal + prediction + floor) | `python -m linear.reproduce_performance` | `linear/results/linear_performance.{png,pdf}` |
| Fig. `cov` (3 coherence matrices: broad / narrow@t1 / narrow@t2) | `python -m coherence.reproduce_cov` | `coherence/results/coherence_cov.{png,pdf}` |
| Fig. `spike` (coefficient evolution from rolling refits) | `python -m linear.coefficient_evolution` | `linear/results/coefficient_evolution.{png,pdf}` |

All three share `seismic_data.py` (raw `.mat` loader, the three bandpass filters,
and the multiple-coherence floor).

## Data: raw `.mat` vs shipped `.npy`

For **full fidelity** use the raw broadband file
`MLdata_L1HAM5_1381528818_4000_matrix_v2.mat` (128 Hz, all channels). Only this
supports the 0.1–100 Hz panel and the six GS13 degrees of freedom:

```bash
python -m linear.reproduce_performance --mat MLdata_L1HAM5_1381528818_4000_matrix_v2.mat
python -m coherence.reproduce_cov      --mat MLdata_L1HAM5_1381528818_4000_matrix_v2.mat --broad 0.1 100 --narrow 0.1 0.3
python -m linear.coefficient_evolution --mat MLdata_L1HAM5_1381528818_4000_matrix_v2.mat --step 10
```

The shipped `data/*_1381528818.npy` is a quick stand-in but is sampled at 4 Hz
(Nyquist 2 Hz), bandpassed to 0.1–0.3 Hz, and only contains `GS13X`, so the broad
panel and multi-DOF are unavailable:

```bash
python -m linear.reproduce_performance --npy data/train_1381528818.npy --npy-fs 4
python -m coherence.reproduce_cov      --npy data/train_1381528818.npy --npy-fs 4 --broad 0.1 1.9 --t1 0 --t2 120 --seg-len 120
python -m linear.coefficient_evolution --npy data/train_1381528818.npy --npy-fs 4 --step 60 --win 120
```

### Channel map (important for the raw `.mat`)

`seismic_data.py` maps channel names to `.mat` columns. `DEFAULT_CHANNEL_MAP`
matches `data_prep.py` (10 channels at columns 0–9). The paper's figures use the
**six GS13 DOF** plus CPS+GND witnesses, so if your `.mat` carries the extra GS13
columns, supply them with a JSON override:

```json
{
  "L1:ISI-GND_STS_ITMY_X_DQ": 0,
  "...": 1,
  "L1:ISI-HAM5_BLND_GS13X_IN1_DQ": 9,
  "L1:ISI-HAM5_BLND_GS13Y_IN1_DQ": 10,
  "L1:ISI-HAM5_BLND_GS13Z_IN1_DQ": 11
}
```

```bash
python -m linear.reproduce_performance --mat your.mat --channel-map map.json \
    --target 'L1:ISI-HAM5_BLND_GS13X_IN1_DQ'
```

The scripts fit **one** target at a time; loop `--target` over the six DOF to
cover all of them (the figure itself shows the X direction).

## Per-figure details

### Fig. `performance` — `linear.reproduce_performance`
For each of the three filters (`tophat`, `f2`, `causal`) the witnesses and target
are band-limited, a linear regression (`--n-taps 1` = instantaneous) predicts the
target, and the panel shows the target ASD (blue), prediction (green), residual
(orange) and the floor `ASD_target · √(1−γ_M²(f))` (dashed). The top-hat panel is
the non-realizable best case; the causal panel is the realizable baseline.
Key options: `--fmin/--fmax`, `--n-taps`, `--alpha`, `--witness`, `--nperseg`.

### Fig. `cov` — `coherence.reproduce_cov`
Three band-averaged pairwise coherence matrices over all channels: broad band at
`--t1`, narrow band at `--t1`, and narrow band at `--t2`. The narrow-band panels
differ between times, revealing the time-dependent cross coupling.
Key options: `--broad`, `--narrow`, `--seg-len`, `--t1`, `--t2`, `--channels`.

### Fig. `spike` — `linear.coefficient_evolution`
Refits the regression every `--step` seconds (window `--win`, default = step),
records each coefficient and its OLS `±1σ` uncertainty, and plots the `--top`
most-varying coefficients vs time. Witnesses are globally standardised so the
coefficients are comparable; the fluctuations exceed the error bands, indicating
genuine non-stationarity.
Key options: `--step`, `--win`, `--filter`, `--top`, `--fmin/--fmax`.

> Outputs land in `linear/results/` and `coherence/results/`, which are
> git-ignored. Both `.png` and `.pdf` are written.
