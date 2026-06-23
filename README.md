# Microseismic noise suppression with Machine Learning (ML4Seismic)

This repo contains all the code, including test versions, to learn the microseismic motion from monitoring seismometers. It is based on `PyTorch Lightning` to facilitate training, testing, and evaluation. An introduction to `PyTorch Lightning` can be found [here](https://lightning.ai/docs/pytorch/stable/starter/introduction.html). This README will guide you through the steps to prepare the inputs, configure and run a training, and access the results.

### Data preparation
This will be incorporated later. Currently, please use the provided `.npz` files in the [data/](data) folder. If downloaded correctly, no further processing is needed at the moment.

### Training
`PyTorch Lightning` trainings are configured via a config file. An example config file is available in [config/config_DeepClean.yaml](config/config_DeepClean.yaml) and can be easily adopted for tests. It particularly needs to specify the input and output dimensionality (aka the number of observing channels and the number of output channels to fit). The training will automatically start by running
```
python cli.py fit -c configs/config.yaml
```
The pipeline automatically checks for GPUs and submits accordingly. While GPUs are not strictly required (all networks are small), they are recommended for speed-ups. All training characteristics are tracked via [weights and biases](https://wandb.ai/), so we recommend signing up for optimal user experience.

### Evaluation
The `jupyter` notebook [eval.ipynb](eval.ipynb) implements basic model loading and evaluation alongside diagnostics plots.

### LSTM (witness-only regression)
The `LSTMForecast` model ([model.py](model.py)) predicts the GS13 target from the
witness sensors. It is configured as a **witness-only regressor** (it ingests the
GND/CPS witness channels and now-casts the target), rather than an autoregressive
forecaster of the target's own past. By default it uses GND + CPS in the targeted
(X) direction; the witness set is the `witness_channels` list in
[configs/config_LSTM.yaml](configs/config_LSTM.yaml) (keep the model's
`input_size` equal to its length). Train with:
```
python cli.py fit -c configs/config_LSTM.yaml
```

### Linear baseline and coherence analysis
Two standalone analyses accompanying the paper [*Microseismic Noise Mitigation with Machine Learning for Advanced LIGO*](https://arxiv.org/abs/2511.19682) live in their own subfolders:
- [linear/](linear) — the linear (Wiener / FIR least-squares) noise-subtraction baseline that the ML models are compared against. By default it uses the same GND+CPS X-direction witnesses as the LSTM. Run with `python -m linear.run_linear`.
- [coherence/](coherence) — the coherence analysis between witness and target channels, including the time-dependent cross-coupling in the microseismic band. Run with `python -m coherence.run_coherence`.

See each folder's `README.md` for details.

## Where do we stand?
The code should be ready and easy to run many (parallel) trainings. We plan to add to the code further, e.g., incorporating data preparation and postprocessing directly into the pipeline. Postprocessing is currently done manually in [eval.ipynb](eval.ipynb). Furthermore, we are working on making the models causal, so keep updated! 
