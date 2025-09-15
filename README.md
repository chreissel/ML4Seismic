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

## Where do we stand?
The code should be ready and easy to run many (parallel) trainings. We plan to add to the code further, e.g., incorporating data preparation and postprocessing directly into the pipeline. Postprocessing is currently done manually in [eval.ipynb](eval.ipynb). Furthermore, we are working on making the models causal, so keep updated! 
