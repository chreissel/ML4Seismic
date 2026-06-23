from models.s4d import S4D
from models.networks import S4DModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

class LSTMForecast(L.LightningModule):
    """LSTM regressor predicting the target (GS13) from witness channels.

    Originally an *autoregressive* forecaster (it predicted the GS13 channel from
    its own past). It is now a witness-only regressor: it ingests the witness
    channels over an input window and predicts the target over the most recent
    ``predict_horizon`` samples of that same window (causal now-casting), so it
    performs noise subtraction rather than self-forecasting.

    Input  x : (B, seq_length, input_size)   -- witness channels
    Output y : (B, predict_horizon, output_size)  -- target prediction
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 128,
        predict_horizon: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        output_size: int = 1,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=self.hparams.input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=dropout if self.hparams.num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        out, _ = self.lstm(x)
        # keep only the last `predict_horizon` time steps (causal now-cast)
        y_hat = self.fc(out[:, -self.hparams.predict_horizon:, :])
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        with torch.no_grad():
            self.log("val/mae", torch.mean(torch.abs(y_hat - y)), on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("test/loss", loss, on_epoch=True)
        with torch.no_grad():
            self.log("test/mae", torch.mean(torch.abs(y_hat - y)), on_epoch=True)
            self.log("test/rmse", torch.sqrt(loss), on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Lightning CLI may override this via the YAML optimizer/lr_scheduler keys.
        return optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                           weight_decay=1e-4)


class LitModel(L.LightningModule):
    def __init__(self, d_input, d_output, encoder: nn.Module, loss='MSELoss'):
        super().__init__()

        self.encoder = encoder
        self.loss = loss
        self.d_output = d_output
        if self.loss=='MSELoss':
            self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def __loss__(self, X, y):
        y_preds = self.forward(X)
        if self.loss=='MSELoss':
            return self.criterion(y, y_preds)

    def forward(self, x):
        x = self.encoder(x)  
        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return optimizer, scheduler

    def training_step(self, batch, batch_idx, log=True):
        X, y = batch
        loss = self.__loss__(X, y)

        if log:
            self.log("train/loss",
                    loss,
                    on_step=False,
                    on_epoch=True,
                    reduce_fx='mean',
                    logger=True,
                    prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, log=True):
        X, y = batch
        loss = self.__loss__(X, y)

        if log:
            self.log("val/loss",
                    loss,
                    on_step=False,
                    on_epoch=True,
                    reduce_fx='mean',
                    logger=True,
                    prog_bar=True)

        return loss
