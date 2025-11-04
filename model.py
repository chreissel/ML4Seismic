import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L


class LSTMForecast(L.LightningModule):
    
    def __init__(
        self,
        input_size: int = 15,  
        hidden_size: int = 128,
        predict_horizon: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        output_size: int = 6,  
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.hparams.input_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.num_layers,
            batch_first=True,
            dropout=dropout if self.hparams.num_layers > 1 else 0.0
        )
        
        # Output layer: maps LSTM hidden state to prediction
        self.fc = nn.Linear(self.hparams.hidden_size, self.hparams.output_size)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        y_hat = self.fc(out[:, -self.hparams.predict_horizon: , :])
        return y_hat
    
    def training_step(self, batch, batch_idx):
        x, y = batch  
        y_hat = self(x) 
        loss = self.loss_fn(y_hat, y) 
        
        # Log metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Additional metrics
        with torch.no_grad():
            mae = torch.mean(torch.abs(y_hat - y))
            self.log("val/mae", mae, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Log metrics
        self.log("test/loss", loss, on_epoch=True)
        
        # Additional metrics
        with torch.no_grad():
            mae = torch.mean(torch.abs(y_hat - y))
            rmse = torch.sqrt(loss)
            self.log("test/mae", mae, on_epoch=True)
            self.log("test/rmse", rmse, on_epoch=True)
        return loss

class LitModel(L.LightningModule):
    
    def __init__(self, d_input, d_output, encoder: nn.Module, loss='MSELoss'):
        super().__init__()

        self.encoder = encoder
        self.loss = loss
        self.d_output = d_output
        if self.loss == 'MSELoss':
            self.criterion = nn.MSELoss()
        self.save_hyperparameters(ignore=['encoder'])

    def __loss__(self, X, y):
        y_preds = self.forward(X)
        if self.loss == 'MSELoss':
            return self.criterion(y, y_preds)

    def forward(self, x):
        x = self.encoder(x)  
        return x

    def configure_optimizers(self):
        # Note: When using Lightning CLI, optimizer and scheduler
        # are configured in the YAML file
        optimizer = optim.AdamW(self.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        X, y = batch
        loss = self.__loss__(X, y)

        self.log("train/loss",
                loss,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        loss = self.__loss__(X, y)

        self.log("val/loss",
                loss,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True)

        return loss
