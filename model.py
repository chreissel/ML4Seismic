import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L


class LSTMForecast(L.LightningModule):
    """
    LSTM model for autoregressive forecasting of GS13 channels.
    
    Input: All 15 channels (3 GND + 6 GS13 + 6 CPS)
    Output: 6 GS13 values (X,Y,Z,RX,RY,RZ) 1 second (4 timesteps) into the future
    """
    
    def __init__(
        self,
        input_size: int = 15,  # Will be set by config
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        output_size: int = 6,  # Will be set by config
        learning_rate: float = 1e-3,
    ):
        """
        Args:
            input_size: Number of input features (15 channels)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate between LSTM layers
            output_size: Number of outputs (6 for GS13 channels)
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        # input_size and output_size will be populated by LightningCLI
        # from your .yaml config file.
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
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
               e.g., (32, 240, 15) for 60 seconds at 4Hz with 15 channels
        
        Returns:
            Output tensor of shape (batch_size, output_size)
            e.g., (32, 6)
        """
        # LSTM forward pass
        # x shape: (batch_size, seq_length, 15)
        out, _ = self.lstm(x)
        
        # out shape: (batch_size, seq_length, hidden_size)
        
        # Take the last timestep's output
        # out[:, -1, :] shape: (batch_size, hidden_size)
        y_hat = self.fc(out[:, -1, :])
        
        # y_hat shape: (batch_size, 6)
        return y_hat
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch  # x: (B, 240, 15), y: (B, 6)
        y_hat = self(x) # y_hat: (B, 6)
        loss = self.loss_fn(y_hat, y) # MSELoss compares (B, 6) and (B, 6) -> scalar
        
        # Log metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Additional metrics
        with torch.no_grad():
            # torch.abs will work element-wise on (B, 6) tensors
            mae = torch.mean(torch.abs(y_hat - y))
            self.log("val/mae", mae, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
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
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Note: Lightning CLI handles optimizer and scheduler configuration
        # This method is here for completeness when not using CLI
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss',
            }
        }


# Keep LitModel for S4D and other encoder-based models
class LitModel(L.LightningModule):
    """
    Generic Lightning module wrapper for encoder-based models.
    Used for S4D, DeepClean, etc.
    """
    
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