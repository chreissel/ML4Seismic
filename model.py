import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L


class LSTMForecast(L.LightningModule):
    """
    LSTM model for causal forecasting of GS13.X from multi-channel input.
    
    Input: All 10 channels (GND xyz, CPS xyz, CPSR xyz, GS13X)
    Output: GS13.X value 1 second (4 timesteps) into the future
    """
    
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        output_size: int = 1,
        learning_rate: float = 1e-3,
    ):
        """
        Args:
            input_size: Number of input features (10 channels)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate between LSTM layers
            output_size: Number of outputs (1 for GS13.X)
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output layer: maps LSTM hidden state to prediction
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
               e.g., (32, 240, 10) for 60 seconds at 4Hz with 10 channels
        
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM forward pass
        out, _ = self.lstm(x)
        
        # Take the last timestep's output
        return self.fc(out[:, -1, :])
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
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