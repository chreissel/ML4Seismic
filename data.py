import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from gwpy.timeseries import TimeSeries
import scipy.io


def get_valid_input(prompt, input_type=str, default=None, validator=None):
    """
    Get validated input from user.
    
    Args:
        prompt: Prompt message
        input_type: Type to convert input to (str, int, float, bool)
        default: Default value if user presses Enter
        validator: Optional function to validate input
    
    Returns:
        Validated input value
    """
    while True:
        if default is not None:
            user_input = input(f"{prompt} [{default}]: ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt}: ").strip()
            if not user_input:
                print("❌ This field is required. Please enter a value.")
                continue
        
        # Handle boolean inputs
        if input_type == bool:
            if user_input.lower() in ['y', 'yes', 'true', '1']:
                return True
            elif user_input.lower() in ['n', 'no', 'false', '0']:
                return False
            else:
                print("❌ Please enter 'y' or 'n'")
                continue
        
        # Try to convert to requested type
        try:
            value = input_type(user_input)
        except ValueError:
            print(f"❌ Invalid input. Expected {input_type.__name__}")
            continue
        
        # Apply custom validator if provided
        if validator and not validator(value):
            continue
        
        return value


class GS13PreProcessor:
    """
    Handle preprocessing of LIGO GS13 data.
    
    Takes raw .mat file and produces train/val/test splits with:
    - Resampling to 4Hz
    - Bandpass filtering (0.1-0.3 Hz)
    - Train/val/test split (60/20/20)
    - Optional normalization
    """
    
    def __init__(
        self, 
        mat_file: str,
        time: int,
        sample_rate: int = 128,
        target_rate: float = 4.0,
        bandpass_low: float = 0.1,
        bandpass_high: float = 0.3,
        skip_seconds: int = 100,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        output_dir: str = "data",
        ifo: str = "L1",
        sts2: str = "ITMY",
        chamber: str = "HAM5"
    ):
        """
        Args:
            mat_file: Path to .mat file containing data_matrix
            time: GPS time identifier
            sample_rate: Original sampling rate in Hz (default: 128)
            target_rate: Target resampling rate in Hz (default: 4.0)
            bandpass_low: Lower bandpass frequency in Hz (default: 0.1)
            bandpass_high: Upper bandpass frequency in Hz (default: 0.3)
            skip_seconds: Seconds to skip at start for stability (default: 100)
            train_frac: Fraction for training set (default: 0.6)
            val_frac: Fraction for validation set (default: 0.2)
            output_dir: Directory to save processed data (default: "data")
            ifo: Interferometer prefix (e.g., "L1")
            sts2: STS2 component (e.g., "ITMY")
            chamber: Chamber name (e.g., "HAM5")
        """
        self.mat_file = mat_file
        self.time = time
        self.sample_rate = sample_rate
        self.target_rate = target_rate
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.skip_seconds = skip_seconds
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = 1.0 - train_frac - val_frac
        self.output_dir = output_dir
        
        # Define all 15 channels.
        self.channels = [
            # GND Channels (Indices 0-2)
            f"{ifo}:ISI-GND_STS_{sts2}_X_DQ", 
            f"{ifo}:ISI-GND_STS_{sts2}_Y_DQ", 
            f"{ifo}:ISI-GND_STS_{sts2}_Z_DQ",
            # GS13 Channels (Indices 3-8)
            f"{ifo}:ISI-{chamber}_BLND_GS13X_IN1_DQ", # 3
            f"{ifo}:ISI-{chamber}_BLND_GS13Y_IN1_DQ", # 4
            f"{ifo}:ISI-{chamber}_BLND_GS13Z_IN1_DQ", # 5
            f"{ifo}:ISI-{chamber}_BLND_GS13RX_IN1_DQ", # 6
            f"{ifo}:ISI-{chamber}_BLND_GS13RY_IN1_DQ", # 7
            f"{ifo}:ISI-{chamber}_BLND_GS13RZ_IN1_DQ", # 8
            # CPS Channels (Indices 9-14)
            f"{ifo}:ISI-{chamber}_SCSUM_CPS_X_IN_DQ", 
            f"{ifo}:ISI-{chamber}_SCSUM_CPS_Y_IN_DQ", 
            f"{ifo}:ISI-{chamber}_SCSUM_CPS_Z_IN_DQ",
            f"{ifo}:ISI-{chamber}_BLND_CPSRX_IN1_DQ", 
            f"{ifo}:ISI-{chamber}_BLND_CPSRY_IN1_DQ", 
            f"{ifo}:ISI-{chamber}_BLND_CPSRZ_IN1_DQ"
        ]
        self.gs13_indices = {
            'GS13X': 3, 'GS13Y': 4, 'GS13Z': 5,
            'GS13RX': 6, 'GS13RY': 7, 'GS13RZ': 8
        }
        
    def load_mat_file(self):
        """
        Load data from .mat file.
        
        Returns:
            dict: Dictionary mapping channel names to raw data arrays
        """
        print(f"Loading data from {self.mat_file}...")
        file = scipy.io.loadmat(self.mat_file)
        
        # Check if 'data_matrix' exists, else try to find a matrix
        if 'data_matrix' in file:
            data_matrix = file['data_matrix']
        else:
            # Fallback: find the first non-metadata key with the right shape
            valid_keys = [k for k, v in file.items() if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim == 2]
            if not valid_keys:
                raise ValueError("Could not find 'data_matrix' or any suitable data array in .mat file")
            key = valid_keys[0]
            print(f"Warning: 'data_matrix' not found. Using array '{key}' instead.")
            data_matrix = file[key]

        if data_matrix.shape[1] < len(self.channels):
             raise ValueError(
                f"Data matrix has only {data_matrix.shape[1]} columns, "
                f"but {len(self.channels)} channels were defined."
            )
        
        # Create dictionary mapping channels to data
        channel_data = {}
        for idx, channel in enumerate(self.channels):
            channel_data[channel] = data_matrix[:, idx]
            
        print(f"Loaded {len(self.channels)} channels with {data_matrix.shape[0]} samples each")
        return channel_data
    
    def preprocess_channel(self, data: np.ndarray, channel_name: str):
        """
        Preprocess a single channel.
        
        Steps:
        1. Skip initial seconds for stability
        2. Convert to TimeSeries
        3. Resample to target rate
        4. Apply bandpass filter
        
        Args:
            data: Raw data array
            channel_name: Name of channel (for logging)
            
        Returns:
            np.ndarray: Preprocessed data array
        """
        # Skip initial seconds for stability
        start_idx = self.skip_seconds * self.sample_rate
        if start_idx >= len(data):
            raise ValueError(f"skip_seconds ({self.skip_seconds}) is too large for data length ({len(data) / self.sample_rate:.2f}s)")
        
        data_stable = data[start_idx:]
        
        # Convert to GWpy TimeSeries
        ts = TimeSeries(
            data_stable, 
            dt=1.0/self.sample_rate, 
            t0=self.time + self.skip_seconds
        )
        
        # Resample to target rate
        ts_resampled = ts.resample(self.target_rate)
        
        # Apply bandpass filter
        ts_filtered = ts_resampled.bandpass(self.bandpass_low, self.bandpass_high)
        
        return np.array(ts_filtered.value)
    
    def preprocess_all(self, normalize: bool = False):
        """
        Preprocess all channels and split into train/val/test.
        
        Args:
            normalize: Whether to normalize data using training set statistics
            
        Returns:
            dict: Dictionary with keys 'train', 'val', 'test', each containing
                  numpy array of shape (n_channels, n_samples)
        """
        # Load raw data
        raw_data = self.load_mat_file()
        
        # Preprocess each channel
        print("\nPreprocessing channels...")
        processed_data = {}
        min_len = float('inf')
        for i, channel in enumerate(self.channels, 1):
            processed_data[channel] = self.preprocess_channel(raw_data[channel], channel)
            min_len = min(min_len, len(processed_data[channel]))
            print(f"  [{i:2d}/{len(self.channels)}] {channel}: {len(processed_data[channel])} samples")
        
        print(f"\nAll channels will be truncated to shortest length: {min_len} samples")
        
        # Stack all channels: shape (n_channels, n_samples)
        all_channels_array = np.stack(
            [processed_data[ch][:min_len] for ch in self.channels], 
            axis=0
        )
        print(f"\nStacked array shape: {all_channels_array.shape}")
        
        # Split into train/val/test
        n_samples = all_channels_array.shape[1]
        train_end = int(self.train_frac * n_samples)
        val_end = int((self.train_frac + self.val_frac) * n_samples)
        
        train_data = all_channels_array[:, :train_end]
        val_data = all_channels_array[:, train_end:val_end]
        test_data = all_channels_array[:, val_end:]
        
        print(f"\nSplit sizes:")
        print(f"  Train: {train_data.shape} ({self.train_frac:.1%})")
        print(f"  Val:   {val_data.shape} ({self.val_frac:.1%})")
        print(f"  Test:  {test_data.shape} ({self.test_frac:.1%})")
        
        # Normalize if requested 
        if normalize:
            print("\nNormalizing data using training statistics...")
            mean = train_data.mean(axis=1, keepdims=True)
            std = train_data.std(axis=1, keepdims=True) + 1e-8
            
            train_data = (train_data - mean) / std
            val_data = (val_data - mean) / std
            test_data = (test_data - mean) / std
            
            # Save normalization statistics
            os.makedirs(self.output_dir, exist_ok=True)
            np.savez(
                f'{self.output_dir}/norm_stats_{self.time}.npz',
                mean=mean,
                std=std,
                channels=self.channels
            )
            print(f"✓ Saved normalization statistics to {self.output_dir}/norm_stats_{self.time}.npz")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def save_processed_data(self, normalize: bool = False):
        """
        Preprocess and save data to .npy files.
        
        Args:
            normalize: Whether to normalize the data
        """
        data = self.preprocess_all(normalize=normalize)
        
        os.makedirs(self.output_dir, exist_ok=True)
        np.save(f'{self.output_dir}/train_{self.time}.npy', data['train'])
        np.save(f'{self.output_dir}/val_{self.time}.npy', data['val'])
        np.save(f'{self.output_dir}/test_{self.time}.npy', data['test'])
        
        print(f"\n{'='*70}")
        print("✓ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"Saved preprocessed data:")
        print(f"  • {self.output_dir}/train_{self.time}.npy")
        print(f"  • {self.output_dir}/val_{self.time}.npy")
        print(f"  • {self.output_dir}/test_{self.time}.npy")
        if normalize:
            print(f"  • {self.output_dir}/norm_stats_{self.time}.npz")
        print(f"{'='*70}")
        print("\nChannel indices for reference:")
        for name, idx in self.gs13_indices.items():
            print(f"  • {name}: {idx}")
        print(f"{'='*70}")


class GS13ForecastDataset(Dataset):
    """
    Dataset for autoregressive forecasting.
    
    Input: All 15 channels (3 GND + 6 GS13 + 6 CPS)
    Target: All 6 GS13 channels, predict_horizon steps ahead
    """
    
    def __init__(
        self,
        split: str,
        time: int,
        seq_length: int = 240,
        predict_horizon: int = 4,
        normalize: bool = False,
        data_dir: str = "data"
    ):
        """
        Args:
            split: One of 'train', 'val', or 'test'
            time: GPS time identifier
            seq_length: Length of input sequence (default 240 = 60s at 4Hz)
            predict_horizon: Steps ahead to predict (default 4 = 1s at 4Hz)
            normalize: Whether data is normalized
            data_dir: Directory containing processed data (default: "data")
        """
        super().__init__()
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"
        
        self.split = split
        self.seq_length = seq_length
        self.predict_horizon = predict_horizon
        self.data_dir = data_dir
        
        # Load preprocessed data
        path = f"{data_dir}/{split}_{time}.npy"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find {path}.\n"
                f"Please run preprocessing first:\n"
                f"  python data.py\n"
            )
        
        # Load data: shape (15 channels, T samples)
        self.data = np.load(path)  # (15, T)
        
        num_channels_loaded = self.data.shape[0]
        if num_channels_loaded < 15:
            print(f"Warning: Loaded data has only {num_channels_loaded} channels, expected 15.")
            print("Ensure you have run preprocessing with the updated 15-channel list.")

        # Define channel indices
        # Input: All 15 channels
        self.input_indices = list(range(num_channels_loaded))
        # Target: The 6 GS13 channels (indices 3 through 8)
        self.target_indices = list(range(3, 9))
        
        print(f"Loaded {split} data: {self.data.shape}")
        print(f"  - Input channels: {len(self.input_indices)} (All channels)")
        print(f"  - Target channels: {len(self.target_indices)} (GS13 channels, indices 3-8)")

        
        # Create sequences
        self.X, self.y = self._create_sequences()
        
        print(f"Created {len(self.X)} sequences for {split} set")
        print(f"  Input shape (X): {self.X.shape}")
        print(f"  Target shape (y): {self.y.shape}")
        
    def _create_sequences(self):
        """
        Create input-output pairs for autoregressive forecasting.
        
        Returns:
            tuple: (X, y) where
                X: (n_sequences, seq_length, 15) - input sequences
                y: (n_sequences, 6) - target values
        """
        T = self.data.shape[1]
        
        X_list = []
        y_list = []
        
        # Create sequences where we predict predict_horizon steps ahead
        for i in range(T - self.seq_length - self.predict_horizon + 1):
            # Input: seq_length timesteps of all 15 channels
            x = self.data[self.input_indices, i:i+self.seq_length]  # (15, seq_length)
            
            # Target: 6 GS13 values predict_horizon steps ahead
            y = self.data[self.target_indices, i+self.seq_length+self.predict_horizon-1]  # (6,)
            
            X_list.append(x)
            y_list.append(y)
        
        # Convert to arrays
        X = np.array(X_list, dtype=np.float32)  # (N, 15, seq_length)
        y = np.array(y_list, dtype=np.float32)  # (N, 6)
        
        # Transpose X to (N, seq_length, 15) for easier processing
        X = np.transpose(X, (0, 2, 1))  # (N, seq_length, 15)
        
        # Reshape y to (N, 6)
        y = y.reshape(-1, len(self.target_indices))
        
        # Convert to tensors
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        
        return X, y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GS13DataModule(L.LightningDataModule):
    """
    Lightning DataModule for GS13 autoregressive forecasting.
    
    Handles:
    - Loading train/val/test datasets
    - Creating dataloaders
    - Managing batch size and workers
    """
    
    def __init__(
        self,
        time: int,
        seq_length: int = 240,
        predict_horizon: int = 4,
        normalize: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        data_dir: str = "data"
    ):
        """
        Args:
            time: GPS time identifier
            seq_length: Input sequence length (default 240 = 60s at 4Hz)
            predict_horizon: Steps ahead to predict (default 4 = 1s at 4Hz)
            normalize: Whether data is normalized
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            pin_memory: Whether to pin memory (set True for GPU)
            data_dir: Directory containing processed data (default: "data")
        """
        super().__init__()
        self.time = time
        self.seq_length = seq_length
        self.predict_horizon = predict_horizon
        self.normalize = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_dir = data_dir
        
        self.save_hyperparameters()
        
    def setup(self, stage=None):
        """Setup datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = GS13ForecastDataset(
                "train", 
                self.time, 
                self.seq_length,
                self.predict_horizon,
                self.normalize,
                self.data_dir
            )
            self.val_dataset = GS13ForecastDataset(
                "val",
                self.time,
                self.seq_length,
                self.predict_horizon,
                self.normalize,
                self.data_dir
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = GS13ForecastDataset(
                "test",
                self.time,
                self.seq_length,
                self.predict_horizon,
                self.normalize,
                self.data_dir
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )


def run_preprocessing():
    """
    Interactive preprocessing function that asks user for all parameters.
    Run this with: python data.py
    """
    print("=" * 70)
    print(" " * 15 + "GS13 Data Preprocessing - Interactive Mode")
    print("=" * 70)
    print()
    print("This will guide you through preprocessing LIGO GS13 data.")
    print("Press Ctrl+C at any time to cancel.")
    print()
    
    # Validation functions
    def validate_file(filepath):
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return False
        return True
    
    def validate_positive(value):
        if value <= 0:
            print(f"❌ Value must be positive, got {value}")
            return False
        return True
    
    def validate_fraction(value):
        if not (0 < value < 1):
            print(f"❌ Fraction must be between 0 and 1, got {value}")
            return False
        return True
    
    # ========== INPUT FILE ==========
    print("📁 INPUT FILE")
    print("-" * 70)
    mat_file = get_valid_input(
        "Path to .mat file",
        str,
        default="MLdata_L1HAM5_1381528818_4000_matrix_v2.mat",
        validator=validate_file
    )
    
    # Try to extract time from filename
    default_time = None
    try:
        import re
        match = re.search(r'(\d{10})', mat_file)
        if match:
            default_time = int(match.group(1))
    except:
        pass
    
    time = get_valid_input(
        "GPS time identifier",
        int,
        default=default_time or 1381528818,
        validator=validate_positive
    )
    
    # Get IFO/Chamber details
    default_ifo = "L1"
    default_sts2 = "ITMY"
    default_chamber = "HAM5"
    try:
        if "L1" in mat_file: default_ifo = "L1"
        if "H1" in mat_file: default_ifo = "H1"
        if "HAM5" in mat_file: default_chamber = "HAM5"
        if "HAM6" in mat_file: default_chamber = "HAM6"
    except:
        pass

    ifo = get_valid_input("IFO (e.g., L1)", str, default=default_ifo)
    sts2 = get_valid_input("STS2 (e.g., ITMY)", str, default=default_sts2)
    chamber = get_valid_input("Chamber (e.g., HAM5)", str, default=default_chamber)
    print()
    
    # ========== PREPROCESSING PARAMETERS ==========
    print("⚙️  PREPROCESSING PARAMETERS")
    print("-" * 70)
    
    sample_rate = get_valid_input(
        "Original sampling rate (Hz)",
        int,
        default=128,
        validator=validate_positive
    )
    
    target_rate = get_valid_input(
        "Target resampling rate (Hz)",
        float,
        default=4.0,
        validator=validate_positive
    )
    
    bandpass_low = get_valid_input(
        "Bandpass filter - lower frequency (Hz)",
        float,
        default=0.1,
        validator=validate_positive
    )
    
    def validate_bandpass_high(x):
        if x <= bandpass_low:
            print(f"❌ Upper frequency must be > lower frequency ({bandpass_low} Hz)")
            return False
        return validate_positive(x)
    
    bandpass_high = get_valid_input(
        "Bandpass filter - upper frequency (Hz)",
        float,
        default=0.3,
        validator=validate_bandpass_high
    )
    
    def validate_skip(x):
        if x < 0:
            print("❌ Skip seconds must be non-negative")
            return False
        return True
    
    skip_seconds = get_valid_input(
        "Seconds to skip at start (for stability)",
        int,
        default=100,
        validator=validate_skip
    )
    print()
    
    # ========== DATA SPLIT ==========
    print("✂️  TRAIN/VAL/TEST SPLIT")
    print("-" * 70)
    
    train_frac = get_valid_input(
        "Training set fraction (e.g., 0.6 for 60%)",
        float,
        default=0.6,
        validator=validate_fraction
    )
    
    def validate_val_frac(x):
        if not validate_fraction(x):
            return False
        if train_frac + x >= 1.0:
            print(f"❌ Train + Val must be < 1.0, got {train_frac + x}")
            return False
        return True
    
    val_frac = get_valid_input(
        "Validation set fraction (e.g., 0.2 for 20%)",
        float,
        default=0.2,
        validator=validate_val_frac
    )
    
    test_frac = 1.0 - train_frac - val_frac
    print(f"   → Test set fraction: {test_frac:.1%}")
    print()
    
    # ========== NORMALIZATION ==========
    print("📊 NORMALIZATION")
    print("-" * 70)
    normalize = get_valid_input(
        "Normalize data using training statistics? (y/n)",
        bool,
        default=True
    )
    print()
    
    # ========== OUTPUT ==========
    print("💾 OUTPUT")
    print("-" * 70)
    output_dir = get_valid_input(
        "Output directory for processed data",
        str,
        default="data"
    )
    print()
    
    # ========== SUMMARY ==========
    print("=" * 70)
    print("📋 CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Input:")
    print(f"  • MAT file: {mat_file}")
    print(f"  • GPS time: {time}")
    print(f"  • Channels: 15 (based on {ifo}, {sts2}, {chamber})")
    print()
    print(f"Preprocessing:")
    print(f"  • Original rate: {sample_rate} Hz → Target rate: {target_rate} Hz")
    print(f"  • Bandpass filter: {bandpass_low}-{bandpass_high} Hz")
    print(f"  • Skip initial: {skip_seconds} seconds")
    print()
    print(f"Data split:")
    print(f"  • Train: {train_frac:.1%}")
    print(f"  • Val:   {val_frac:.1%}")
    print(f"  • Test:  {test_frac:.1%}")
    print()
    print(f"Normalization: {'✓ Yes' if normalize else '✗ No'}")
    print()
    print(f"Output directory: {output_dir}/")
    print(f"  • train_{time}.npy")
    print(f"  • val_{time}.npy")
    print(f"  • test_{time}.npy")
    if normalize:
        print(f"  • norm_stats_{time}.npz")
    print("=" * 70)
    print()
    
    # Confirm
    proceed = get_valid_input(
        "Proceed with preprocessing? (y/n)",
        bool,
        default=True
    )
    
    if not proceed:
        print("\n❌ Preprocessing cancelled.")
        return
    
    print()
    print("🚀 Starting preprocessing...")
    print("=" * 70)
    
    # Create preprocessor and run
    try:
        preprocessor = GS13PreProcessor(
            mat_file=mat_file,
            time=time,
            sample_rate=sample_rate,
            target_rate=target_rate,
            bandpass_low=bandpass_low,
            bandpass_high=bandpass_high,
            skip_seconds=skip_seconds,
            train_frac=train_frac,
            val_frac=val_frac,
            output_dir=output_dir,
            ifo=ifo,
            sts2=sts2,
            chamber=chamber
        )
        
        preprocessor.save_processed_data(normalize=normalize)
        
        print("\nYou can now train models using:")
        print("  python cli.py fit --config configs/config_LSTM.yaml")
        print("\nIMPORTANT: Update config_LSTM.yaml and model.py for 15 inputs and 6 outputs!")
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()


# Allow running as script
if __name__ == "__main__":
    run_preprocessing()