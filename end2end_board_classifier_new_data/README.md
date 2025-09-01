# Chess Board Recognition Training

Train chess board recognition models using the ChessReD dataset from Google Cloud Storage.

## Quick Start

```bash
# 1. Install dependencies
pip install tensorflow google-cloud-storage requests

# 2. Run training
python train.py

# 3. Follow prompts to select model and training mode
```

## System Requirements

- **Python**: 3.8+
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2GB+ free space for model outputs
- **Network**: Stable internet connection (images downloaded during training)

## Installation

### 1. Install Python Dependencies

```bash
pip install tensorflow>=2.13.0 google-cloud-storage requests
```

### 2. Optional: Google Cloud Authentication

For model upload functionality (optional):

```bash
# Install Google Cloud SDK
# Download from: https://cloud.google.com/sdk/docs/install

# Authenticate (one-time setup)
gcloud auth application-default login
```

**Note**: Training works without authentication. Upload feature requires authentication.

## Usage

### Basic Training

```bash
python train.py
```

The script will prompt you for:

1. **Model Architecture**:
   - `l` - Light model (~1M params, fast training, 3.6MB)
   - `r` - ResNeXt-101 (~88M params, better accuracy, 338MB)

2. **Training Mode**:
   - `q` - Quick (testing/debugging)
   - `t` - Thorough (default, balanced)
   - `p` - Production (full training)

### Example Session

```
Select model architecture:
  [l] light      -> lightweight CNN (~2M params, faster training)
  [r] resnext    -> ResNeXt-101 32x8d (~88M params, better accuracy)

Enter l / r (or press Enter for resnext): l

Select training mode:
  [q] quick      -> sanity check / tiny run (fast)
  [t] thorough   -> solid baseline (default)  
  [p] production -> full training (slow)

Enter q / t / p (or press Enter for thorough): t
```

## Training Configurations

### Model Options

| Model | Parameters | Size | Training Time | Accuracy |
|-------|------------|------|---------------|----------|
| Light | ~1M | 3.6MB | Fast | Good |
| ResNeXt-101 | ~88M | 338MB | Slow | Better |

### Training Modes

| Mode | Epochs | Use Case |
|------|--------|----------|
| Quick | 20 | Testing, debugging |
| Thorough | 50-200 | Default training |
| Production | 100-300 | Final models |

## Outputs

Training generates files in organized directories:

### Model Files
```
outputs/
├── light_model/                    # Light model outputs
│   ├── best_model.keras           # Best model (for loading)
│   └── final_light_quick_YYYYMMDD.keras  # Final model (for archiving)
└── best_model/                    # ResNeXt model outputs
    ├── best_model.keras
    └── final_large_thorough_YYYYMMDD.keras
```

### Training Analytics
```
analytics/
├── history_light_quick_YYYYMMDD.json     # Training metrics
├── train_log_quick.csv                   # Epoch-by-epoch log
└── tb_quick/                            # TensorBoard logs
```

### Log Files
```
logs/
└── training.log                         # Detailed training log
```

## Model Upload (Optional)

After training, you can upload models to Google Cloud Storage:

```
Upload to GCS? [y/N]: y
Enter destination bucket name: my-models-bucket
Enter base path in bucket: chess-models/experiments
```

Uploads:
- Final model file
- Best model checkpoint  
- Training history JSON

## Data Source

Uses the ChessReD dataset from public Google Cloud Storage:
- **Bucket**: `gs://chess_red_dataset`
- **Images**: Downloaded on-demand during training
- **Preprocessed data**: Downloaded once and cached locally

The training script automatically:
1. Checks for preprocessed data availability
2. Downloads required metadata files  
3. Streams images during training

## Monitoring Training

### Console Output
- Real-time progress updates
- Model metrics per epoch
- Board-level accuracy statistics

### TensorBoard (Optional)
```bash
tensorboard --logdir=analytics/tb_[mode]/
```

### Log Files
Detailed logs saved to `logs/training.log`

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
pip install tensorflow google-cloud-storage requests
```

**2. Network/Download Issues**
- Check internet connection
- Images download during training (expect some 404s for missing files)
- Training continues with fallback patterns for missing images

**3. Memory Issues**
- Try light model instead of ResNeXt-101
- Use quick mode for testing
- Close other applications

**4. GPU Not Detected**
```
[INFO] No GPUs detected, using CPU
```
This is normal if you don't have a CUDA-compatible GPU. Training works on CPU.

**5. Upload Authentication**
```bash
gcloud auth application-default login
```

### Performance Tips

**For Faster Training**:
- Use light model
- Use quick mode for testing
- GPU acceleration (if available)

**For Better Accuracy**:
- Use ResNeXt-101 model
- Use thorough/production mode
- More epochs

## Model Loading

Load trained models in your code:

```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('outputs/light_model/best_model.keras')

# Make predictions
predictions = model.predict(image_batch)  # Shape: (batch, 64, 13)
```

## File Structure

```
end2end_board_classifier/
├── train.py              # Main training script
├── _helpers.py           # Model architectures and utilities  
├── preprocess.py         # Data preprocessing (optional)
├── README.md            # This file
├── outputs/             # Generated model files
├── analytics/           # Training logs and metrics
├── logs/                # Detailed training logs
└── gcs_cache/          # Downloaded preprocessed data
```

## Advanced Usage

### Custom Configuration

Edit configuration in `_helpers.py`:
- `get_light_model_config()` - Light model settings
- `get_mode_config()` - ResNeXt model settings

### Skip Preprocessing

The training script automatically handles data preparation. Manual preprocessing only needed for custom datasets.

## Support

For issues:
1. Check log files in `logs/training.log`
2. Verify internet connection for data download
3. Ensure sufficient disk space and memory
4. Try light model + quick mode for testing

Training successfully runs on both CPU and GPU systems.