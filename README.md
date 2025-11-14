# Journal Entry Prediction Model

Advanced transformer-based ML model for predicting accounting journal entries from text descriptions.

## Project Structure

```
new-je-model/
├── config/                      # Configuration files
│   ├── model_config.yaml       # Model architecture hyperparameters
│   └── training_config.yaml    # Training hyperparameters
├── data/                        # Data loaders and preprocessing (user-implemented)
│   ├── __init__.py
│   ├── dataset.py              # Dataset classes
│   ├── preprocessing.py        # Data preprocessing utilities
│   └── data_loaders.py         # DataLoader setup
├── models/                      # Model architecture components
│   ├── __init__.py
│   ├── text_encoder.py         # Transformer-based text encoder
│   ├── hierarchical_encoder.py # Account hierarchy encoder
│   ├── cross_attention.py      # Cross-attention fusion layers
│   ├── set_decoder.py          # DETR-style set prediction decoder
│   └── journal_entry_model.py  # Complete model combining all components
├── training/                    # Training utilities
│   ├── __init__.py
│   ├── trainer.py              # Main training loop
│   ├── losses.py               # Hungarian matching and custom losses
│   └── metrics.py              # Evaluation metrics
├── inference/                   # Inference utilities
│   ├── __init__.py
│   ├── predictor.py            # Clean inference interface
│   └── calibration.py          # Temperature scaling and confidence calibration
├── utils/                       # General utilities
│   ├── __init__.py
│   ├── checkpoint.py           # Checkpoint saving/loading
│   ├── logging_utils.py        # Logging configuration
│   └── visualization.py        # Attention visualization, etc.
├── experiments/                 # Experiment tracking and notebooks
│   └── README.md
├── artifacts/                   # Saved models, configs, results
│   ├── checkpoints/            # Model checkpoints
│   ├── best_models/            # Best performing models
│   └── logs/                   # Training logs
├── tests/                       # Unit tests
│   └── __init__.py
├── train.py                     # Main training script
├── infer.py                     # Main inference script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Architecture Overview

Based on research in `thoughts.md`, the model implements:

1. **Text Encoder**: Transformer-based encoder for processing transaction descriptions
2. **Hierarchical Encoder**: Encodes account code structure using entity embeddings and hierarchical relationships
3. **Cross-Attention Fusion**: Bidirectional attention between text and hierarchical features
4. **Set Prediction Decoder**: DETR-style parallel prediction of journal entry lines
5. **Confidence Calibration**: Temperature scaling for calibrated probability estimates

## Database Schema

The model works with journal entry data from three tables:
- `journal_entry`: Main entry metadata (date, description, type, etc.)
- `entry_line`: Individual debit/credit lines within each entry
- `ledger_account`: Chart of accounts with hierarchical structure

See `Database_structure.md` for complete schema.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p artifacts/checkpoints artifacts/best_models artifacts/logs
```

## Training

```bash
python train.py --config config/training_config.yaml
```

## Inference

```bash
python infer.py --checkpoint artifacts/best_models/model_best.pt --input "Payment for office supplies"
```

## Key Features

- **Modular Architecture**: Each component is independently testable and swappable
- **Clean Inference API**: Simple interface for production deployment
- **Checkpoint Management**: Automatic saving of best models and training state
- **Configuration-Driven**: YAML configs for reproducible experiments
- **Memory Tracking**: Important decisions and dimensions tracked in knowledge graph

## Development Notes

- User implements data loaders and preprocessing in `data/` directory
- Model dimensions and architecture decisions stored in memory system
- All artifacts (checkpoints, logs) saved in `artifacts/` directory
- Inference designed for production use with batch processing support

