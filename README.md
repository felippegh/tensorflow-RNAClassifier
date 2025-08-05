# RNA Sequence Classifier

A machine learning pipeline for classifying RNA sequences as coding or non-coding using TensorFlow 1.x Support Vector Machines (SVM) and k-mer feature extraction.

## Overview

This project implements a binary classification system to distinguish between coding and non-coding RNA sequences. It uses k-mer frequency analysis to extract features from RNA sequences and trains an SVM classifier with cross-validation for robust performance evaluation.

### Key Features

- **TensorFlow 1.x SVM**: Uses `tf.contrib.learn.SVM` and custom TensorFlow implementations
- **K-mer Feature Extraction**: Extracts features based on nucleotide patterns (1-mer to 5-mer, configurable)
- **Cross-Validation**: Built-in k-fold cross-validation for reliable performance metrics
- **Batch Processing**: Handles large datasets efficiently with batch processing
- **Comprehensive Metrics**: Provides detailed per-class metrics including specificity, sensitivity, and F-measures
- **Modular Architecture**: Clean, type-hinted, well-documented code structure
- **Multiple Interfaces**: Command-line interface and shell script automation

## Installation

### Prerequisites

- **Python 3.6 or higher** (tested with Python 3.6-3.8)
- **pip package manager**
- **Git** (for cloning the repository)

### Step-by-Step Setup

1. **Clone the repository:**
```bash
git clone https://github.com/felippe/tensorflow-RNAClassifier.git
cd tensorflow-RNAClassifier
```

2. **Create a virtual environment (strongly recommended):**
```bash
python3 -m venv rna_classifier_env
source rna_classifier_env/bin/activate  # On Windows: rna_classifier_env\Scripts\activate
```

3. **Install TensorFlow 1.x and dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```
You should see: `TensorFlow version: 1.15.5`

5. **Test project setup (optional but recommended):**
```bash
python3 test_project.py
```
This will check your installation and project structure.

## Usage

### Quick Start

**⚠️ Important**: Before running, place your FASTA files in the `data/raw/` directory:
- `sequences_translated.fa` (coding sequences)
- `Noncoding.fa` (non-coding sequences)

**Method 1: Shell Script (Recommended)**
```bash
./run_pipeline_tf.sh 1000  # Process 1000 sequences per class
```

**Method 2: Python Command-Line Interface**
```bash
python3 main.py pipeline --max-sequences 1000
```

**Method 3: Direct Implementation**
```bash
python3 run_original_tf.py 1000
```

**What happens when you run these commands:**
1. **Data Processing**: Converts FASTA files to k-mer feature vectors
2. **Training**: Trains TensorFlow SVM with 10-fold cross-validation
3. **Evaluation**: Calculates accuracy, precision, recall, F-measures
4. **Output**: Saves metrics to `results/` and models to `models/`

### Step-by-Step Usage

#### 1. Prepare Your Data

Place your FASTA files in the correct directory:
```bash
mkdir -p data/raw
# Copy your files:
# - sequences_translated.fa (coding RNA sequences)
# - Noncoding.fa (non-coding RNA sequences)
```

#### 2. Run the Complete Pipeline

**Option A: Automated Shell Script (Easiest)**
```bash
./run_pipeline_tf.sh 1000 10 0.01 0.1 100
# Parameters: max_sequences cv_folds learning_rate l2_reg epochs
```

**Option B: Python Command Line**
```bash
python3 main.py pipeline --max-sequences 1000 --cv-folds 10
```

**Option C: Step by Step**
```bash
# Step 1: Preprocess data
python3 main.py preprocess --max-sequences 1000

# Step 2: Train model
python3 main.py train --cv-folds 10 --learning-rate 0.01
```

#### 3. Advanced Usage

**Custom Parameters:**
```bash
python3 main.py pipeline \
    --max-sequences 5000 \
    --cv-folds 10 \
    --learning-rate 0.005 \
    --l2-reg 0.05 \
    --epochs 200 \
    --log-level DEBUG
```

**Multiple Dataset Sizes:**
```bash
# Test different dataset sizes
for size in 100 200 500 1000 2000; do
    python3 run_original_tf.py $size
done
```

#### 4. Understanding the Output

After running, check these directories:
- `results/` - Cross-validation metrics and performance reports
- `models/` - Trained TensorFlow models 
- `logs/` - Detailed execution logs
- `data/processed/` - Generated feature files

### Command-Line Arguments

#### Global Arguments
- `--log-level`: Set logging verbosity (DEBUG, INFO, WARNING, ERROR)

#### Preprocess Command
- `--coding`: Path to coding sequences FASTA file
- `--noncoding`: Path to non-coding sequences FASTA file  
- `--output`: Output CSV file path
- `--max-sequences`: Limit sequences per class (useful for testing)
- `--force-preprocess`: Regenerate features even if output exists

#### Train Command (TensorFlow 1.x)
- `--data`: Path to preprocessed features CSV
- `--model-output`: Path to save trained model
- `--cv-folds`: Number of cross-validation folds (default: 10)
- `--learning-rate`: Learning rate for optimizer (default: 0.01)
- `--l2-reg`: L2 regularization parameter (default: 0.1)
- `--batch-size`: Batch size for training (default: 100)
- `--epochs`: Number of training epochs (default: 100)

## Project Structure

```
tensorflow-RNAClassifier/
├── src/                           # Modern modular implementation
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py             # Centralized configuration
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── feature_extractor.py  # K-mer feature extraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tf_contrib_svm.py     # TF 1.x contrib.learn SVM
│   │   └── tf_svm_classifier.py  # Custom TF 1.x SVM implementation
│   └── __init__.py
├── data/
│   ├── raw/                      # Original FASTA files
│   └── processed/                # Processed feature files
│
├── main.py                       # Modern TensorFlow 1.x pipeline
├── run_original_tf.py            # Exact 2017 workflow replica
├── run_pipeline_tf.sh            # Shell script runner
│
├── original_svm_main.py          # Original 2017 implementation (reference)
├── tf_svm_estimator.py           # Original TF SVM estimator class
├── fasta_preprocessor.py         # FASTA file preprocessor
├── extract_kmer_features.py      # K-mer feature extraction script
├── tf_svm_example.py             # Simple TF contrib.learn example
│
├── requirements.txt              # TensorFlow 1.x dependencies
├── CLAUDE.md                     # Claude Code configuration
└── README.md                     # This file
```

### File Categories

**Primary Interface:**
- `main.py` - Command-line interface for TF 1.x pipeline with subcommands
- `run_pipeline_tf.sh` - Shell script for automated execution
- `src/` - Modular, type-hinted, well-documented implementation

**Core Implementation:**
- `original_svm_main.py` - Main SVM classification with cross-validation
- `tf_svm_estimator.py` - TensorFlow SVM estimator class
- `run_original_tf.py` - Direct workflow execution

**Processing Tools:**
- `fasta_preprocessor.py` - FASTA file simplification utilities
- `extract_kmer_features.py` - K-mer feature extraction algorithms
- `tf_svm_example.py` - Simple TensorFlow contrib.learn usage example

## Data Format

### Input: FASTA Files

The pipeline expects standard FASTA format files:

```
>sequence_header_1
ACGTACGTACGTACGT...
>sequence_header_2
GCTAGCTAGCTAGCTA...
```

Place your files in `data/raw/`:
- Coding sequences: `sequences_translated.fa`
- Non-coding sequences: `Noncoding.fa`

### Output: Feature Matrix

The preprocessor generates a CSV file with:
- Columns: k-mer features (1364 features for k=1..5) + label
- Rows: One per sequence
- Labels: 1 for coding, 0 for non-coding

## Model Performance

The classifier provides comprehensive metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positive / (True positive + False positive)
- **Recall**: True positive / (True positive + False negative)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Per-class metrics**: Separate metrics for coding and non-coding classes

Results are saved in JSON format in the `results/` directory.

## Configuration

Key parameters can be modified in `src/config/config.py`:

```python
KMER_SIZES = [1, 2, 3, 4, 5]  # K-mer sizes to extract
CROSS_VALIDATION_FOLDS = 10    # Number of CV folds
BATCH_SIZE = 1000              # Batch size for processing
SVM_KERNEL = 'linear'          # Default SVM kernel
SVM_C = 1.0                    # Default regularization
```

## Troubleshooting

### Common Issues

**1. TensorFlow Installation Problems**
```bash
# If you get TensorFlow errors, try:
pip uninstall tensorflow
pip install tensorflow==1.15.5

# For Apple Silicon Macs (M1/M2), you might need:
pip install tensorflow==1.15.5 --no-deps
pip install numpy>=1.16.0,<1.19.0
```

**2. Memory Issues with Large Datasets**
```bash
# Reduce the number of sequences processed
python3 main.py pipeline --max-sequences 500

# Or edit src/config/config.py:
BATCH_SIZE = 500  # Default is 1000
```

**3. FASTA File Not Found**
```bash
# Make sure your files are in the right place:
ls data/raw/
# Should show: sequences_translated.fa  Noncoding.fa

# If files have different names, use:
python3 main.py preprocess --coding data/raw/your_coding_file.fa --noncoding data/raw/your_noncoding_file.fa
```

**4. Slow Training**
```bash
# Reduce cross-validation folds:
python3 main.py train --cv-folds 5

# Reduce training epochs:
python3 main.py train --epochs 50
```

**5. Permission Denied on Shell Script**
```bash
chmod +x run_pipeline_tf.sh
```

### Getting Help

If you encounter issues:
1. Check the `logs/` directory for detailed error messages
2. Try running with `--log-level DEBUG` for more information
3. Ensure you're using Python 3.6-3.8 (TensorFlow 1.15 compatibility)
4. Make sure you're in a virtual environment

### Expected Output

A successful run should produce:
```
========================================
TENSORFLOW 1.x SVM CROSS-VALIDATION RESULTS
========================================
Accuracy: 0.8245 (+/- 0.0123)

Per-Class Metrics:
  Coding - Precision: 0.8156
  Coding - Recall: 0.8401
  Coding - F1: 0.8277
  Non-coding - Precision: 0.8334
  Non-coding - Recall: 0.8089
  Non-coding - F1: 0.8210
========================================
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rna_classifier_2024,
  author = {Felippe},
  title = {RNA Sequence Classifier},
  year = {2024},
  url = {https://github.com/felippe/tensorflow-RNAClassifier}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Original implementation from 2017 final project
- Updated and modernized in 2024
- Uses scikit-learn for machine learning
- K-mer feature extraction inspired by bioinformatics best practices