# RNA Sequence Classifier

## Overview
A comprehensive RNA sequence classification system using TensorFlow SVM implementations to distinguish between coding and non-coding RNA sequences through k-mer feature analysis and cross-validation.

## Project Architecture

### Core Components
```
tensorflow-RNAClassifier/
├── src/                         # Modular implementation
│   ├── config/config.py        # Configuration management
│   ├── models/                 # TensorFlow SVM implementations
│   └── preprocessing/          # Feature extraction modules
├── data/                       
│   ├── raw/                    # FASTA input files
│   └── processed/              # Processed feature matrices
├── logs/                       # Execution logs
├── results/                    # Cross-validation results
├── models/                     # Trained model storage
├── main.py                     # Primary analysis pipeline
├── run_pipeline_tf.sh          # Automated execution script
├── run_original_tf.py          # Alternative workflow
├── original_svm_main.py        # Core SVM implementation
├── tf_svm_estimator.py         # TensorFlow SVM class
├── fasta_preprocessor.py       # FASTA processing utilities
├── extract_kmer_features.py    # K-mer feature extraction
└── tf_svm_example.py           # Simple usage example
```

## Usage

### Quick Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete analysis pipeline
./run_pipeline_tf.sh 1000
```

### Detailed Control
```bash
# Preprocessing step
python main.py preprocess --max-sequences 1000

# Training with cross-validation
python main.py train --cv-folds 10

# Complete pipeline with parameters
python main.py pipeline --max-sequences 5000 --learning-rate 0.01
```

### Alternative Implementation
```bash
# Direct SVM execution
python run_original_tf.py 1000

# Multiple dataset analysis
for size in 100 200 400 800 1600; do
    python run_original_tf.py $size
done
```

## Technical Implementation

### Machine Learning Framework
- **TensorFlow 1.x** with contrib.learn SVM implementations
- **K-fold cross-validation** (10-fold default) for robust evaluation
- **Multiple SVM approaches**: contrib.learn wrapper and custom implementation
- **Feature scaling** and regularization options

### Feature Engineering
- **K-mer analysis**: 1-5 nucleotide pattern extraction
- **Sequence normalization** by length
- **1364 total features** from nucleotide combinations
- **Batch processing** for large datasets

### Data Processing
- **FASTA file parsing** and simplification
- **CSV feature matrix generation** 
- **Configurable sequence limits** for testing
- **Memory-efficient batch processing**

## File Descriptions

### Primary Interface
- `main.py` - Command-line interface with subcommands for preprocessing, training, and pipeline execution
- `run_pipeline_tf.sh` - Shell script for streamlined execution with parameter control

### Core Implementation
- `original_svm_main.py` - Main SVM classification with cross-validation
- `tf_svm_estimator.py` - TensorFlow SVM estimator class
- `run_original_tf.py` - Direct workflow execution matching core implementation

### Preprocessing Tools
- `fasta_preprocessor.py` - FASTA file simplification and formatting
- `extract_kmer_features.py` - K-mer feature extraction and CSV generation
- `tf_svm_example.py` - Simple TensorFlow contrib.learn usage example

### Modular Components
- `src/config/config.py` - Centralized configuration and parameter management
- `src/models/tf_contrib_svm.py` - TensorFlow contrib.learn SVM wrapper
- `src/models/tf_svm_classifier.py` - Custom TensorFlow SVM implementation
- `src/preprocessing/feature_extractor.py` - Advanced feature extraction utilities

## Performance Metrics

The system provides comprehensive evaluation metrics:
- **Overall accuracy** with confidence intervals
- **Per-class specificity and sensitivity** for coding/non-coding sequences
- **F-measures** for both sequence types
- **Confusion matrices** for detailed analysis
- **Execution time tracking** for performance monitoring

## Configuration

Key parameters can be adjusted in `src/config/config.py`:
- K-mer sizes for feature extraction
- Cross-validation fold count
- Batch processing sizes
- File paths and naming conventions
- Logging and output formats

## Getting Started

1. **Setup Environment**: Install TensorFlow 1.x and dependencies
2. **Prepare Data**: Place FASTA files in `data/raw/` directory
3. **Execute Pipeline**: Run `./run_pipeline_tf.sh 1000` for 1000 sequences per class
4. **Review Results**: Check `results/` directory for cross-validation metrics

The system is designed for bioinformatics research requiring reliable RNA sequence classification with comprehensive evaluation and reproducible results.