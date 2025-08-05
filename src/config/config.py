"""Configuration settings for RNA Classifier project."""

from pathlib import Path
from typing import Dict, Any
import json
import os


class Config:
    """Central configuration management for RNA Classifier."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = PROJECT_ROOT / "models"
    LOG_DIR = PROJECT_ROOT / "logs"
    
    # Data files
    CODING_SEQUENCES_FILE = "sequences_translated.fa"
    NONCODING_SEQUENCES_FILE = "Noncoding.fa"
    PROCESSED_CSV_FILE = "processed_sequences.csv"
    
    # Feature extraction parameters
    KMER_SIZES = [1, 2, 3, 4, 5]  # K-mer sizes to extract
    MAX_KMER_SIZE = 5
    NUCLEOTIDES = ['A', 'C', 'G', 'T']
    TOTAL_FEATURES = sum(4**k for k in KMER_SIZES)  # 1364 features for k=1..5
    
    # Model parameters
    CROSS_VALIDATION_FOLDS = 10
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # SVM specific parameters
    SVM_KERNEL = 'linear'
    SVM_C = 1.0  # Regularization parameter
    SVM_MAX_ITER = 1000
    
    # Processing parameters
    BATCH_SIZE = 1000  # Process sequences in batches
    MAX_SEQUENCE_LENGTH = 10000  # Maximum sequence length to process
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Output files
    METRICS_FILE = "metrics.json"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return {
            "project_root": str(cls.PROJECT_ROOT),
            "data_dir": str(cls.DATA_DIR),
            "kmer_sizes": cls.KMER_SIZES,
            "total_features": cls.TOTAL_FEATURES,
            "cv_folds": cls.CROSS_VALIDATION_FOLDS,
            "batch_size": cls.BATCH_SIZE,
        }
    
    @classmethod
    def save_config(cls, filepath: Path = None):
        """Save configuration to JSON file."""
        if filepath is None:
            filepath = cls.PROJECT_ROOT / "config.json"
        
        with open(filepath, 'w') as f:
            json.dump(cls.get_config(), f, indent=2)
    
    @classmethod
    def create_directories(cls):
        """Create necessary project directories."""
        dirs = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODEL_DIR,
            cls.LOG_DIR,
            cls.RESULTS_DIR,
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)