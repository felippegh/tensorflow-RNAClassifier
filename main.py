"""Main pipeline for RNA sequence classification using TensorFlow 1.x."""

import argparse
import logging
import sys
from pathlib import Path
import json
from datetime import datetime
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.config.config import Config
from src.preprocessing.feature_extractor import KmerFeatureExtractor, SequenceProcessor
from src.models.tf_svm_classifier import TensorFlowSVMClassifier


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Config.LOG_DIR / f"rna_classifier_tf_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def preprocess_data(
    coding_file: Path,
    noncoding_file: Path,
    output_file: Path,
    max_sequences: int = None,
    force_reprocess: bool = False
):
    """
    Preprocess FASTA files and extract features.
    
    Args:
        coding_file: Path to coding sequences
        noncoding_file: Path to non-coding sequences
        output_file: Path to output CSV
        max_sequences: Maximum sequences per class
        force_reprocess: Force reprocessing even if output exists
    """
    logger = logging.getLogger(__name__)
    
    # Check if output already exists
    if output_file.exists() and not force_reprocess:
        logger.info(f"Processed file already exists: {output_file}")
        logger.info("Use --force-preprocess to regenerate")
        return
    
    logger.info("Starting data preprocessing...")
    
    # Initialize processor
    processor = SequenceProcessor()
    
    # Process sequences
    processor.process_fasta_to_features(
        coding_file=coding_file,
        noncoding_file=noncoding_file,
        output_file=output_file,
        max_sequences=max_sequences,
        batch_size=Config.BATCH_SIZE
    )
    
    logger.info("Data preprocessing completed")


def train_model(
    data_file: Path,
    model_output: Path,
    cv_folds: int = 10,
    learning_rate: float = 0.01,
    l2_reg: float = 0.1,
    batch_size: int = 100,
    n_epochs: int = 100
):
    """
    Train TensorFlow SVM classifier with cross-validation.
    
    Args:
        data_file: Path to processed features CSV
        model_output: Path to save trained model
        cv_folds: Number of cross-validation folds
        learning_rate: Learning rate for optimizer
        l2_reg: L2 regularization parameter
        batch_size: Batch size for training
        n_epochs: Number of training epochs
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Loading processed data...")
    
    # Load data
    processor = SequenceProcessor()
    X, y = processor.load_features_from_csv(data_file)
    
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Class distribution: Non-coding: {sum(y==0)}, Coding: {sum(y==1)}")
    
    # Initialize classifier
    classifier = TensorFlowSVMClassifier(
        n_features=X.shape[1],
        learning_rate=learning_rate,
        l2_regularization=l2_reg,
        batch_size=batch_size,
        n_epochs=n_epochs,
        random_state=Config.RANDOM_STATE
    )
    
    # Perform cross-validation
    logger.info(f"Starting {cv_folds}-fold cross-validation...")
    cv_results = classifier.cross_validate(X, y, cv_folds=cv_folds)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics
    metrics_file = Config.RESULTS_DIR / f"tf_cv_metrics_{timestamp}.json"
    classifier.save_metrics(cv_results, metrics_file)
    
    # Train final model on full dataset
    logger.info("Training final model on full dataset...")
    train_metrics = classifier.train(X, y, validation_split=0.2)
    
    # Save model
    model_file = model_output or Config.MODEL_DIR / f"tf_svm_model_{timestamp}"
    classifier.save_model(model_file)
    
    # Print summary
    print("\n" + "="*60)
    print("TENSORFLOW 1.x SVM CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"Accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
    print("\nPer-Class Metrics:")
    print(f"  Coding - Precision: {cv_results['mean_coding_precision']:.4f}")
    print(f"  Coding - Recall: {cv_results['mean_coding_recall']:.4f}")
    print(f"  Coding - F1: {cv_results['mean_coding_f1']:.4f}")
    print(f"  Non-coding - Precision: {cv_results['mean_noncoding_precision']:.4f}")
    print(f"  Non-coding - Recall: {cv_results['mean_noncoding_recall']:.4f}")
    print(f"  Non-coding - F1: {cv_results['mean_noncoding_f1']:.4f}")
    print("="*60)
    print(f"\nResults saved to: {metrics_file}")
    print(f"Model saved to: {model_file}")
    
    # Clean up TensorFlow session
    classifier.close()
    
    return cv_results


def main():
    """Main entry point for the TensorFlow RNA classifier pipeline."""
    parser = argparse.ArgumentParser(description="RNA Sequence Classification Pipeline (TensorFlow 1.x)")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess FASTA files')
    preprocess_parser.add_argument(
        '--coding',
        type=Path,
        default=Config.RAW_DATA_DIR / Config.CODING_SEQUENCES_FILE,
        help='Path to coding sequences FASTA file'
    )
    preprocess_parser.add_argument(
        '--noncoding',
        type=Path,
        default=Config.RAW_DATA_DIR / Config.NONCODING_SEQUENCES_FILE,
        help='Path to non-coding sequences FASTA file'
    )
    preprocess_parser.add_argument(
        '--output',
        type=Path,
        default=Config.PROCESSED_DATA_DIR / Config.PROCESSED_CSV_FILE,
        help='Output CSV file path'
    )
    preprocess_parser.add_argument(
        '--max-sequences',
        type=int,
        default=None,
        help='Maximum sequences per class'
    )
    preprocess_parser.add_argument(
        '--force-preprocess',
        action='store_true',
        help='Force reprocessing even if output exists'
    )
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train TensorFlow SVM classifier')
    train_parser.add_argument(
        '--data',
        type=Path,
        default=Config.PROCESSED_DATA_DIR / Config.PROCESSED_CSV_FILE,
        help='Path to processed features CSV'
    )
    train_parser.add_argument(
        '--model-output',
        type=Path,
        default=None,
        help='Path to save trained model'
    )
    train_parser.add_argument(
        '--cv-folds',
        type=int,
        default=Config.CROSS_VALIDATION_FOLDS,
        help='Number of cross-validation folds'
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Learning rate for optimizer'
    )
    train_parser.add_argument(
        '--l2-reg',
        type=float,
        default=0.1,
        help='L2 regularization parameter'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for training'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument(
        '--max-sequences',
        type=int,
        default=None,
        help='Maximum sequences per class'
    )
    pipeline_parser.add_argument(
        '--cv-folds',
        type=int,
        default=Config.CROSS_VALIDATION_FOLDS,
        help='Number of cross-validation folds'
    )
    pipeline_parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Learning rate for optimizer'
    )
    pipeline_parser.add_argument(
        '--l2-reg',
        type=float,
        default=0.1,
        help='L2 regularization parameter'
    )
    pipeline_parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for training'
    )
    pipeline_parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    # Common arguments
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Create directories
    Config.create_directories()
    
    # Execute command
    if args.command == 'preprocess':
        preprocess_data(
            coding_file=args.coding,
            noncoding_file=args.noncoding,
            output_file=args.output,
            max_sequences=args.max_sequences,
            force_reprocess=args.force_preprocess
        )
    
    elif args.command == 'train':
        train_model(
            data_file=args.data,
            model_output=args.model_output,
            cv_folds=args.cv_folds,
            learning_rate=args.learning_rate,
            l2_reg=args.l2_reg,
            batch_size=args.batch_size,
            n_epochs=args.epochs
        )
    
    elif args.command == 'pipeline':
        # Run full pipeline
        logger.info("Running full TensorFlow pipeline...")
        
        # Preprocess
        coding_file = Config.RAW_DATA_DIR / Config.CODING_SEQUENCES_FILE
        noncoding_file = Config.RAW_DATA_DIR / Config.NONCODING_SEQUENCES_FILE
        processed_file = Config.PROCESSED_DATA_DIR / Config.PROCESSED_CSV_FILE
        
        preprocess_data(
            coding_file=coding_file,
            noncoding_file=noncoding_file,
            output_file=processed_file,
            max_sequences=args.max_sequences,
            force_reprocess=False
        )
        
        # Train
        train_model(
            data_file=processed_file,
            model_output=None,
            cv_folds=args.cv_folds,
            learning_rate=args.learning_rate,
            l2_reg=args.l2_reg,
            batch_size=args.batch_size,
            n_epochs=args.epochs
        )
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()