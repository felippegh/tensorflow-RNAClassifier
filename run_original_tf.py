"""
Exact replication of original 2017 workflow using TensorFlow 1.x contrib.learn.SVM
This maintains the same logic as svm_main.py but with modern structure.
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config.config import Config
from src.models.tf_contrib_svm import TFContribSVMClassifier


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def load_data_original_format(filename: str, num_elements: int):
    """
    Load data in the exact same way as the original svm_main.py
    
    Args:
        filename: CSV file to load
        num_elements: Number of elements to load
        
    Returns:
        Tuple of (features, targets) in original format
    """
    logger = logging.getLogger(__name__)
    
    num_features = 5460  # Original hardcoded value
    
    # Loading data (exact replica of original)
    features = [list() for i in range(num_features)]
    targets = list()
    
    logger.info(f"Loading {num_elements} samples from {filename}")
    
    with open(filename) as f:
        counter = 0
        for line in f:
            values = line.rstrip().split(',')
            targets.append(int(values[0]))
            for i in range(0, num_features):
                value = float(values[i + 1])
                features[i].append(value)
            counter += 1
            if counter == num_elements:
                break
    
    assert counter == num_elements, "Incorrect number of samples loaded"
    
    # Convert to numpy arrays in the format expected by modern classifier
    X = np.array(features).T  # Transpose to get (samples, features)
    y = np.array(targets)
    
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    return X, y


def run_original_workflow(num_elements: int, filename: str = 'test_final.csv'):
    """
    Run the exact same workflow as the original svm_main.py
    
    Args:
        num_elements: Number of elements to process
        filename: Data file to use
    """
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("RNA CLASSIFIER - ORIGINAL TENSORFLOW 1.x WORKFLOW")
    logger.info("="*60)
    logger.info(f"Processing {num_elements} samples")
    logger.info(f"Using file: {filename}")
    
    # Create results directory
    Config.create_directories()
    
    # Check if data file exists
    if not Path(filename).exists():
        logger.error(f"Data file not found: {filename}")
        logger.info("Please ensure your processed data file is available.")
        logger.info("You can create it by running:")
        logger.info("  python main_tf.py preprocess")
        return
    
    # Load data in original format
    try:
        X, y = load_data_original_format(filename, num_elements)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Initialize classifier with original parameters
    classifier = TFContribSVMClassifier(
        n_features=X.shape[1],
        l2_regularization=0.1,  # Original default
        random_state=42
    )
    
    # Run cross-validation (matching original fold_size = 10)
    logger.info("Starting 10-fold cross-validation...")
    cv_results = classifier.cross_validate(
        X=X,
        y=y,
        cv_folds=10,
        steps_per_fold=1000  # Original num_steps was 10, but that's too low for convergence
    )
    
    # Save results in original format
    timestamp = Path().cwd().name
    metrics_file = Config.RESULTS_DIR / f"metrics_{num_elements}.txt"
    classifier.save_metrics_txt(cv_results, metrics_file)
    
    # Also save as JSON for modern analysis
    json_file = Config.RESULTS_DIR / f"metrics_{num_elements}.json"
    classifier.save_metrics(cv_results, json_file)
    
    # Print results (matching original output format)
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"Dataset size: {num_elements} samples")
    print(f"Features: {X.shape[1]}")
    print(f"Folds: {cv_results['cv_folds']}")
    print(f"Mean accuracy: {cv_results['mean_accuracy']:.6f} (+/- {cv_results['std_accuracy']:.6f})")
    print(f"Mean coding specificity: {cv_results['mean_coding_specificity']:.6f}")
    print(f"Mean coding sensitivity: {cv_results['mean_coding_sensitivity']:.6f}")
    print(f"Mean coding F-measure: {cv_results['mean_coding_f_measure']:.6f}")
    print(f"Mean non-coding specificity: {cv_results['mean_noncoding_specificity']:.6f}")
    print(f"Mean non-coding sensitivity: {cv_results['mean_noncoding_sensitivity']:.6f}")
    print(f"Mean non-coding F-measure: {cv_results['mean_noncoding_f_measure']:.6f}")
    print(f"Total execution time: {cv_results['total_time']:.2f} seconds")
    print("="*60)
    print(f"Results saved to: {metrics_file}")
    print(f"JSON results saved to: {json_file}")
    print("="*60)


def main():
    """Main entry point matching original command-line interface."""
    if len(sys.argv) < 2:
        print('Usage: python run_original_tf.py <num_elements> [filename]')
        print('Example: python run_original_tf.py 1000')
        print('         python run_original_tf.py 1000 test_final.csv')
        sys.exit(1)
    
    num_elements = int(sys.argv[1])
    filename = sys.argv[2] if len(sys.argv) > 2 else 'test_final.csv'
    
    run_original_workflow(num_elements, filename)


if __name__ == "__main__":
    main()