"""TensorFlow 1.x contrib.learn SVM - Modernized version of original implementation."""

import tensorflow as tf
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json

# Ensure TF 1.x behavior
tf.compat.v1.disable_v2_behavior()

logger = logging.getLogger(__name__)


class TFContribSVMClassifier:
    """
    TensorFlow 1.x contrib.learn SVM classifier.
    Modernized version of the original 2017 implementation.
    """
    
    def __init__(
        self,
        n_features: int,
        l2_regularization: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the TF contrib SVM classifier.
        
        Args:
            n_features: Number of input features
            l2_regularization: L2 regularization parameter
            random_state: Random seed for reproducibility
        """
        self.n_features = n_features
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        
        # Create feature columns
        self.feature_columns = []
        for i in range(n_features):
            column = tf.contrib.layers.real_valued_column(
                column_name=f'feature_{i}',
                dimension=1
            )
            self.feature_columns.append(column)
        
        # Initialize the SVM estimator
        self.classifier = tf.contrib.learn.SVM(
            feature_columns=self.feature_columns,
            example_id_column='example_id',
            l2_regularization=l2_regularization,
            config=tf.contrib.learn.RunConfig(
                tf_random_seed=random_state,
                save_summary_steps=100,
                save_checkpoints_steps=1000
            )
        )
        
        self.is_trained = False
        self.training_metrics = {}
        
        logger.info(f"Initialized TFContribSVMClassifier with {n_features} features")
    
    def _prepare_input_fn(self, X: np.ndarray, y: np.ndarray = None, batch_size: int = None, num_epochs: int = None):
        """
        Prepare input function for TensorFlow estimator.
        
        Args:
            X: Feature matrix
            y: Labels (optional for prediction)
            batch_size: Batch size
            num_epochs: Number of epochs
            
        Returns:
            Input function for TensorFlow estimator
        """
        def input_fn():
            # Create feature dictionary
            feature_dict = {}
            
            # Add example IDs
            feature_dict['example_id'] = tf.constant(
                [f'example_{i}' for i in range(len(X))],
                dtype=tf.string
            )
            
            # Add features
            for i in range(self.n_features):
                feature_dict[f'feature_{i}'] = tf.constant(
                    X[:, i].astype(np.float32),
                    dtype=tf.float32
                )
            
            if y is not None:
                # For training
                labels = tf.constant(y.astype(np.int32), dtype=tf.int32)
                return feature_dict, labels
            else:
                # For prediction
                return feature_dict
        
        return input_fn
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        steps: int = 1000
    ) -> Dict[str, float]:
        """
        Train the SVM classifier.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            y: Labels (n_samples,) with values 0 or 1
            validation_split: Fraction of data to use for validation
            steps: Number of training steps
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training TF-contrib-SVM on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Split data for validation
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Prepare input functions
        train_input_fn = self._prepare_input_fn(X_train, y_train)
        val_input_fn = self._prepare_input_fn(X_val, y_val)
        
        # Train the model
        start_time = time.time()
        
        self.classifier.fit(
            input_fn=train_input_fn,
            steps=steps
        )
        
        training_time = time.time() - start_time
        
        # Evaluate on validation set
        val_predictions = list(self.classifier.predict(input_fn=lambda: self._prepare_input_fn(X_val)()))
        val_pred_labels = [pred['classes'][0] for pred in val_predictions]
        
        # Calculate metrics
        tp = np.sum((val_pred_labels == 1) & (y_val == 1))
        tn = np.sum((val_pred_labels == 0) & (y_val == 0))
        fp = np.sum((val_pred_labels == 1) & (y_val == 0))
        fn = np.sum((val_pred_labels == 0) & (y_val == 1))
        
        accuracy = (tp + tn) / len(y_val) if len(y_val) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'training_time': training_time,
            'training_steps': steps
        }
        
        self.training_metrics = metrics
        self.is_trained = True
        
        logger.info(f"Training completed in {training_time:.2f}s")
        logger.info(f"Validation accuracy: {accuracy:.4f}")
        
        return metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 10,
        steps_per_fold: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on the dataset.
        
        Args:
            X: Feature matrix
            y: Labels
            cv_folds: Number of cross-validation folds
            steps_per_fold: Training steps per fold
            
        Returns:
            Dictionary containing detailed CV results
        """
        logger.info(f"Starting {cv_folds}-fold cross-validation")
        
        # Prepare k-fold splits (matching original implementation)
        all_indices_folds = [
            fold.tolist() for fold in np.array_split(range(len(X)), cv_folds)
        ]
        
        fold_metrics = []
        start_time = time.time()
        
        for fold in range(cv_folds):
            logger.info(f"Processing fold {fold + 1}/{cv_folds}")
            
            # Split data (matching original logic)
            train_indices = np.concatenate([
                x for i, x in enumerate(all_indices_folds) if i != fold
            ])
            test_indices = np.array(all_indices_folds[fold])
            
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_test_fold = X[test_indices]
            y_test_fold = y[test_indices]
            
            # Create new classifier for this fold
            fold_classifier = TFContribSVMClassifier(
                n_features=self.n_features,
                l2_regularization=self.l2_regularization,
                random_state=self.random_state + fold
            )
            
            # Train on fold
            fold_classifier.train(
                X_train_fold, 
                y_train_fold, 
                validation_split=0.0,  # Use all training data
                steps=steps_per_fold
            )
            
            # Predict on test fold
            test_input_fn = fold_classifier._prepare_input_fn(X_test_fold)
            predictions = list(fold_classifier.classifier.predict(input_fn=test_input_fn))
            pred_labels = [pred['classes'][0] for pred in predictions]
            
            # Calculate metrics (matching original output format)
            tp = np.sum((pred_labels == 1) & (y_test_fold == 1))
            tn = np.sum((pred_labels == 0) & (y_test_fold == 0))
            fp = np.sum((pred_labels == 1) & (y_test_fold == 0))
            fn = np.sum((pred_labels == 0) & (y_test_fold == 1))
            
            accuracy = (tp + tn) / len(y_test_fold) if len(y_test_fold) > 0 else 0
            
            # Per-class metrics (matching original terminology)
            coding_specificity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Coding sensitivity in original
            coding_sensitivity = tp / (tp + fp) if (tp + fp) > 0 else 0  # Coding specificity in original
            noncoding_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Non-coding sensitivity in original
            noncoding_sensitivity = tn / (tn + fn) if (tn + fn) > 0 else 0  # Non-coding specificity in original
            
            fold_metric = {
                'fold': fold + 1,
                'accuracy': accuracy,
                'coding_specificity': coding_specificity,
                'coding_sensitivity': coding_sensitivity, 
                'noncoding_specificity': noncoding_specificity,
                'noncoding_sensitivity': noncoding_sensitivity,
                'confusion_matrix': [[tn, fp], [fn, tp]]
            }
            
            # Calculate F-measures
            fold_metric['coding_f_measure'] = 2 * (coding_specificity * coding_sensitivity) / \
                                            (coding_specificity + coding_sensitivity) \
                                            if (coding_specificity + coding_sensitivity) > 0 else 0
            
            fold_metric['noncoding_f_measure'] = 2 * (noncoding_specificity * noncoding_sensitivity) / \
                                               (noncoding_specificity + noncoding_sensitivity) \
                                               if (noncoding_specificity + noncoding_sensitivity) > 0 else 0
            
            fold_metrics.append(fold_metric)
            
            logger.info(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}")
        
        cv_time = time.time() - start_time
        
        # Aggregate results (matching original format)
        results = {
            'cv_folds': cv_folds,
            'total_time': cv_time,
            'mean_accuracy': np.mean([f['accuracy'] for f in fold_metrics]),
            'std_accuracy': np.std([f['accuracy'] for f in fold_metrics]),
            'mean_coding_specificity': np.mean([f['coding_specificity'] for f in fold_metrics]),
            'mean_coding_sensitivity': np.mean([f['coding_sensitivity'] for f in fold_metrics]),
            'mean_coding_f_measure': np.mean([f['coding_f_measure'] for f in fold_metrics]),
            'mean_noncoding_specificity': np.mean([f['noncoding_specificity'] for f in fold_metrics]),
            'mean_noncoding_sensitivity': np.mean([f['noncoding_sensitivity'] for f in fold_metrics]),
            'mean_noncoding_f_measure': np.mean([f['noncoding_f_measure'] for f in fold_metrics]),
            'fold_details': fold_metrics
        }
        
        logger.info(f"Cross-validation completed in {cv_time:.2f}s")
        logger.info(f"Mean accuracy: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        input_fn = self._prepare_input_fn(X)
        predictions = list(self.classifier.predict(input_fn=input_fn))
        return np.array([pred['classes'][0] for pred in predictions])
    
    def save_metrics(self, metrics: Dict[str, Any], filepath: Path):
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics
            filepath: Path to save the metrics
        """
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def save_metrics_txt(self, metrics: Dict[str, Any], filepath: Path):
        """
        Save metrics in the original text format.
        
        Args:
            metrics: Dictionary of metrics
            filepath: Path to save the metrics
        """
        with open(filepath, 'w') as f:
            f.write(f"Cross-validation results ({metrics['cv_folds']} folds):\n")
            f.write(f"Mean accuracy: {metrics['mean_accuracy']:.6f} (+/- {metrics['std_accuracy']:.6f})\n")
            f.write(f"Mean coding specificity: {metrics['mean_coding_specificity']:.6f}\n")
            f.write(f"Mean coding sensitivity: {metrics['mean_coding_sensitivity']:.6f}\n")
            f.write(f"Mean coding F-measure: {metrics['mean_coding_f_measure']:.6f}\n")
            f.write(f"Mean non-coding specificity: {metrics['mean_noncoding_specificity']:.6f}\n")
            f.write(f"Mean non-coding sensitivity: {metrics['mean_noncoding_sensitivity']:.6f}\n")
            f.write(f"Mean non-coding F-measure: {metrics['mean_noncoding_f_measure']:.6f}\n")
            f.write(f"Total execution time: {metrics['total_time']:.2f} seconds\n\n")
            
            f.write("Detailed fold results:\n")
            for fold_data in metrics['fold_details']:
                f.write(f"Fold {fold_data['fold']}:\n")
                f.write(f"  Accuracy: {fold_data['accuracy']:.6f}\n")
                f.write(f"  Coding specificity: {fold_data['coding_specificity']:.6f}\n")
                f.write(f"  Coding sensitivity: {fold_data['coding_sensitivity']:.6f}\n")
                f.write(f"  Non-coding specificity: {fold_data['noncoding_specificity']:.6f}\n")
                f.write(f"  Non-coding sensitivity: {fold_data['noncoding_sensitivity']:.6f}\n")
                f.write(f"  Confusion matrix: {fold_data['confusion_matrix']}\n")
        
        logger.info(f"Metrics saved to {filepath} (original format)")