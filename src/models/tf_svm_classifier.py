"""TensorFlow 1.x SVM Classifier for RNA sequence classification."""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import csv

from ..config.config import Config

# Ensure TF 1.x behavior
tf.compat.v1.disable_v2_behavior()

logger = logging.getLogger(__name__)


class TensorFlowSVMClassifier:
    """TensorFlow 1.x SVM classifier for RNA sequences."""
    
    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        l2_regularization: float = 0.1,
        batch_size: int = 100,
        n_epochs: int = 100,
        random_state: int = 42
    ):
        """
        Initialize the TensorFlow SVM classifier.
        
        Args:
            n_features: Number of input features
            learning_rate: Learning rate for optimizer
            l2_regularization: L2 regularization strength
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            random_state: Random seed for reproducibility
        """
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state
        
        self.graph = None
        self.session = None
        self.saver = None
        self.is_trained = False
        self.training_metrics = {}
        
        # Set random seeds
        np.random.seed(random_state)
        tf.compat.v1.set_random_seed(random_state)
        
        logger.info(f"Initialized TensorFlowSVMClassifier with {n_features} features")
    
    def _build_graph(self):
        """Build the TensorFlow computation graph for SVM."""
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            # Input placeholders
            self.X_placeholder = tf.compat.v1.placeholder(
                tf.float32, 
                shape=[None, self.n_features],
                name='X_input'
            )
            self.y_placeholder = tf.compat.v1.placeholder(
                tf.float32,
                shape=[None],
                name='y_labels'
            )
            
            # SVM weights and bias
            self.W = tf.Variable(
                tf.random.normal([self.n_features, 1], stddev=0.01),
                name='weights'
            )
            self.b = tf.Variable(
                tf.zeros([1]),
                name='bias'
            )
            
            # Linear model
            self.scores = tf.matmul(self.X_placeholder, self.W) + self.b
            self.predictions = tf.sign(self.scores)
            
            # Convert labels from {0, 1} to {-1, 1} for SVM
            y_svm = 2 * self.y_placeholder - 1
            y_svm = tf.reshape(y_svm, [-1, 1])
            
            # Hinge loss
            margins = y_svm * self.scores
            hinge_loss = tf.reduce_mean(tf.maximum(0.0, 1.0 - margins))
            
            # L2 regularization
            l2_loss = self.l2_regularization * tf.reduce_sum(tf.square(self.W))
            
            # Total loss
            self.loss = hinge_loss + l2_loss
            
            # Optimizer
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
            
            # Metrics
            predictions_binary = tf.cast(tf.greater(self.scores, 0), tf.float32)
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(predictions_binary, tf.reshape(self.y_placeholder, [-1, 1])), tf.float32)
            )
            
            # Initialize variables
            self.init_op = tf.compat.v1.global_variables_initializer()
            
            # Saver for model persistence
            self.saver = tf.compat.v1.train.Saver()
        
        logger.info("TensorFlow computation graph built")
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the SVM classifier.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            y: Labels (n_samples,) with values 0 or 1
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training TF-SVM on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Build graph if not already built
        if self.graph is None:
            self._build_graph()
        
        # Split data for validation
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Start TensorFlow session
        self.session = tf.compat.v1.Session(graph=self.graph)
        
        with self.session.as_default():
            with self.graph.as_default():
                # Initialize variables
                self.session.run(self.init_op)
                
                # Training loop
                start_time = time.time()
                train_losses = []
                val_accuracies = []
                
                n_batches = len(X_train) // self.batch_size
                
                for epoch in range(self.n_epochs):
                    # Shuffle training data
                    shuffle_idx = np.random.permutation(len(X_train))
                    X_train_shuffled = X_train[shuffle_idx]
                    y_train_shuffled = y_train[shuffle_idx]
                    
                    epoch_loss = 0
                    
                    # Mini-batch training
                    for batch in range(n_batches):
                        start_idx = batch * self.batch_size
                        end_idx = start_idx + self.batch_size
                        
                        batch_X = X_train_shuffled[start_idx:end_idx]
                        batch_y = y_train_shuffled[start_idx:end_idx]
                        
                        _, batch_loss = self.session.run(
                            [self.train_op, self.loss],
                            feed_dict={
                                self.X_placeholder: batch_X,
                                self.y_placeholder: batch_y
                            }
                        )
                        
                        epoch_loss += batch_loss
                    
                    # Calculate validation accuracy
                    val_acc = self.session.run(
                        self.accuracy,
                        feed_dict={
                            self.X_placeholder: X_val,
                            self.y_placeholder: y_val
                        }
                    )
                    
                    train_losses.append(epoch_loss / n_batches)
                    val_accuracies.append(val_acc)
                    
                    if (epoch + 1) % 10 == 0:
                        logger.info(
                            f"Epoch {epoch + 1}/{self.n_epochs} - "
                            f"Loss: {epoch_loss / n_batches:.4f}, "
                            f"Val Acc: {val_acc:.4f}"
                        )
                
                training_time = time.time() - start_time
                
                # Final evaluation
                final_predictions = self.session.run(
                    self.predictions,
                    feed_dict={self.X_placeholder: X_val}
                ).flatten()
                
                # Convert predictions from {-1, 1} to {0, 1}
                final_predictions = (final_predictions > 0).astype(float)
                
                # Calculate metrics
                tp = np.sum((final_predictions == 1) & (y_val == 1))
                tn = np.sum((final_predictions == 0) & (y_val == 0))
                fp = np.sum((final_predictions == 1) & (y_val == 0))
                fn = np.sum((final_predictions == 0) & (y_val == 1))
                
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
                    'final_loss': train_losses[-1],
                    'best_val_accuracy': max(val_accuracies)
                }
                
                self.training_metrics = metrics
                self.is_trained = True
                
                logger.info(f"Training completed in {training_time:.2f}s")
                logger.info(f"Final validation accuracy: {accuracy:.4f}")
        
        return metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 10
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on the dataset.
        
        Args:
            X: Feature matrix
            y: Labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing detailed CV results
        """
        logger.info(f"Starting {cv_folds}-fold cross-validation")
        
        # Prepare for k-fold CV
        fold_size = len(X) // cv_folds
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        fold_metrics = []
        start_time = time.time()
        
        for fold in range(cv_folds):
            logger.info(f"Processing fold {fold + 1}/{cv_folds}")
            
            # Split data
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < cv_folds - 1 else len(X)
            
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
            
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_val_fold = X[val_indices]
            y_val_fold = y[val_indices]
            
            # Create new classifier for this fold
            fold_classifier = TensorFlowSVMClassifier(
                n_features=self.n_features,
                learning_rate=self.learning_rate,
                l2_regularization=self.l2_regularization,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                random_state=self.random_state + fold
            )
            
            # Train on fold
            fold_classifier._build_graph()
            
            with tf.compat.v1.Session(graph=fold_classifier.graph) as sess:
                with fold_classifier.graph.as_default():
                    sess.run(fold_classifier.init_op)
                    
                    # Training loop (simplified for CV)
                    for epoch in range(self.n_epochs):
                        for i in range(0, len(X_train_fold), self.batch_size):
                            batch_X = X_train_fold[i:i + self.batch_size]
                            batch_y = y_train_fold[i:i + self.batch_size]
                            
                            sess.run(
                                fold_classifier.train_op,
                                feed_dict={
                                    fold_classifier.X_placeholder: batch_X,
                                    fold_classifier.y_placeholder: batch_y
                                }
                            )
                    
                    # Evaluate on validation fold
                    predictions = sess.run(
                        fold_classifier.predictions,
                        feed_dict={
                            fold_classifier.X_placeholder: X_val_fold
                        }
                    ).flatten()
                    
                    # Convert predictions from {-1, 1} to {0, 1}
                    predictions = (predictions > 0).astype(float)
                    
                    # Calculate metrics
                    tp = np.sum((predictions == 1) & (y_val_fold == 1))
                    tn = np.sum((predictions == 0) & (y_val_fold == 0))
                    fp = np.sum((predictions == 1) & (y_val_fold == 0))
                    fn = np.sum((predictions == 0) & (y_val_fold == 1))
                    
                    accuracy = (tp + tn) / len(y_val_fold) if len(y_val_fold) > 0 else 0
                    
                    fold_metric = {
                        'fold': fold + 1,
                        'accuracy': accuracy,
                        'coding_precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                        'coding_recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                        'noncoding_precision': tn / (tn + fn) if (tn + fn) > 0 else 0,
                        'noncoding_recall': tn / (tn + fp) if (tn + fp) > 0 else 0,
                        'confusion_matrix': [[tn, fp], [fn, tp]]
                    }
                    
                    # Calculate F-measures
                    fold_metric['coding_f1'] = 2 * (fold_metric['coding_precision'] * fold_metric['coding_recall']) / \
                                              (fold_metric['coding_precision'] + fold_metric['coding_recall']) \
                                              if (fold_metric['coding_precision'] + fold_metric['coding_recall']) > 0 else 0
                    
                    fold_metric['noncoding_f1'] = 2 * (fold_metric['noncoding_precision'] * fold_metric['noncoding_recall']) / \
                                                  (fold_metric['noncoding_precision'] + fold_metric['noncoding_recall']) \
                                                  if (fold_metric['noncoding_precision'] + fold_metric['noncoding_recall']) > 0 else 0
                    
                    fold_metrics.append(fold_metric)
            
            # Clean up TensorFlow graph for this fold
            tf.compat.v1.reset_default_graph()
        
        cv_time = time.time() - start_time
        
        # Aggregate results
        results = {
            'cv_folds': cv_folds,
            'total_time': cv_time,
            'mean_accuracy': np.mean([f['accuracy'] for f in fold_metrics]),
            'std_accuracy': np.std([f['accuracy'] for f in fold_metrics]),
            'mean_coding_precision': np.mean([f['coding_precision'] for f in fold_metrics]),
            'mean_coding_recall': np.mean([f['coding_recall'] for f in fold_metrics]),
            'mean_coding_f1': np.mean([f['coding_f1'] for f in fold_metrics]),
            'mean_noncoding_precision': np.mean([f['noncoding_precision'] for f in fold_metrics]),
            'mean_noncoding_recall': np.mean([f['noncoding_recall'] for f in fold_metrics]),
            'mean_noncoding_f1': np.mean([f['noncoding_f1'] for f in fold_metrics]),
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
        
        with self.session.as_default():
            with self.graph.as_default():
                predictions = self.session.run(
                    self.predictions,
                    feed_dict={self.X_placeholder: X}
                ).flatten()
                
                # Convert from {-1, 1} to {0, 1}
                return (predictions > 0).astype(float)
    
    def save_model(self, filepath: Path):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        with self.session.as_default():
            with self.graph.as_default():
                save_path = self.saver.save(self.session, str(filepath))
                logger.info(f"Model saved to {save_path}")
        
        # Save metadata
        metadata = {
            'n_features': self.n_features,
            'learning_rate': self.learning_rate,
            'l2_regularization': self.l2_regularization,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'training_metrics': self.training_metrics
        }
        
        metadata_path = Path(str(filepath) + '.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, filepath: Path):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        # Load metadata
        metadata_path = Path(str(filepath) + '.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.n_features = metadata['n_features']
        self.learning_rate = metadata['learning_rate']
        self.l2_regularization = metadata['l2_regularization']
        self.batch_size = metadata['batch_size']
        self.n_epochs = metadata['n_epochs']
        self.training_metrics = metadata['training_metrics']
        
        # Build graph and restore weights
        self._build_graph()
        self.session = tf.compat.v1.Session(graph=self.graph)
        
        with self.session.as_default():
            with self.graph.as_default():
                self.saver.restore(self.session, str(filepath))
                self.is_trained = True
                logger.info(f"Model loaded from {filepath}")
    
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
    
    def close(self):
        """Close the TensorFlow session."""
        if self.session is not None:
            self.session.close()
            self.session = None
            logger.info("TensorFlow session closed")