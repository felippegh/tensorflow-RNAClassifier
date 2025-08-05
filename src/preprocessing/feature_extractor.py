"""Feature extraction module for RNA sequences."""

import itertools
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import csv
from collections import Counter

from ..config.config import Config

logger = logging.getLogger(__name__)


class KmerFeatureExtractor:
    """Extract k-mer features from RNA sequences."""
    
    def __init__(self, kmer_sizes: List[int] = None):
        """
        Initialize the k-mer feature extractor.
        
        Args:
            kmer_sizes: List of k-mer sizes to extract. Defaults to Config.KMER_SIZES
        """
        self.kmer_sizes = kmer_sizes or Config.KMER_SIZES
        self.nucleotides = Config.NUCLEOTIDES
        self.feature_names = self._generate_feature_names()
        self.feature_to_index = {name: i for i, name in enumerate(self.feature_names)}
        
        logger.info(f"Initialized KmerFeatureExtractor with {len(self.feature_names)} features")
    
    def _generate_feature_names(self) -> List[str]:
        """Generate all possible k-mer combinations."""
        feature_names = []
        
        for k in self.kmer_sizes:
            kmers = [''.join(p) for p in itertools.product(self.nucleotides, repeat=k)]
            feature_names.extend(kmers)
        
        return feature_names
    
    def extract_features(self, sequence: str) -> np.ndarray:
        """
        Extract k-mer features from a single sequence.
        
        Args:
            sequence: RNA sequence string
            
        Returns:
            Feature vector as numpy array
        """
        sequence = sequence.upper().strip()
        
        # Filter out non-standard nucleotides
        valid_sequence = ''.join([n for n in sequence if n in self.nucleotides])
        
        if len(valid_sequence) == 0:
            logger.warning("Sequence contains no valid nucleotides")
            return np.zeros(len(self.feature_names))
        
        # Count k-mers
        feature_vector = np.zeros(len(self.feature_names))
        
        for k in self.kmer_sizes:
            if len(valid_sequence) >= k:
                for i in range(len(valid_sequence) - k + 1):
                    kmer = valid_sequence[i:i+k]
                    if kmer in self.feature_to_index:
                        idx = self.feature_to_index[kmer]
                        feature_vector[idx] += 1
        
        # Normalize by sequence length
        if len(valid_sequence) > 0:
            feature_vector = feature_vector / len(valid_sequence)
        
        return feature_vector
    
    def extract_features_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Extract features from multiple sequences.
        
        Args:
            sequences: List of RNA sequences
            
        Returns:
            Feature matrix (n_sequences x n_features)
        """
        features = []
        
        for i, seq in enumerate(sequences):
            if i % 100 == 0 and i > 0:
                logger.debug(f"Processed {i}/{len(sequences)} sequences")
            
            features.append(self.extract_features(seq))
        
        return np.array(features)


class SequenceProcessor:
    """Process and prepare RNA sequences for classification."""
    
    def __init__(self, feature_extractor: Optional[KmerFeatureExtractor] = None):
        """
        Initialize the sequence processor.
        
        Args:
            feature_extractor: KmerFeatureExtractor instance
        """
        self.feature_extractor = feature_extractor or KmerFeatureExtractor()
    
    def read_fasta(self, filepath: Path) -> List[Tuple[str, str]]:
        """
        Read sequences from FASTA file.
        
        Args:
            filepath: Path to FASTA file
            
        Returns:
            List of (header, sequence) tuples
        """
        sequences = []
        current_header = None
        current_sequence = []
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    if line.startswith('>'):
                        # Save previous sequence if exists
                        if current_header is not None:
                            sequences.append((current_header, ''.join(current_sequence)))
                        
                        current_header = line[1:]  # Remove '>'
                        current_sequence = []
                    else:
                        current_sequence.append(line)
                
                # Save last sequence
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_sequence)))
        
        except FileNotFoundError:
            logger.error(f"FASTA file not found: {filepath}")
            raise
        
        logger.info(f"Read {len(sequences)} sequences from {filepath}")
        return sequences
    
    def process_fasta_to_features(
        self,
        coding_file: Path,
        noncoding_file: Path,
        output_file: Path,
        max_sequences: Optional[int] = None,
        batch_size: int = 1000
    ) -> None:
        """
        Process FASTA files and extract features to CSV.
        
        Args:
            coding_file: Path to coding sequences FASTA
            noncoding_file: Path to non-coding sequences FASTA
            output_file: Path to output CSV file
            max_sequences: Maximum sequences per class (None for all)
            batch_size: Process sequences in batches
        """
        # Read sequences
        logger.info("Reading coding sequences...")
        coding_sequences = self.read_fasta(coding_file)
        
        logger.info("Reading non-coding sequences...")
        noncoding_sequences = self.read_fasta(noncoding_file)
        
        # Limit sequences if specified
        if max_sequences:
            coding_sequences = coding_sequences[:max_sequences]
            noncoding_sequences = noncoding_sequences[:max_sequences]
        
        # Prepare data
        all_sequences = []
        all_labels = []
        
        for header, seq in coding_sequences:
            all_sequences.append(seq)
            all_labels.append(1)  # Coding = 1
        
        for header, seq in noncoding_sequences:
            all_sequences.append(seq)
            all_labels.append(0)  # Non-coding = 0
        
        logger.info(f"Processing {len(all_sequences)} total sequences")
        
        # Extract features and write to CSV
        with open(output_file, 'w', newline='') as csvfile:
            # Write header
            header = self.feature_extractor.feature_names + ['label']
            writer = csv.writer(csvfile)
            writer.writerow(header)
            
            # Process in batches
            for i in range(0, len(all_sequences), batch_size):
                batch_sequences = all_sequences[i:i+batch_size]
                batch_labels = all_labels[i:i+batch_size]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_sequences)-1)//batch_size + 1}")
                
                # Extract features
                features = self.feature_extractor.extract_features_batch(batch_sequences)
                
                # Write to CSV
                for feat_vec, label in zip(features, batch_labels):
                    row = list(feat_vec) + [label]
                    writer.writerow(row)
        
        logger.info(f"Features saved to {output_file}")
    
    def load_features_from_csv(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features and labels from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        data = []
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            for row in reader:
                data.append([float(x) for x in row])
        
        data = np.array(data)
        
        # Last column is label
        features = data[:, :-1]
        labels = data[:, -1]
        
        logger.info(f"Loaded {len(features)} samples with {features.shape[1]} features")
        
        return features, labels