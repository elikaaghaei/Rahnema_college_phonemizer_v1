#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persian Phonemizer/Diacritization Dataset Loader
Author: Cursor:Claude-Sonnet
Date: 2025-10-22
Purpose: Load and preprocess Persian diacritization dataset with robust handling
"""

import os
import json
import random
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple, Literal, Optional, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset


# =============================================================================
# DIRECTORY SCANNING UTILITIES
# =============================================================================

def scan_directory(path: str = ".") -> List[str]:
    """
    Scan directory and return list of files/folders.
    
    Args:
        path: Directory path to scan (default: current directory)
    
    Returns:
        List of file/folder names found
        
    Examples:
        >>> files = scan_directory(".")
        >>> isinstance(files, list)
        True
    """
    path_obj = Path(path)
    items = []
    
    if path_obj.exists():
        for item in path_obj.iterdir():
            items.append(str(item.relative_to(path_obj)))
    
    return sorted(items)


def print_directory_contents(path: str = ".") -> None:
    """Print formatted directory listing."""
    print(f"\n{'='*60}")
    print(f"📁 Directory contents: {Path(path).absolute()}")
    print(f"{'='*60}")
    
    items = scan_directory(path)
    for item in items:
        icon = "📁" if Path(path) / item / "." in Path(path).rglob("*") else "📄"
        full_path = Path(path) / item
        if full_path.is_dir():
            icon = "📁"
        else:
            icon = "📄"
        print(f"  {icon} {item}")
    
    print(f"{'='*60}\n")


# =============================================================================
# PERSIAN TEXT PREPROCESSING
# =============================================================================

# Persian-specific character mappings
PERSIAN_NORMALIZATION = {
    'ك': 'ک',  # Arabic kaf -> Persian kaf
    'ي': 'ی',  # Arabic yeh -> Persian yeh
    'ى': 'ی',  # Alef maksura -> Persian yeh
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
}

# Zero-width and control characters to handle
ZERO_WIDTH_CHARS = [
    '\u200c',  # ZWNJ (Zero Width Non-Joiner)
    '\u200d',  # ZWJ (Zero Width Joiner)
    '\u200e',  # LRM (Left-to-Right Mark)
    '\u200f',  # RLM (Right-to-Left Mark)
    '\ufeff',  # Zero Width No-Break Space
]


def normalize_persian_text(
    text: str,
    preserve_diacritics: bool = True,
    normalize_zwj: bool = True
) -> str:
    """
    Normalize Persian text with Unicode normalization and character standardization.
    
    Args:
        text: Input Persian text
        preserve_diacritics: Keep diacritical marks (default: True)
        normalize_zwj: Replace ZWJ/ZWNJ with standard space (default: True)
    
    Returns:
        Normalized text string
        
    Examples:
        >>> normalize_persian_text("سلام")
        'سلام'
        >>> normalize_persian_text("كتاب")  # Arabic kaf
        'کتاب'
        >>> "ک" in normalize_persian_text("ك")  # Normalize to Persian
        True
    """
    if not text:
        return ""
    
    # Apply NFKC normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Persian character normalization
    for old_char, new_char in PERSIAN_NORMALIZATION.items():
        text = text.replace(old_char, new_char)
    
    # Handle zero-width characters
    if normalize_zwj:
        for zwchar in ZERO_WIDTH_CHARS:
            text = text.replace(zwchar, ' ')
    
    # Remove diacritics if requested
    if not preserve_diacritics:
        # Persian/Arabic diacritics range: \u064B-\u065F
        text = ''.join(
            char for char in text
            if not ('\u064B' <= char <= '\u065F')
        )
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()


# =============================================================================
# DATASET CLASS
# =============================================================================

class PhonemizerDataset(Dataset):
    """
    PyTorch Dataset for Persian diacritization/phonemization.
    
    Reads CSV with text columns and provides tokenized samples.
    Handles multiple column name variants robustly.
    
    Args:
        csv_path: Path to CSV file
        mode: Tokenization mode - "char" for character-level, "word" for word-level
        preserve_diacritics: Keep existing diacritics in preprocessing
        max_samples: Maximum number of samples to load (None = all)
        
    Examples:
        >>> # This would require an actual CSV file
        >>> # dataset = PhonemizerDataset("data.csv", mode="char")
        >>> # len(dataset) > 0
        >>> pass
    """
    
    # Column name alternatives (in order of preference)
    SOURCE_COLUMNS = ['text_no_eraab', 'text', 'sentence', 'input', 'source']
    TARGET_COLUMNS = ['sentence', 'phonemes', 'diacritized', 'targets', 'target', 'output']
    
    def __init__(
        self,
        csv_path: str,
        mode: Literal["char", "word"] = "char",
        preserve_diacritics: bool = True,
        max_samples: Optional[int] = None
    ):
        self.csv_path = Path(csv_path)
        self.mode = mode
        self.preserve_diacritics = preserve_diacritics
        
        # Validate file exists
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"❌ CSV file not found: {self.csv_path}\n"
                f"Please check the path."
            )
        
        # Load data
        self.df = pd.read_csv(self.csv_path)
        
        if len(self.df) == 0:
            raise ValueError(f"❌ CSV file is empty: {self.csv_path}")
        
        # Limit samples if requested
        if max_samples is not None and max_samples < len(self.df):
            self.df = self.df.head(max_samples)
        
        # Identify columns
        self.source_col = self._find_column(self.SOURCE_COLUMNS, "source/input")
        self.target_col = self._find_column(self.TARGET_COLUMNS, "target/output")
        
        print(f"✅ Dataset loaded: {len(self.df)} samples")
        print(f"   Input column: '{self.source_col}'")
        print(f"   Output column: '{self.target_col}'")
    
    def _find_column(self, candidates: List[str], col_type: str) -> str:
        """Find first matching column from candidates."""
        for col in candidates:
            if col in self.df.columns:
                return col
        
        raise ValueError(
            f"❌ No {col_type} column found.\n"
            f"   Available columns: {list(self.df.columns)}\n"
            f"   Expected columns: {candidates}"
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text based on mode."""
        if self.mode == "char":
            return list(text)
        elif self.mode == "word":
            return text.split()
        else:
            raise ValueError(f"❌ Invalid mode: {self.mode}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get single sample.
        
        Returns:
            Dict with keys: 'text', 'phonemes', 'tokens'
        """
        row = self.df.iloc[idx]
        
        # Get raw text
        source_raw = str(row[self.source_col])
        target_raw = str(row[self.target_col])
        
        # Preprocess
        source_normalized = normalize_persian_text(
            source_raw,
            preserve_diacritics=False,  # Source = no diacritics
            normalize_zwj=True
        )
        
        target_normalized = normalize_persian_text(
            target_raw,
            preserve_diacritics=self.preserve_diacritics,
            normalize_zwj=True
        )
        
        # Tokenize
        tokens = self._tokenize(source_normalized)
        
        return {
            'text': source_normalized,
            'phonemes': target_normalized,
            'tokens': tokens
        }


# =============================================================================
# COLLATE FUNCTION
# =============================================================================

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader - pads token sequences.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Dict with batched data and lengths
        
    Examples:
        >>> samples = [
        ...     {'text': 'سلام', 'phonemes': 'سَلام', 'tokens': ['س', 'ل', 'ا', 'م']},
        ...     {'text': 'به', 'phonemes': 'بِه', 'tokens': ['ب', 'ه']}
        ... ]
        >>> result = collate_fn(samples)
        >>> len(result['texts']) == 2
        True
    """
    texts = [item['text'] for item in batch]
    phonemes = [item['phonemes'] for item in batch]
    tokens = [item['tokens'] for item in batch]
    lengths = [len(t) for t in tokens]
    
    return {
        'texts': texts,
        'phonemes': phonemes,
        'tokens': tokens,
        'lengths': lengths,
        'batch_size': len(batch)
    }


# =============================================================================
# TRAIN/VAL SPLIT
# =============================================================================

def train_val_split(
    dataset: Dataset,
    val_frac: float = 0.1,
    seed: int = 42
) -> Tuple[Subset, Subset]:
    """
    Split dataset into train and validation subsets.
    
    Args:
        dataset: PyTorch Dataset object
        val_frac: Fraction for validation (default: 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_subset, val_subset)
        
    Examples:
        >>> from torch.utils.data import TensorDataset
        >>> dummy = TensorDataset(torch.randn(100, 5))
        >>> train, val = train_val_split(dummy, val_frac=0.2, seed=42)
        >>> len(train) + len(val) == 100
        True
    """
    total_size = len(dataset)
    val_size = int(total_size * val_frac)
    train_size = total_size - val_size
    
    # Deterministic shuffle
    indices = list(range(total_size))
    random.seed(seed)
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    print(f"📊 Dataset split:")
    print(f"   Training: {len(train_subset)} samples ({100*(1-val_frac):.1f}%)")
    print(f"   Validation: {len(val_subset)} samples ({100*val_frac:.1f}%)")
    
    return train_subset, val_subset


# =============================================================================
# SAMPLE CREATION & LOADING
# =============================================================================

def create_sample_file(
    csv_path: str,
    output_path: str = "data/sample_phonemizer.jsonl",
    n_samples: int = 200,
    seed: int = 42
) -> None:
    """
    Create sample JSONL file from CSV for quick testing.
    
    Args:
        csv_path: Path to full CSV dataset
        output_path: Where to save sample JSONL
        n_samples: Number of samples to extract
        seed: Random seed
    """
    df = pd.read_csv(csv_path)
    
    if len(df) <= n_samples:
        print(f"⚠️  Dataset smaller than {n_samples} samples, saving all.")
        sample_df = df
    else:
        sample_df = df.sample(n=n_samples, random_state=seed)
    
    # Create output directory
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in sample_df.iterrows():
            json.dump(row.to_dict(), f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Sample saved: {output_path} ({len(sample_df)} samples)")


def load_sample(path: str = "data/sample_phonemizer.jsonl") -> List[Dict[str, Any]]:
    """
    Load sample JSONL file.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of sample dictionaries
    """
    samples = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"✅ {len(samples)} samples loaded from {path}")
    return samples


# =============================================================================
# TEACHING SECTION
# =============================================================================

def TEACH_SECTION():
    """
    Line-by-Line Teaching Guide for Junior Developers
    ==================================================
    
    📚 Overview:
    This module is a Dataset loader for Persian texts with diacritics (diacritization).
    Goal: Read CSV, preprocess Persian text, and prepare for model training.
    
    🔍 Section Breakdown:
    
    1️⃣ DIRECTORY SCANNING (lines 30-60):
       - scan_directory(): Returns list of files in a folder
       - print_directory_contents(): Pretty display of folder contents
       Usage: Verify that required files exist
    
    2️⃣ PERSIAN TEXT PREPROCESSING (lines 65-145):
       - PERSIAN_NORMALIZATION: Maps Arabic to Persian characters (ك→ک, ي→ی)
       - ZERO_WIDTH_CHARS: Invisible characters (ZWNJ, ZWJ, ...)
       - normalize_persian_text(): 
         * NFKC normalization for Unicode standardization
         * Convert Arabic characters to Persian
         * Handle zero-width characters
         * Remove/preserve diacritics (optional)
         * Normalize whitespace
    
    3️⃣ DATASET CLASS (lines 150-265):
       - PhonemizerDataset(torch.utils.data.Dataset):
         * __init__: Load CSV and identify columns
         * _find_column(): Flexible column name search
         * _tokenize(): Convert text to tokens (character or word level)
         * __len__: Number of samples
         * __getitem__: Return one sample (preprocessed)
       
       Important note: Uses SOURCE_COLUMNS and TARGET_COLUMNS for handling
       different column names (flexibility).
    
    4️⃣ COLLATE FUNCTION (lines 270-295):
       - collate_fn(): Combine multiple samples into a batch
       - Currently simple padding (just lists)
       - Can be converted to tensors later
    
    5️⃣ TRAIN/VAL SPLIT (lines 300-345):
       - train_val_split(): Deterministic split into train/val
       - Uses random.seed for reproducibility
       - Returns: Two Subsets of the original dataset
    
    6️⃣ SAMPLE FILE UTILITIES (lines 350-410):
       - create_sample_file(): Save small sample for quick testing
       - load_sample(): Load JSONL sample
       Usage: Quick testing without loading entire dataset
    
    ⚠️ Three Common Debugging Tips:
    
    1. CSV Reading Issues:
       - Check encoding (should be UTF-8)
       - If UnicodeDecodeError: add to pd.read_csv:
         pd.read_csv(path, encoding='utf-8-sig')
       - Check separator (if using semicolon: sep=';')
    
    2. Unexpected Column Names:
       - Run print(df.columns)
       - Add new columns to SOURCE_COLUMNS/TARGET_COLUMNS
       - Possibly extra whitespace in column names: df.columns.str.strip()
    
    3. Text Normalization Problems:
       - Small test: normalize_persian_text("test ك ي")
       - Check if special characters are removed
       - For ZWJ/ZWNJ debugging: repr(text) to see hidden characters
    
    🎯 Three Recommended Next Tasks:
    
    1. Build Vocabulary Builder:
       - Vocab class that extracts all unique tokens
       - token→id and id→token mapping
       - Special tokens: <PAD>, <UNK>, <SOS>, <EOS>
       - Save/load vocab to JSON
    
    2. Advanced Collate Function:
       - Convert tokens to IDs using Vocab
       - Padding to fixed length
       - Create attention masks
       - Return torch.Tensor instead of lists
    
    3. Data Augmentation Utilities:
       - random_mask_chars() function: randomly hide characters
       - synonym_replacement() function: replace with synonyms
       - back_translation() stub for future
       - Maintain balance between input and output
    
    💡 Optimization Notes:
    - For large datasets (>1M): use chunked reading
    - Cache preprocessing in __init__ (if enough RAM)
    - Use multiprocessing in DataLoader: num_workers=4
    """
    pass


# =============================================================================
# MAIN - RUNNABLE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Running data/loader.py example")
    print("="*60)
    
    # 1. Print directory contents
    print_directory_contents("/workspace")
    
    # 2. Try to find the dataset
    possible_paths = [
        "/workspace/phonemizer _dataset_v1.csv/phonemizer _dataset_v1.csv",
        "/workspace/phonemizer_dataset_v1.csv",
        "phonemizer_dataset_v1.csv"
    ]
    
    dataset_path = None
    for path in possible_paths:
        if Path(path).exists():
            dataset_path = path
            break
    
    if dataset_path is None:
        print("❌ CSV file not found!")
        print("\n📝 Please create a sample CSV file with this format:")
        print("="*60)
        print("sentence,text_no_eraab")
        print("سَلامُ,سلام")
        print("کِتابِ خوب,کتاب خوب")
        print("دُنیایِ زیبا,دنیای زیبا")
        print("="*60)
        exit(1)
    
    print(f"✅ Dataset file found: {dataset_path}\n")
    
    # 3. Load dataset and show 3 samples
    print("📖 Loading dataset...")
    dataset = PhonemizerDataset(
        csv_path=dataset_path,
        mode="char",
        preserve_diacritics=True,
        max_samples=1000  # For quick testing
    )
    
    print(f"\n{'='*60}")
    print("📋 Showing 3 samples from dataset:")
    print("="*60)
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample #{i+1}:")
        print(f"  Input (without diacritics): {sample['text']}")
        print(f"  Output (with diacritics): {sample['phonemes']}")
        print(f"  Tokens: {sample['tokens'][:20]}...")  # First 20 tokens
        print(f"  Token count: {len(sample['tokens'])}")
    
    # 4. Create DataLoader and show one batch
    print(f"\n{'='*60}")
    print("🔄 Creating DataLoader...")
    print("="*60)
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    
    print(f"\n📦 Batch summary:")
    print(f"  Batch size: {batch['batch_size']}")
    print(f"  Number of texts: {len(batch['texts'])}")
    print(f"  Token lengths: {batch['lengths']}")
    print(f"\n  First sample:")
    print(f"    Text: {batch['texts'][0]}")
    print(f"    Phonemes: {batch['phonemes'][0]}")
    
    # 5. Create sample file if dataset is large enough
    df = pd.read_csv(dataset_path)
    if len(df) > 500:
        print(f"\n{'='*60}")
        print("💾 Creating sample file (dataset larger than 500 lines)...")
        print("="*60)
        create_sample_file(
            csv_path=dataset_path,
            output_path="/workspace/data/sample_phonemizer.jsonl",
            n_samples=200,
            seed=42
        )
    
    # 6. Test train/val split
    print(f"\n{'='*60}")
    print("✂️  Testing train/val split...")
    print("="*60)
    train_subset, val_subset = train_val_split(dataset, val_frac=0.15, seed=42)
    
    print("\n" + "="*60)
    print("✅ Successful run! All tests passed.")
    print("="*60 + "\n")
