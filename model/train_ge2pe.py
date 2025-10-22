#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GE2PE Model Training Script for Persian Diacritization
Author: Cursor:Claude-Sonnet
Date: 2025-10-22
Purpose: Train T5-based GE2PE model on Persian diacritization dataset
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer, AdamW
from tqdm import tqdm

# Import custom modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import PhonemizerDataset, train_val_split, collate_fn
from data.tokenizer import PersianTokenizer


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
        
    Examples:
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 5)
        >>> count_parameters(model)
        55
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Character Error Rate (CER) using edit distance.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        
    Returns:
        CER as float (0.0 = perfect, 1.0 = complete mismatch)
    """
    def edit_distance(s1, s2):
        """Levenshtein distance."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] if s1[i-1] == s2[j-1] else \
                          1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]
    
    total_errors = sum(edit_distance(p, r) for p, r in zip(predictions, references))
    total_chars = sum(len(r) for r in references)
    return total_errors / total_chars if total_chars > 0 else 0.0


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_dir: str,
    config: Dict[str, Any]
) -> None:
    """
    Save model checkpoint and training config.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        checkpoint_dir: Directory to save checkpoint
        config: Training configuration dict
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Save model state
    model_path = checkpoint_path / f"model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, model_path)
    
    # Save config
    config_path = checkpoint_path / "training_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Checkpoint saved: {model_path}")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tokenizer: AutoTokenizer,
    max_grad_norm: float = 1.0
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: T5 model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: CUDA or CPU device
        tokenizer: Hugging Face tokenizer
        max_grad_norm: Gradient clipping threshold
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        # Get texts
        source_texts = batch['texts']  # Without diacritics
        target_texts = batch['phonemes']  # With diacritics
        
        # Tokenize
        inputs = tokenizer(
            source_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)
        
        targets = tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)
        
        # Forward pass
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=targets['input_ids']
        )
        
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer: AutoTokenizer
) -> Tuple[float, float]:
    """
    Validate model.
    
    Args:
        model: T5 model
        dataloader: Validation dataloader
        device: CUDA or CPU device
        tokenizer: Hugging Face tokenizer
        
    Returns:
        Tuple of (average_loss, CER)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    predictions = []
    references = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in pbar:
            source_texts = batch['texts']
            target_texts = batch['phonemes']
            
            # Tokenize
            inputs = tokenizer(
                source_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)
            
            targets = tokenizer(
                target_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)
            
            # Forward pass
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=targets['input_ids']
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            
            # Generate predictions for CER
            generated = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=128
            )
            
            pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            predictions.extend(pred_texts)
            references.extend(target_texts)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    cer = compute_cer(predictions, references)
    
    return avg_loss, cer


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: AutoTokenizer,
    num_epochs: int = 10,
    learning_rate: float = 5e-5,
    checkpoint_dir: str = "checkpoints",
    save_every: int = 2,
    device: Optional[torch.device] = None
) -> Dict[str, List[float]]:
    """
    Full training loop.
    
    Args:
        model: T5 model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        tokenizer: Hugging Face tokenizer
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        checkpoint_dir: Directory for checkpoints
        save_every: Save checkpoint every N epochs
        device: CUDA or CPU device
        
    Returns:
        Dictionary with training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training config
    config = {
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'batch_size': train_loader.batch_size,
        'model_params': count_parameters(model),
        'device': str(device)
    }
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting training on {device}")
    print(f"{'='*60}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"{'='*60}\n")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_cer': []
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, tokenizer)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss, val_cer = validate(model, val_loader, device, tokenizer)
        history['val_loss'].append(val_loss)
        history['val_cer'].append(val_cer)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | CER: {val_cer:.4f}")
        
        # Save checkpoint
        if epoch % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                checkpoint_dir, config
            )
    
    print(f"\n{'='*60}")
    print("✅ Training completed!")
    print(f"{'='*60}\n")
    
    return history


# =============================================================================
# TEACHING SECTION
# =============================================================================

"""
TEACHING: GE2PE Training Script
================================

📚 OVERVIEW:
T5-based model for Persian diacritization. Input: no diacritics → Output: with diacritics

🔍 COMPONENTS:
1. count_parameters(): Count trainable params
2. compute_cer(): Character Error Rate (edit distance)
3. train_epoch(): One training pass with gradient clipping
4. validate(): Evaluation with loss + CER
5. train_model(): Full loop with checkpointing

⚠️ DEBUGGING:
1. CUDA OOM: Reduce batch_size (16→8) or max_length (128→64)
2. Dimension mismatch: Use return_tensors='pt', check shapes
3. CER not improving: Verify tokenizer matches model checkpoint

🎯 NEXT TASKS:
1. model/metrics.py: Per-diacritic accuracy, confusion matrix, BLEU
2. model/inference.py: FastAPI endpoint for deployment
"""


# =============================================================================
# MAIN - RUNNABLE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 GE2PE Training Script - Demo Mode")
    print("="*60)
    
    # Config
    DATASET_PATH = "phonemizer _dataset_v1.csv/phonemizer _dataset_v1.csv"
    MODEL_NAME = "google/mt5-small"
    BATCH_SIZE, NUM_EPOCHS, MAX_SAMPLES = 4, 2, 100
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not Path(DATASET_PATH).exists():
        print(f"❌ Dataset not found: {DATASET_PATH}")
        exit(1)
    
    # Load dataset
    print("\n1️⃣  Loading dataset...")
    dataset = PhonemizerDataset(
        csv_path=DATASET_PATH, mode="char",
        preserve_diacritics=True, max_samples=MAX_SAMPLES
    )
    train_dataset, val_dataset = train_val_split(dataset, val_frac=0.2)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Load model
    print("\n2️⃣  Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test batch
    print("\n3️⃣  Testing batch shapes...")
    batch = next(iter(train_loader))
    print(f"Batch size: {batch['batch_size']}")
    inputs = tokenizer(batch['texts'][:2], padding=True, 
                      truncation=True, max_length=128, return_tensors='pt')
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    # Train
    print("\n4️⃣  Training (demo: 2 epochs, 100 samples)...")
    history = train_model(
        model, train_loader, val_loader, tokenizer,
        num_epochs=NUM_EPOCHS, learning_rate=5e-5,
        checkpoint_dir="checkpoints_demo", device=device
    )
    
    # Summary
    print(f"\n5️⃣  Results:")
    print(f"Train loss: {history['train_loss'][-1]:.4f}")
    print(f"Val loss: {history['val_loss'][-1]:.4f}")
    print(f"CER: {history['val_cer'][-1]:.4f}")
    print("\n✅ Demo completed!")
    print("Next: Train on full dataset, implement metrics, create API\n")
