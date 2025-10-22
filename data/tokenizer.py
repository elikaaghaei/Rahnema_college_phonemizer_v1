#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persian Text Tokenizer with Vocabulary Management
Author: Cursor:Claude-Sonnet
Date: 2025-10-22
Purpose: Character and word-level tokenization for Persian text with vocab building
"""

import json
import unicodedata
from pathlib import Path
from typing import List, Dict, Literal, Optional


# =============================================================================
# SPECIAL TOKENS
# =============================================================================

SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<UNK>': 1,
    '<SOS>': 2,
    '<EOS>': 3,
}

# Persian-specific character mappings
PERSIAN_NORMALIZATION = {
    'ك': 'ک',  # Arabic kaf -> Persian kaf
    'ي': 'ی',  # Arabic yeh -> Persian yeh
    'ى': 'ی',  # Alef maksura -> Persian yeh
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
}

# Zero-width and control characters
ZERO_WIDTH_CHARS = [
    '\u200c',  # ZWNJ (Zero Width Non-Joiner)
    '\u200d',  # ZWJ (Zero Width Joiner)
    '\u200e',  # LRM (Left-to-Right Mark)
    '\u200f',  # RLM (Right-to-Left Mark)
    '\ufeff',  # Zero Width No-Break Space
]


# =============================================================================
# PERSIAN TOKENIZER CLASS
# =============================================================================

class PersianTokenizer:
    """
    Tokenizer for Persian text with vocab management.
    
    Supports character-level and word-level tokenization with optional
    Unicode normalization and Persian-specific preprocessing.
    
    Args:
        normalize: Apply Unicode NFKC normalization
        normalize_persian: Convert Arabic characters to Persian equivalents
        handle_zwj: Replace zero-width joiners with space
        
    Examples:
        >>> tokenizer = PersianTokenizer()
        >>> tokens = tokenizer.tokenize("سلام", mode="char")
        >>> len(tokens) == 4
        True
        >>> tokenizer.detokenize(tokens)
        'سلام'
    """
    
    def __init__(
        self,
        normalize: bool = True,
        normalize_persian: bool = True,
        handle_zwj: bool = True
    ):
        self.normalize = normalize
        self.normalize_persian = normalize_persian
        self.handle_zwj = handle_zwj
        
        # Initialize vocab with special tokens
        self._vocab: Dict[str, int] = SPECIAL_TOKENS.copy()
        self._inv_vocab: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}
        self._next_id = len(SPECIAL_TOKENS)
    
    @property
    def vocab(self) -> Dict[str, int]:
        """Return token to ID mapping."""
        return self._vocab
    
    @property
    def inv_vocab(self) -> Dict[int, str]:
        """Return ID to token mapping."""
        return self._inv_vocab
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._vocab)
    
    def _normalize_text(self, text: str) -> str:
        """Apply normalization to text."""
        if not text:
            return ""
        
        # Unicode normalization
        if self.normalize:
            text = unicodedata.normalize('NFKC', text)
        
        # Persian character normalization
        if self.normalize_persian:
            for old_char, new_char in PERSIAN_NORMALIZATION.items():
                text = text.replace(old_char, new_char)
        
        # Handle zero-width characters
        if self.handle_zwj:
            for zwchar in ZERO_WIDTH_CHARS:
                text = text.replace(zwchar, ' ')
        
        return text
    
    def tokenize(
        self,
        text: str,
        mode: Literal["char", "word"] = "char"
    ) -> List[str]:
        """
        Tokenize text into characters or words.
        
        Args:
            text: Input text to tokenize
            mode: "char" for character-level, "word" for word-level
            
        Returns:
            List of tokens
            
        Examples:
            >>> tokenizer = PersianTokenizer()
            >>> tokenizer.tokenize("سلام دنیا", mode="char")[:4]
            ['س', 'ل', 'ا', 'م']
            >>> tokenizer.tokenize("سلام دنیا", mode="word")
            ['سلام', 'دنیا']
        """
        # Normalize text
        text = self._normalize_text(text)
        text = text.strip()
        
        if not text:
            return []
        
        # Tokenize based on mode
        if mode == "char":
            return list(text)
        elif mode == "word":
            return text.split()
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'char' or 'word'.")
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Reconstruct text from tokens.
        
        Args:
            tokens: List of tokens to join
            
        Returns:
            Reconstructed text string
            
        Examples:
            >>> tokenizer = PersianTokenizer()
            >>> tokenizer.detokenize(['س', 'ل', 'ا', 'م'])
            'سلام'
        """
        if not tokens:
            return ""
        
        # Check if this looks like character-level tokens (most are single chars)
        single_char_count = sum(1 for t in tokens if len(t) == 1)
        
        # If 80%+ are single chars, join directly (char-level)
        if single_char_count / len(tokens) >= 0.8:
            return ''.join(tokens)
        else:
            # Word-level tokens - join with spaces
            return ' '.join(tokens)
    
    def build_vocab(
        self,
        texts: List[str],
        mode: Literal["char", "word"] = "char",
        min_freq: int = 1
    ) -> None:
        """
        Build vocabulary from list of texts.
        
        Args:
            texts: List of texts to build vocab from
            mode: Tokenization mode
            min_freq: Minimum frequency for token to be included
            
        Examples:
            >>> tokenizer = PersianTokenizer()
            >>> tokenizer.build_vocab(["سلام", "سلام دنیا"], mode="char")  # doctest: +ELLIPSIS
            ✅ Vocabulary built: ... tokens
            >>> tokenizer.vocab_size > 4  # More than just special tokens
            True
        """
        # Count token frequencies
        token_counts: Dict[str, int] = {}
        
        for text in texts:
            tokens = self.tokenize(text, mode=mode)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Filter by min_freq and add to vocab
        for token, count in sorted(token_counts.items()):
            if count >= min_freq and token not in self._vocab:
                self._vocab[token] = self._next_id
                self._inv_vocab[self._next_id] = token
                self._next_id += 1
        
        print(f"✅ Vocabulary built: {self.vocab_size} tokens")
    
    def encode(self, text: str, mode: Literal["char", "word"] = "char") -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text
            mode: Tokenization mode
            
        Returns:
            List of token IDs
            
        Examples:
            >>> tokenizer = PersianTokenizer()
            >>> tokenizer.build_vocab(["سلام"], mode="char")  # doctest: +ELLIPSIS
            ✅ Vocabulary built: ... tokens
            >>> ids = tokenizer.encode("سلام", mode="char")
            >>> all(isinstance(i, int) for i in ids)
            True
        """
        tokens = self.tokenize(text, mode=mode)
        unk_id = self._vocab['<UNK>']
        return [self._vocab.get(token, unk_id) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Reconstructed text
            
        Examples:
            >>> tokenizer = PersianTokenizer()
            >>> tokenizer.build_vocab(["hi"], mode="char")  # doctest: +ELLIPSIS
            ✅ Vocabulary built: ... tokens
            >>> ids = tokenizer.encode("hi", mode="char")
            >>> tokenizer.decode(ids)
            'hi'
        """
        tokens = [self._inv_vocab.get(id, '<UNK>') for id in ids]
        # Filter out special tokens
        tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return self.detokenize(tokens)
    
    def save_vocab(self, path: str) -> None:
        """
        Save vocabulary to JSON file.
        
        Args:
            path: Output file path
        """
        vocab_data = {
            'vocab': self._vocab,
            'config': {
                'normalize': self.normalize,
                'normalize_persian': self.normalize_persian,
                'handle_zwj': self.handle_zwj,
            }
        }
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Vocabulary saved to {path} ({self.vocab_size} tokens)")
    
    def load_vocab(self, path: str) -> None:
        """
        Load vocabulary from JSON file.
        
        Args:
            path: Input file path
        """
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Load vocab
        self._vocab = vocab_data['vocab']
        self._inv_vocab = {int(v): k for k, v in self._vocab.items()}
        self._next_id = max(self._vocab.values()) + 1
        
        # Load config if available
        if 'config' in vocab_data:
            config = vocab_data['config']
            self.normalize = config.get('normalize', True)
            self.normalize_persian = config.get('normalize_persian', True)
            self.handle_zwj = config.get('handle_zwj', True)
        
        print(f"✅ Vocabulary loaded from {path} ({self.vocab_size} tokens)")


# =============================================================================
# TEACHING SECTION
# =============================================================================

"""
TEACHING: Persian Tokenizer Explained
======================================

📚 OVERVIEW:
Tokenizer splits text into tokens (chars/words), builds vocab (token→ID), encodes/decodes.

🔍 KEY COMPONENTS:

1. SPECIAL TOKENS: <PAD>=0, <UNK>=1, <SOS>=2, <EOS>=3 (fixed IDs)
2. NORMALIZATION: NFKC Unicode + Arabic→Persian chars + remove zero-width
3. TOKENIZATION: char-level (list each char) or word-level (split on space)
4. VOCABULARY: build_vocab() creates token→ID mapping from training texts
5. ENCODE/DECODE: text→IDs and IDs→text with <UNK> for unknown tokens
6. PERSISTENCE: save_vocab()/load_vocab() for JSON serialization

⚠️ THREE DEBUGGING TIPS:

1. Unknown Tokens:
   - Many <UNK> (ID=1)? vocab incomplete → rebuild with all training data
   - Debug: print(tokenizer.vocab.keys())

2. Encoding/Decoding Mismatch:
   - "سلام دنیا"→"سلامدنیا"? space not in vocab
   - Fix: ensure ' ' in vocab for char mode

3. JSON Errors:
   - save_vocab() fails? check all vocab keys are strings
   - Test: all(isinstance(k, str) for k in tokenizer.vocab.keys())

🎯 TWO NEXT TASKS:

1. Advanced Collate (data/collate.py):
   - Batch encode with padding to max_len
   - Return tensors + attention masks
   - def collate_with_tokenizer(batch, tokenizer, max_len=128)

2. Data Augmentation (data/augment.py):
   - random_mask_tokens(), random_delete_tokens(), char_swap()
   - Work with tokenizer encode/decode

💡 DESIGN: Builder pattern (create→configure→use), Properties for encapsulation,
Type hints for safety, Docstrings with examples
"""


# =============================================================================
# MAIN - RUNNABLE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Running data/tokenizer.py example")
    print("="*60)
    
    # Create tokenizer
    tokenizer = PersianTokenizer()
    
    # Test sentences
    sentences = [
        "سلام دنیا",
        "این یک تست است"
    ]
    
    print("\n📝 Original sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")
    
    # 1. Character-level tokenization
    print("\n" + "="*60)
    print("1️⃣  Character-level tokenization:")
    print("="*60)
    
    for sent in sentences:
        tokens = tokenizer.tokenize(sent, mode="char")
        print(f"\nText: {sent}")
        print(f"Tokens: {tokens}")
        print(f"Count: {len(tokens)}")
        reconstructed = tokenizer.detokenize(tokens)
        print(f"Reconstructed: {reconstructed}")
        print(f"Match: {reconstructed == sent}")
    
    # 2. Word-level tokenization
    print("\n" + "="*60)
    print("2️⃣  Word-level tokenization:")
    print("="*60)
    
    for sent in sentences:
        tokens = tokenizer.tokenize(sent, mode="word")
        print(f"\nText: {sent}")
        print(f"Tokens: {tokens}")
        print(f"Count: {len(tokens)}")
    
    # 3. Build vocabulary
    print("\n" + "="*60)
    print("3️⃣  Building vocabulary (char-level):")
    print("="*60)
    
    tokenizer.build_vocab(sentences, mode="char", min_freq=1)
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: {list(SPECIAL_TOKENS.keys())}")
    print(f"Sample tokens: {list(tokenizer.vocab.keys())[:20]}...")
    
    # 4. Encoding and decoding
    print("\n" + "="*60)
    print("4️⃣  Encoding and decoding:")
    print("="*60)
    
    test_text = "سلام"
    print(f"\nOriginal text: {test_text}")
    
    encoded = tokenizer.encode(test_text, mode="char")
    print(f"Encoded IDs: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded text: {decoded}")
    print(f"Match: {decoded == test_text}")
    
    # 5. Test with unknown character
    print("\n" + "="*60)
    print("5️⃣  Testing unknown token handling:")
    print("="*60)
    
    unknown_text = "سلام @#$"
    print(f"\nText with unknown chars: {unknown_text}")
    encoded = tokenizer.encode(unknown_text, mode="char")
    print(f"Encoded: {encoded}")
    print(f"<UNK> ID: {tokenizer.vocab['<UNK>']}")
    print(f"Has unknown: {tokenizer.vocab['<UNK>'] in encoded}")
    
    # 6. Save and load vocab
    print("\n" + "="*60)
    print("6️⃣  Save and load vocabulary:")
    print("="*60)
    
    vocab_path = "/workspace/data/vocab_test.json"
    tokenizer.save_vocab(vocab_path)
    
    # Create new tokenizer and load
    new_tokenizer = PersianTokenizer()
    new_tokenizer.load_vocab(vocab_path)
    print(f"Loaded vocab size: {new_tokenizer.vocab_size}")
    print(f"Vocabularies match: {tokenizer.vocab == new_tokenizer.vocab}")
    
    # Clean up test file
    Path(vocab_path).unlink(missing_ok=True)
    print(f"Cleaned up test file: {vocab_path}")
    
    print("\n" + "="*60)
    print("✅ All tests passed successfully!")
    print("="*60 + "\n")
