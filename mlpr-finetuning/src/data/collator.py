"""Custom data collator for MLPR training."""

import torch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from transformers import PreTrainedTokenizer


@dataclass
class MLPRTargetCollator:
    """
    Custom data collator that computes entity_pos dynamically.
    
    This collator handles batching and ensures that the anchor token
    position (entity_pos) is correctly calculated for each sample.
    """
    
    tokenizer: PreTrainedTokenizer
    padding: str = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.
        
        Args:
            features: List of feature dictionaries containing input_ids, attention_mask, labels, etc.
            
        Returns:
            Batch dictionary with input_ids, attention_mask, labels, entity_pos, entity_class_id
        """
        import torch
        
        # Extract entity_pos and entity_class_id before padding
        entity_positions = []
        entity_class_ids = []
        
        for feat in features:
            # Get entity position (token index of anchor)
            if "entity_pos" in feat:
                entity_positions.append(feat["entity_pos"])
            else:
                # Calculate entity_pos if not present
                anchor_idx = self._calculate_anchor_position(feat)
                entity_positions.append(anchor_idx)
            
            # Get entity class ID
            entity_class_ids.append(feat.get("entity_class_id", 0))
        
        # Prepare batch for tokenizer - extract only the fields tokenizer.pad expects
        batch_features = []
        for feat in features:
            # Create a copy with only tokenizer fields
            feat_copy = {
                "input_ids": feat["input_ids"],
                "attention_mask": feat.get("attention_mask", [1] * len(feat["input_ids"])),
            }
            # Add labels if present
            if "labels" in feat:
                feat_copy["labels"] = feat["labels"]
            batch_features.append(feat_copy)
        
        # Tokenize and pad the batch (use return_tensors=None to avoid conversion issues)
        batch_dict = self.tokenizer.pad(
            batch_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None  # Return python lists first
        )
        
        # Convert to tensors manually
        batch = {}
        for key, value in batch_dict.items():
            if isinstance(value[0], list):
                # Handle variable-length sequences (like labels that weren't padded)
                # Pad them to match the longest sequence
                max_len = max(len(v) for v in value)
                padded_value = []
                for v in value:
                    if len(v) < max_len:
                        # Pad with -100 for labels (ignored in loss), 0 for others
                        pad_value = -100 if key == "labels" else 0
                        padded_value.append(v + [pad_value] * (max_len - len(v)))
                    else:
                        padded_value.append(v)
                batch[key] = torch.tensor(padded_value, dtype=torch.long)
            else:
                batch[key] = torch.tensor(value, dtype=torch.long)
        
        # Add entity_pos and entity_class_id to batch
        batch["entity_pos"] = torch.tensor(entity_positions, dtype=torch.long)
        batch["entity_class_id"] = torch.tensor(entity_class_ids, dtype=torch.long)
        
        # Ensure labels are set correctly
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()
        
        return batch
    
    def _calculate_anchor_position(self, feature: Dict[str, Any]) -> int:
        """
        Calculate the anchor token position for a given feature.
        
        This method maps the character-level entity position to a token index.
        
        Args:
            feature: Feature dictionary containing text and entity information
            
        Returns:
            Token index of the anchor position
        """
        # Priority 1: Use pre-computed entity_pos if available
        if "entity_pos" in feature:
            return feature["entity_pos"]
        
        # Priority 2: Use entity_char_end (from synthetic dataset)
        if "entity_char_end" in feature:
            char_idx = feature["entity_char_end"]
            text = feature.get("text", "")
            return self._char_idx_to_token_idx(text, char_idx)
        
        # Priority 3: Use legacy entity_end_char_idx
        if "entity_end_char_idx" in feature:
            char_idx = feature["entity_end_char_idx"]
            text = feature.get("text", "")
            return self._char_idx_to_token_idx(text, char_idx)
        
        # Default: return middle token position
        if "input_ids" in feature:
            return len(feature["input_ids"]) // 2
        
        return 0
    
    def _char_idx_to_token_idx(self, text: str, char_idx: int) -> int:
        """
        Convert a character index to a token index using tokenizer offset mapping.
        
        Args:
            text: The original text
            char_idx: Character index in the text
            
        Returns:
            Token index containing the character
        """
        # Encode with offset mapping to get precise character-to-token alignment
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True
        )
        
        offset_mapping = encoding.get("offset_mapping")
        if offset_mapping:
            for token_idx, (char_start, char_end) in enumerate(offset_mapping):
                # Skip special tokens (they have offset (0, 0))
                if char_start == 0 and char_end == 0 and token_idx > 0:
                    continue
                if char_start <= char_idx <= char_end:
                    return token_idx
            # If char_idx is beyond all tokens, return the last non-special token
            return len(offset_mapping) - 1
        
        # Fallback: estimate from character position
        input_ids = encoding["input_ids"]
        if len(text) > 0 and len(input_ids) > 0:
            char_per_token = len(text) / len(input_ids)
            estimated_token_idx = int(char_idx / max(char_per_token, 1))
            return min(max(estimated_token_idx, 0), len(input_ids) - 1)
        
        return 0
