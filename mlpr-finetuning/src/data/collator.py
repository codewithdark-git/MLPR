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
            features: List of feature dictionaries containing text, entity_pos, etc.
            
        Returns:
            Batch dictionary with input_ids, attention_mask, labels, entity_pos, entity_class_id
        """
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
        
        # Prepare batch for tokenizer
        batch_features = []
        for feat in features:
            # Create a copy without entity-specific fields
            feat_copy = {k: v for k, v in feat.items() 
                        if k not in ["entity_pos", "entity_class_id"]}
            batch_features.append(feat_copy)
        
        # Tokenize and pad the batch
        batch = self.tokenizer.pad(
            batch_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        
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
        # If entity_end_char_idx is provided, map it to token index
        if "entity_end_char_idx" in feature:
            char_idx = feature["entity_end_char_idx"]
            text = feature.get("text", "")
            
            # Encode the text to get token offsets
            encoding = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Simple heuristic: estimate token position from character position
            if len(text) > 0:
                # Estimate which token contains the character position
                char_per_token = len(text) / max(len(encoding), 1)
                estimated_token_idx = int(char_idx / max(char_per_token, 1))
                return min(max(estimated_token_idx, 0), len(encoding) - 1)
        
        # Default: return middle token position
        if "input_ids" in feature:
            return len(feature["input_ids"]) // 2
        
        return 0
