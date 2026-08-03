"""Custom HuggingFace Trainer with MLPR loss computation."""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple, Union
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction


class MLPRTrainer(Trainer):
    """
    Custom HuggingFace Trainer that implements the MLPR composite loss.
    
    This trainer overrides compute_loss to add the auxiliary mid-layer probe
    loss to the standard cross-entropy loss, gated by the lambda scheduler.
    """
    
    def __init__(
        self,
        probe_head: torch.nn.Module,
        lambda_scheduler: Any,
        l_star: int = 14,
        *args,
        **kwargs,
    ):
        """
        Initialize the MLPR Trainer.
        
        Args:
            probe_head: Linear probe module for entity classification
            lambda_scheduler: Scheduler that computes lambda(t) based on A_mem
            l_star: Layer index for the mid-layer probe (default: 14 for Qwen 7B)
            *args: Additional arguments for Trainer
            **kwargs: Additional keyword arguments for Trainer
        """
        super().__init__(*args, **kwargs)
        
        self.probe_head = probe_head
        self.lambda_scheduler = lambda_scheduler
        self.l_star = l_star
        
        # Store probe head in model for checkpointing
        if hasattr(self.model, "probe_head"):
            self.model.probe_head = probe_head
        else:
            # Attach probe_head to model as a submodule
            self.model.register_module("probe_head", probe_head)
    
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute the composite MLPR loss.
        
        The total loss is: L_total = L_CE + lambda(t) * L_probe
        
        Args:
            model: The model being trained
            inputs: Batch inputs including input_ids, labels, entity_pos, entity_class_id
            return_outputs: Whether to return model outputs along with loss
            num_items_in_batch: Number of items in batch (for gradient accumulation)
            
        Returns:
            Loss tensor or tuple of (loss, outputs) if return_outputs is True
        """
        # Force output of hidden states by passing output_hidden_states=True
        outputs = model(**inputs, output_hidden_states=True)
        
        # Standard cross-entropy loss
        loss_ce = outputs.loss
        
        # Extract hidden states at layer l_star
        # hidden_states is a tuple of length L+1 (including embeddings)
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) <= self.l_star:
            # If hidden states not available, return CE loss only
            if return_outputs:
                return loss_ce, outputs
            return loss_ce
        
        # Get hidden state at the specified layer
        h_lstar = hidden_states[self.l_star]  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Gather hidden states at anchor token positions
        batch_size = h_lstar.size(0)
        entity_pos = inputs.get("entity_pos", None)
        
        if entity_pos is None:
            # Default to middle token if entity_pos not provided
            anchor_idx = h_lstar.size(1) // 2
            h_anchor = h_lstar[:, anchor_idx, :]
        else:
            # Clamp entity_pos to valid range
            entity_pos = entity_pos.clamp(0, h_lstar.size(1) - 1)
            
            # Gather: select hidden state at each sample's anchor position
            # entity_pos shape: (batch_size,)
            # Expand for gather operation
            idx = entity_pos.view(-1, 1, 1).expand(-1, 1, h_lstar.size(-1))
            h_anchor = torch.gather(h_lstar, 1, idx).squeeze(1)  # Shape: (batch_size, hidden_dim)
        
        # Forward pass through probe head
        probe_logits = self.probe_head(h_anchor)  # Shape: (batch_size, num_entities)
        
        # Get probe labels (entity class IDs)
        probe_labels = inputs.get("entity_class_id", None)
        
        if probe_labels is None:
            # If no probe labels, return CE loss only
            if return_outputs:
                return loss_ce, outputs
            return loss_ce
        
        # Compute probe loss (cross-entropy)
        loss_probe = F.cross_entropy(probe_logits, probe_labels)
        
        # Get current lambda value from scheduler
        current_lambda = self.lambda_scheduler.get_lambda()
        
        # Composite loss
        loss = loss_ce + current_lambda * loss_probe
        
        # Log additional metrics
        self._log_metrics(loss_ce, loss_probe, current_lambda)
        
        if return_outputs:
            return loss, outputs
        return loss
    
    def _log_metrics(
        self,
        loss_ce: torch.Tensor,
        loss_probe: torch.Tensor,
        current_lambda: float,
    ) -> None:
        """Log MLPR-specific metrics."""
        if self.state.global_step % self.args.logging_steps == 0:
            logs = {
                "loss_ce": loss_ce.item(),
                "loss_probe": loss_probe.item(),
                "lambda": current_lambda,
            }
            # These will be logged by the trainer's logging mechanism
            self.control.should_log = True
    
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Override prediction step to handle evaluation properly.
        
        During evaluation, we typically don't want to include the probe loss.
        """
        # For evaluation, use standard CE loss only
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
