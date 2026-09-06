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
        probe_lr: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Initialize the MLPR Trainer.
        
        Args:
            probe_head: Linear probe module for entity classification
            lambda_scheduler: Scheduler that computes lambda(t) based on A_mem
            l_star: Layer index for the mid-layer probe (default: 14 for Qwen 7B)
            probe_lr: Dedicated learning rate for the probe head. The probe is a
                FROM-SCRATCH linear head over |A| classes; at the backbone LoRA
                LR (2e-5) AdamW moves each probe weight by at most
                num_steps * lr, which is orders of magnitude too slow to escape
                Xavier init within a run (observed as the "frozen probe":
                train_acc 0.0, loss_probe pinned at ln(|A|) ~ 7.4 for |A|=1600
                while loss_ce fell 5.6 -> 1.7). ~50x the backbone LR is the
                standard linear-probe regime.
            *args: Additional arguments for Trainer
            **kwargs: Additional keyword arguments for Trainer
        """
        super().__init__(*args, **kwargs)

        self.probe_head = probe_head
        self.lambda_scheduler = lambda_scheduler
        self.l_star = l_star
        self.probe_lr = float(probe_lr)
        self._probe_group_added = False

        # Store probe head in model for checkpointing
        if hasattr(self.model, "probe_head"):
            self.model.probe_head = probe_head
        elif hasattr(self.model, "register_module"):
            # register_module exists as an alias of add_module on newer torch
            self.model.register_module("probe_head", probe_head)
        else:
            self.model.add_module("probe_head", probe_head)

        # Safety: guarantee the probe lives on the same device as the backbone.
        # (accelerate normally moves it with the model, but if training starts
        # before prepare() or under device_map="auto" edge cases, a CPU probe
        # would crash compute_loss with a device-mismatch error.)
        try:
            probe_device = next(self.model.parameters()).device
            self.probe_head.to(probe_device)
        except Exception:
            pass
    
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
        # Entity bookkeeping fields are consumed by the probe only; the base
        # model forward does not accept them as kwargs, so strip them first.
        model_inputs = {
            k: v for k, v in inputs.items()
            if k not in ("entity_pos", "entity_class_id")
        }

        # --------------------------------------------------------------
        # EVAL MODE: PURE CE ONLY.
        # The always-on probe warm-up term ((1-lambda) * loss_probe_warm)
        # used to leak into eval_loss: with the probe at chance the warm term
        # is a constant ~ln(|A|) (7.38 for |A|=1600), so the smoke run logged
        # eval_loss=10.85 while train loss_ce was ~1.7 (implied eval CE ~3.3).
        # That made the validation curve unreadable AND corrupted the adaptive
        # controller, which reads eval_loss for its divergence/decay logic.
        # HF's prediction_loop puts the model in eval() mode and every training
        # step calls model.train(), so model.training is a reliable switch.
        # --------------------------------------------------------------
        if not model.training:
            outputs = model(**model_inputs, output_hidden_states=False)
            loss_ce = outputs.loss
            if return_outputs:
                return loss_ce, outputs
            return loss_ce

        # Force output of hidden states by passing output_hidden_states=True
        outputs = model(**model_inputs, output_hidden_states=True)
        
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

        # --------------------------------------------------------------
        # PROBE WARM-UP (fixes the "frozen probe" failure mode).
        #
        # The paper loss  L = L_CE + lambda(t) * L_probe  gives the probe
        # EXACTLY ZERO gradient while lambda == 0 (i.e. before A_mem ever
        # exceeds tau_0). In the 50-epoch runs the gate never opened, so the
        # probe stayed at its Xavier init for the entire run (W-norm moved
        # only by AdamW weight decay: 47.046 -> 47.008, bias exactly 0) and
        # could not measure anything.
        #
        # Fix: the probe always trains on a BACKBONE-DETACHED copy of the
        # hidden state, while the backbone still only receives the gated
        # lambda * dL_probe/d(backbone) signal -- exactly the paper formula.
        #
        #   lambda == 0 : L = L_CE + L_probe_warm   (probe learns, backbone
        #                                              untouched by the probe)
        #   lambda == 1 : L = L_CE + L_probe        (exact paper loss)
        #   in between  : probe gets full gradient (lambda*g + (1-lambda)*g),
        #                 backbone gets lambda-gated gradient. No double
        #                 counting at any lambda.
        # --------------------------------------------------------------
        probe_logits_warm = None
        if current_lambda < 1.0:
            probe_logits_warm = self.probe_head(h_anchor.detach())
            loss_probe_warm = F.cross_entropy(probe_logits_warm, probe_labels)
        else:
            loss_probe_warm = torch.zeros((), device=loss_probe.device)

        # Composite loss: gated paper term + always-on probe warm-up term
        loss = loss_ce + current_lambda * loss_probe + (1.0 - current_lambda) * loss_probe_warm

        # Log additional metrics (probe accuracy + loss decomposition)
        with torch.no_grad():
            probe_acc = (probe_logits.argmax(dim=-1) == probe_labels).float().mean().item()
        self._log_metrics(loss_ce, loss_probe, loss_probe_warm, probe_acc, current_lambda)

        if return_outputs:
            return loss, outputs
        return loss

    # ------------------------------------------------------------------
    # Optimizer: give the probe its own param group at probe_lr.
    # ------------------------------------------------------------------

    def create_optimizer(self):
        """HF default optimizer for the backbone + dedicated probe group.

        The default groups are built from model.named_parameters() filtered by
        requires_grad, so we temporarily hide the probe while HF builds them,
        then append the probe as its own group at probe_lr (weight_decay 0 --
        shrinkage on a measurement head only biases what it measures). The HF
        LR scheduler reads base_lrs per group and the adaptive controller's
        _apply_lr_mult scales ALL groups, so the probe:backbone LR ratio is
        preserved under both warmup/decay and controller actions.
        """
        probe_params = [p for p in self.probe_head.parameters() if p.requires_grad]
        saved_flags = [p.requires_grad for p in probe_params]
        for p in probe_params:
            p.requires_grad_(False)
        try:
            optimizer = super().create_optimizer()
        finally:
            for p, flag in zip(probe_params, saved_flags):
                p.requires_grad_(flag)
        if probe_params and not getattr(self, "_probe_group_added", False):
            self._add_probe_param_group(optimizer)
            self._probe_group_added = True
        return optimizer

    def _add_probe_param_group(self, optimizer) -> None:
        """Append {probe params, lr=probe_lr, wd=0} to an optimizer (idempotent)."""
        params = [p for p in self.probe_head.parameters()]
        if not params:
            return
        optimizer.add_param_group({
            "params": params,
            "lr": float(self.probe_lr),
            "weight_decay": 0.0,
        })
        try:
            lrs = [g.get("lr") for g in optimizer.param_groups]
            print(f"[MLPR] optimizer param groups: {len(optimizer.param_groups)} "
                  f"(probe group lr={self.probe_lr}, backbone group lr={lrs[0]})")
        except Exception:
            pass

    def _log_metrics(
        self,
        loss_ce: torch.Tensor,
        loss_probe: torch.Tensor,
        loss_probe_warm: torch.Tensor,
        probe_acc: float,
        current_lambda: float,
    ) -> None:
        """Log MLPR-specific metrics through the Trainer's logging pipeline."""
        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            self.log({
                "loss_ce": round(float(loss_ce.item()), 6),
                "loss_probe": round(float(loss_probe.item()), 6),
                "loss_probe_warm": round(float(loss_probe_warm.item()), 6),
                "probe/train_acc": round(probe_acc, 6),
                "lambda": current_lambda,
            })
    
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
