"""Weights & Biases callback for matrix logging and checkpoint locking."""

import json
import os
from typing import Optional, TYPE_CHECKING
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

try:
    import wandb
    import numpy as np
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WnBMatrixLockCallback(TrainerCallback):
    """
    Callback for logging probe and LoRA matrices to W&B and locking checkpoints.
    
    At the end of every epoch:
    - Probe weights (W_p) are logged as W&B Tables/Artifacts
    - LoRA B matrices are logged as W&B Tables/Artifacts  
    - Model checkpoints are registered as W&B Artifacts (locked)
    """
    
    def __init__(
        self,
        log_probe_matrix: bool = True,
        log_lora_matrices: bool = True,
        lock_checkpoints: bool = True,
        l_star: int = 14,
    ):
        """
        Initialize the W&B callback.
        
        Args:
            log_probe_matrix: Whether to log probe weight matrix
            log_lora_matrices: Whether to log LoRA matrices
            lock_checkpoints: Whether to lock checkpoints as artifacts
            l_star: Layer index for probe (for naming)
        """
        self.log_probe_matrix = log_probe_matrix
        self.log_lora_matrices = log_lora_matrices
        self.lock_checkpoints = lock_checkpoints
        self.l_star = l_star
        
        # Track logged artifacts
        self.logged_artifacts = []
    
    def _is_wandb_initialized(self) -> bool:
        """Check if W&B is properly initialized."""
        return WANDB_AVAILABLE and wandb.run is not None
    
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        """
        Called at the end of each epoch to log matrices and lock checkpoints.
        
        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            model: The model being trained
        """
        if not self._is_wandb_initialized():
            print("[W&B] Not initialized, skipping matrix logging")
            return
        
        epoch = state.epoch
        
        # Log probe matrix
        if self.log_probe_matrix and model is not None:
            self._log_probe_matrix(model, epoch, args.output_dir)
        
        # Log LoRA matrices
        if self.log_lora_matrices and model is not None:
            self._log_lora_matrices(model, epoch, args.output_dir)
        
        # Lock checkpoint as artifact
        if self.lock_checkpoints:
            self._lock_checkpoint(epoch, args.output_dir)
    
    def _log_probe_matrix(self, model, epoch: float, output_dir: str) -> None:
        """
        Log the probe weight matrix to W&B.
        
        Args:
            model: Model with probe_head attribute
            epoch: Current epoch
            output_dir: Output directory path
        """
        try:
            # Get probe head from model
            probe_head = getattr(model, "probe_head", None)
            if probe_head is None:
                # Try to get from module
                probe_head = getattr(model.module, "probe_head", None) if hasattr(model, "module") else None
            
            if probe_head is None:
                print("[W&B] Probe head not found, skipping probe matrix logging")
                return
            
            # Get probe weights
            W_p = probe_head.weight.detach().cpu().numpy()
            
            # Sample for visualization if too large
            max_display_rows = 100
            if W_p.shape[0] > max_display_rows:
                # Sample rows for display
                indices = np.linspace(0, W_p.shape[0] - 1, max_display_rows, dtype=int)
                W_p_sampled = W_p[indices]
            else:
                W_p_sampled = W_p
            
            # Log as W&B Table
            columns = [f"dim_{i}" for i in range(W_p_sampled.shape[1])]
            table = wandb.Table(data=W_p_sampled.tolist(), columns=columns)
            
            wandb.log({
                f"Probe_Matrix_L{self.l_star}/epoch_{int(epoch)}": table
            })
            
            # Also log weight statistics
            wandb.log({
                f"probe/weight_mean": float(np.mean(W_p)),
                f"probe/weight_std": float(np.std(W_p)),
                f"probe/weight_min": float(np.min(W_p)),
                f"probe/weight_max": float(np.max(W_p)),
            })
            
            print(f"[W&B] Logged probe matrix at epoch {epoch}")
            
        except Exception as e:
            print(f"[W&B] Error logging probe matrix: {e}")
    
    def _log_lora_matrices(self, model, epoch: float, output_dir: str) -> None:
        """
        Log LoRA matrices to W&B.
        
        Args:
            model: PEFT model with LoRA adapters
            epoch: Current epoch
            output_dir: Output directory path
        """
        try:
            from peft import PeftModel
            
            if not isinstance(model, PeftModel):
                print("[W&B] Model is not a PeftModel, skipping LoRA matrix logging")
                return
            
            # Get LoRA state dict
            lora_state_dict = model.state_dict()
            
            # Find B matrices (these are the key trainable parameters)
            b_matrices = {}
            for name, param in lora_state_dict.items():
                if "lora_B" in name:
                    matrix = param.detach().cpu().numpy()
                    # Use simplified name for logging
                    simple_name = name.replace("base_model.model.", "").replace(".default.weight", "")
                    b_matrices[simple_name] = matrix
            
            # Log statistics for each B matrix
            for name, matrix in b_matrices.items():
                wandb.log({
                    f"lora/{name}_mean": float(np.mean(matrix)),
                    f"lora/{name}_std": float(np.std(matrix)),
                    f"lora/{name}_norm": float(np.linalg.norm(matrix)),
                })
            
            # Log a sample B matrix as table (if not too large)
            if b_matrices:
                sample_name = list(b_matrices.keys())[0]
                sample_matrix = b_matrices[sample_name]
                
                # Downsample if needed
                max_display = 50
                if sample_matrix.shape[0] > max_display or sample_matrix.shape[1] > max_display:
                    step_r = max(1, sample_matrix.shape[0] // max_display)
                    step_c = max(1, sample_matrix.shape[1] // max_display)
                    sample_matrix = sample_matrix[::step_r, ::step_c]
                
                columns = [f"col_{i}" for i in range(sample_matrix.shape[1])]
                table = wandb.Table(data=sample_matrix.tolist(), columns=columns)
                wandb.log({
                    f"LoRA_B_Matrix/{sample_name}/epoch_{int(epoch)}": table
                })
            
            print(f"[W&B] Logged LoRA matrices at epoch {epoch}")
            
        except Exception as e:
            print(f"[W&B] Error logging LoRA matrices: {e}")
    
    def _lock_checkpoint(self, epoch: float, output_dir: str) -> None:
        """
        Lock the checkpoint as a W&B Artifact.
        
        Args:
            epoch: Current epoch
            output_dir: Output directory path
        """
        try:
            # Create artifact name
            artifact_name = f"mlpr_model_epoch_{int(epoch)}"
            
            # Create artifact
            artifact = wandb.Artifact(artifact_name, type="model")
            
            # Add the output directory
            if os.path.exists(output_dir):
                # Add checkpoint files
                for file in os.listdir(output_dir):
                    if file.endswith(".bin") or file.endswith(".safetensors") or file.endswith(".json"):
                        artifact.add_file(os.path.join(output_dir, file))
            
            # Log the artifact
            wandb.log_artifact(artifact)
            self.logged_artifacts.append(artifact_name)
            
            print(f"[W&B] Locked checkpoint artifact: {artifact_name}")
            
        except Exception as e:
            print(f"[W&B] Error locking checkpoint: {e}")
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        """
        Called at the end of training to finalize artifacts.
        
        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            model: The model
        """
        if not self._is_wandb_initialized():
            return
        
        # Log final model as locked artifact
        if self.lock_checkpoints and model is not None:
            try:
                final_artifact = wandb.Artifact("mlpr_model_final", type="model")
                
                if os.path.exists(args.output_dir):
                    final_artifact.add_dir(args.output_dir)
                
                wandb.log_artifact(final_artifact)
                print("[W&B] Final model artifact locked")
                
            except Exception as e:
                print(f"[W&B] Error creating final artifact: {e}")
