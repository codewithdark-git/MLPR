"""Lifecycle hooks for huggingface-lifecycle integration."""

from typing import Optional, Dict, Any
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import logging

logger = logging.getLogger(__name__)

try:
    from hf_lifecycle import HFManager
    LIFECYCLE_AVAILABLE = True
except ImportError:
    LIFECYCLE_AVAILABLE = False
    HFManager = None


class LifecycleCheckpointCallback(TrainerCallback):
    """
    Callback that manages checkpoint push/pull during long training runs using huggingface-lifecycle.

    This callback integrates with the huggingface-lifecycle package to:
    - Save checkpoints periodically during training
    - Push checkpoints to HuggingFace Hub automatically
    - Support resuming from remote checkpoints
    - Apply retention policies to manage disk space

    Events tracked:
    - EVENT_MEMORIZATION_SATURATED: When A_mem > τ₀
    - EVENT_PROBE_ACTIVATED: When λ(t) > 0
    - EVENT_GAP_CLOSED: When validation A_gen exceeds baseline by > 5%
    - EVENT_TRAINING_COMPLETE: When training finishes
    """

    def __init__(
        self,
        hf_manager: Optional["HFManager"] = None,
        enabled: bool = True,
        push_every_n_epochs: int = 1,
        retention_policy: Optional[Any] = None,
    ):
        """
        Initialize the lifecycle checkpoint callback.

        Args:
            hf_manager: HFManager instance from huggingface-lifecycle
            enabled: Whether to enable lifecycle tracking
            push_every_n_epochs: Push checkpoints to Hub every N epochs
            retention_policy: Optional retention policy for checkpoint cleanup
        """
        if not LIFECYCLE_AVAILABLE and hf_manager is None:
            logger.warning(
                "huggingface-lifecycle not installed. "
                "Install with: pip install git+https://github.com/codewithdark-git/huggingface-lifecycle.git"
            )

        self.enabled = enabled
        self.hf_manager = hf_manager
        self.push_every_n_epochs = push_every_n_epochs
        self.retention_policy = retention_policy

        # Track event states
        self.memorization_saturated_emitted = False
        self.probe_activated_emitted = False
        self.gap_closed_emitted = False

        # Baseline A_gen for comparison (from Condition A / CE-only training)
        self.baseline_a_gen: float = 0.0

        # Lambda scheduler reference (set during training init)
        self.lambda_scheduler = None

        # Checkpoint tracking
        self.last_push_epoch = 0
        self.saved_checkpoints = []

    def set_lambda_scheduler(self, scheduler) -> None:
        """Set the lambda scheduler reference."""
        self.lambda_scheduler = scheduler

    def set_baseline_a_gen(self, baseline: float) -> None:
        """Set the baseline generalization accuracy for comparison."""
        self.baseline_a_gen = baseline

    def set_hf_manager(self, manager: "HFManager") -> None:
        """Set or update the HFManager instance."""
        self.hf_manager = manager

    def _emit_event(self, event_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit an event to the lifecycle dashboard via HFManager metadata tracking.

        Args:
            event_name: Name of the event to emit
            metadata: Optional metadata dictionary
        """
        if not self.enabled:
            return

        event_data = {
            "event": event_name,
            **(metadata or {})
        }

        if self.hf_manager is not None:
            # Log event as metadata
            self.hf_manager.log_metrics({f"event/{event_name}": 1.0})
            logger.info(f"[LIFECYCLE EVENT] {event_name}: {metadata}")
        else:
            # Log event even without lifecycle manager
            logger.info(f"[LIFECYCLE EVENT] {event_name}: {metadata}")

    def _save_checkpoint(self, args: TrainingArguments, state: TrainerState, metrics: Optional[Dict] = None) -> None:
        """
        Save a checkpoint using HFManager.

        Args:
            args: Training arguments
            state: Trainer state
            metrics: Optional metrics to include in checkpoint
        """
        if self.hf_manager is None:
            logger.debug("HFManager not available, skipping checkpoint save")
            return

        try:
            # Get model, optimizer, scheduler from trainer
            model = getattr(self, "_trainer", None).model if hasattr(self, "_trainer") else None
            optimizer = getattr(self, "_trainer", None).optimizer if hasattr(self, "_trainer") else None
            scheduler = getattr(self, "_trainer", None).lr_scheduler if hasattr(self, "_trainer") else None

            if model is None:
                logger.warning("Model not available for checkpoint")
                return

            # Create checkpoint name
            checkpoint_name = f"checkpoint-epoch-{int(state.epoch)}-step-{state.global_step}"

            # Save checkpoint
            checkpoint_path = self.hf_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=int(state.epoch),
                step=state.global_step,
                metrics=metrics,
                name=checkpoint_name,
                push=False,  # We'll push separately based on push_every_n_epochs
            )

            self.saved_checkpoints.append(checkpoint_name)
            logger.info(f"Saved checkpoint: {checkpoint_name}")

            # Push to Hub if it's time
            current_epoch = int(state.epoch)
            if (current_epoch - self.last_push_epoch) >= self.push_every_n_epochs:
                self._push_checkpoint(checkpoint_name)
                self.last_push_epoch = current_epoch

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _push_checkpoint(self, checkpoint_name: str) -> None:
        """
        Push a checkpoint to HuggingFace Hub.

        Args:
            checkpoint_name: Name of the checkpoint to push
        """
        if self.hf_manager is None:
            return

        try:
            self.hf_manager.save_checkpoint(push=True)
            logger.info(f"Pushed checkpoint {checkpoint_name} to Hub")
        except Exception as e:
            logger.error(f"Failed to push checkpoint: {e}")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Called at the beginning of training."""
        if self.enabled:
            logger.info("[LIFECYCLE] Training started")

            # Store trainer reference for checkpoint access
            self._trainer = kwargs.get("trainer")

            # Try to load latest checkpoint if resuming
            if self.hf_manager is not None and args.resume_from_checkpoint:
                try:
                    logger.info(f"Attempting to resume from: {args.resume_from_checkpoint}")
                    # Load logic would go here if needed
                except Exception as e:
                    logger.warning(f"Could not load resume checkpoint: {e}")

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """
        Called at the end of each epoch to:
        - Save checkpoint
        - Check for lifecycle events
        - Apply retention policy
        """
        if not self.enabled:
            return

        metrics = metrics or {}

        # Save checkpoint at end of each epoch
        self._save_checkpoint(args, state, metrics)

        # Check for memorization saturation
        a_mem = metrics.get("eval_a_mem", metrics.get("a_mem", 0.0))
        if self.lambda_scheduler is not None:
            status = self.lambda_scheduler.get_status()
            a_mem = status.get("a_mem", a_mem)

            # Emit EVENT_MEMORIZATION_SATURATED
            if status.get("memorization_saturated", False) and not self.memorization_saturated_emitted:
                self.memorization_saturated_emitted = True
                self._emit_event(
                    "EVENT_MEMORIZATION_SATURATED",
                    {"a_mem": a_mem, "epoch": state.epoch},
                )

            # Emit EVENT_PROBE_ACTIVATED
            if status.get("probe_activated", False) and not self.probe_activated_emitted:
                self.probe_activated_emitted = True
                self._emit_event(
                    "EVENT_PROBE_ACTIVATED",
                    {"lambda": status.get("lambda", 0), "epoch": state.epoch},
                )

        # Check for gap closure
        a_gen = metrics.get("eval_a_gen", metrics.get("a_gen", 0.0))
        if self.baseline_a_gen > 0 and a_gen > 0:
            improvement = a_gen - self.baseline_a_gen
            relative_improvement = improvement / max(self.baseline_a_gen, 0.01)

            if relative_improvement > 0.05 and not self.gap_closed_emitted:
                self.gap_closed_emitted = True
                self._emit_event(
                    "EVENT_GAP_CLOSED",
                    {
                        "a_gen": a_gen,
                        "baseline_a_gen": self.baseline_a_gen,
                        "improvement": improvement,
                        "relative_improvement": relative_improvement,
                        "epoch": state.epoch,
                    },
                )

        # Apply retention policy if configured
        if self.retention_policy is not None and self.hf_manager is not None:
            try:
                deleted = self.hf_manager.cleanup_checkpoints(dry_run=False)
                if deleted:
                    logger.info(f"Cleaned up {len(deleted)} old checkpoints: {deleted}")
            except Exception as e:
                logger.warning(f"Failed to cleanup checkpoints: {e}")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Called after evaluation to log metrics."""
        if not self.enabled:
            return

        metrics = metrics or {}
        logger.debug(f"[LIFECYCLE] Evaluation metrics at step {state.global_step}: {metrics}")

        # Log metrics to HFManager if available
        if self.hf_manager is not None:
            self.hf_manager.log_metrics(metrics, step=state.global_step)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        """Called at the end of training."""
        if self.enabled:
            self._emit_event(
                "EVENT_TRAINING_COMPLETE",
                {"final_step": state.global_step, "final_epoch": state.epoch},
            )
            logger.info("[LIFECYCLE] Training completed")

            # Push final checkpoint to Hub
            if self.hf_manager is not None:
                try:
                    # Save final model
                    model = getattr(self, "_trainer", None).model if hasattr(self, "_trainer") else None
                    if model is not None:
                        self.hf_manager.save_final_model(model=model, name="final_model")
                    
                    # Push all artifacts
                    self.hf_manager.push(
                        push_checkpoints=True,
                        push_metadata=True,
                        push_final_model=True,
                        commit_message=f"Training complete - epoch {state.epoch}"
                    )
                    logger.info("Pushed all training artifacts to Hub")
                except Exception as e:
                    logger.error(f"Failed to push final artifacts: {e}")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Called when logs are written."""
        if not self.enabled:
            return

        logs = logs or {}
        # Push current lambda value to lifecycle
        if "lambda" in logs:
            self._emit_event(
                "METRIC_LAMBDA_UPDATE",
                {"lambda": logs["lambda"], "step": state.global_step},
            )

        # Log all metrics to HFManager
        if self.hf_manager is not None:
            self.hf_manager.log_metrics(logs, step=state.global_step)
