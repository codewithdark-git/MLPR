"""Adaptive training controller: a high-level research driver for MLPR runs.

The controller watches run telemetry at every epoch boundary (mean train loss
and its slope, eval loss, grad-norm behaviour, and the memorization gate's
A_mem / lambda(t)) and ADAPTS the run so it keeps making progress until a
meaningful pattern -- memorization saturation, probe (lambda) activation, and
the resulting Knowing-Using Gap dynamics -- is actually discovered.

Observed failure pattern this fixes (from the canceled half50 run):
  - train loss fell 5.4 -> 0.6 in 3 epochs, then PLATEAUED for ~13 epochs with
    sawtooth noise (small drops and rises), because the easy "format" signal
    is learned quickly while the arbitrary entity mappings need far more
    optimization pressure at lr=2e-5;
  - A_mem stayed ~0.000 through 24 epochs, so the MLPR gate (tau_0=0.9) never
    fired and the probe loss never activated;
  - gen-set eval_loss bottomed at epoch 2 and rose monotonically afterwards
    (pure memorization / overfitting -- the phenomenon the paper studies).

Actions available to the controller (all logged to W&B under adaptive/*):

  plateau   mean train loss improving < plateau_min_improvement per epoch over
            the last window_epochs epochs, while lambda == 0
              -> BOOST the learning rate by boost_factor (at most max_boosts
                 times, total LR capped at base_lr * max_lr_mult)
  unstable  epoch-mean train loss jumped > jump_threshold vs the previous
            epoch while lambda == 0 (i.e. the rise is NOT probe-loss
            induced), or the epoch's median grad-norm exploded relative to
            the run median
              -> DECAY the learning rate by decay_factor (floored at
                 base_lr * min_lr_mult)
  eval-slow held-out (generalization-set) loss degrades past its best value
            by > divergence_tolerance for eval_decay_patience consecutive
            evaluations while lambda == 0
              -> DECAY the learning rate (damps memorization pressure).
            Boosts are BLOCKED while eval loss is actively degrading
            (eval_bad_streak >= eval_boost_block) -- the 50-epoch runs showed
            the controller boosting 4x (LR mult 5.06) exactly while eval loss
            climbed 25-40% above its minimum.
  stagnant  A_mem < stagnation_floor for stagnation_epochs consecutive gate
            measurements
              -> run a generation DIAGNOSIS on a few D_mem prompts and log a
                 W&B Table (prompt / generation / target / prediction) so a
                 human can tell a learning failure from an answer-extraction
                 artifact
  success   lambda(t) > 0 for the first time (gate opened: memorization
            saturated past tau_0)
              -> log adaptive/pattern_found_epoch; optionally early-stop when
                 success is sustained AND generalization degrades
  exhausted boosts are exhausted AND A_mem is still flat
              -> stop the run (no signal left at this schedule); configurable

The controller never raises: any failure inside a callback hook is caught and
logged so a research run is never killed by its own instrumentation.
"""

import random
from typing import Optional

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

# action codes (also written to adaptive/action)
ACTION_NONE = 0
ACTION_BOOST = 1
ACTION_DECAY = 2
ACTION_DIAGNOSE = 3
ACTION_STOP_SUCCESS = 4
ACTION_STOP_DIVERGE = 5
ACTION_STOP_EXHAUSTED = 6
ACTION_NAMES = {
    ACTION_NONE: "none",
    ACTION_BOOST: "lr_boost",
    ACTION_DECAY: "lr_decay",
    ACTION_DIAGNOSE: "mem_diagnosis",
    ACTION_STOP_SUCCESS: "stop_success",
    ACTION_STOP_DIVERGE: "stop_eval_divergence",
    ACTION_STOP_EXHAUSTED: "stop_exhausted",
}


class AdaptiveTrainingController(TrainerCallback):
    """Per-epoch adaptive driver; add it AFTER MemorizationGateCallback so it
    observes the freshest A_mem / lambda values."""

    def __init__(
        self,
        lambda_scheduler,
        tokenizer=None,
        device: str = "cuda",
        mem_dataset=None,
        # plateau / boost
        window_epochs: int = 4,
        plateau_min_improvement: float = 0.005,
        boost_factor: float = 1.5,
        max_boosts: int = 4,
        max_lr_mult: float = 8.0,
        # instability / decay
        decay_factor: float = 0.7,
        min_lr_mult: float = 0.3,
        jump_threshold: float = 0.25,
        grad_spike_mult: float = 6.0,
        # stagnation diagnosis
        stagnation_epochs: int = 6,
        stagnation_floor: float = 0.05,
        diagnose_samples: int = 8,
        diagnose_every: int = 6,
        # divergence / stopping
        divergence_tolerance: float = 0.15,
        divergence_patience: int = 6,
        # eval-aware LR control (fixes the "boost while validation diverges"
        # failure mode observed in the 50-epoch adaptive runs, where the
        # controller issued 4 boosts / 0 decays and eval loss rose 25-40%)
        eval_decay_patience: int = 2,
        eval_boost_block: int = 2,
        # cumulative-multiplier floor for eval-driven decays (see 1b):
        # damping stops here so memorization can still saturate the gate
        eval_decay_floor: float = 0.5,
        stop_on_success: bool = False,
        success_patience: int = 3,
        stop_on_exhausted: bool = True,
        exhausted_stagnation_mult: int = 2,
        seed: int = 42,
    ):
        self.lambda_scheduler = lambda_scheduler
        self.tokenizer = tokenizer
        self.device = device
        self.mem_dataset = mem_dataset

        self.window_epochs = window_epochs
        self.plateau_min_improvement = plateau_min_improvement
        self.boost_factor = boost_factor
        self.max_boosts = max_boosts
        self.max_lr_mult = max_lr_mult
        self.decay_factor = decay_factor
        self.min_lr_mult = min_lr_mult
        self.jump_threshold = jump_threshold
        self.grad_spike_mult = grad_spike_mult

        self.stagnation_epochs = stagnation_epochs
        self.stagnation_floor = stagnation_floor
        self.diagnose_samples = diagnose_samples
        self.diagnose_every = diagnose_every

        self.divergence_tolerance = divergence_tolerance
        self.divergence_patience = divergence_patience
        self.eval_decay_patience = max(1, eval_decay_patience)
        self.eval_boost_block = max(0, eval_boost_block)
        self.eval_decay_floor = float(eval_decay_floor)
        self.stop_on_success = stop_on_success
        self.success_patience = success_patience
        self.stop_on_exhausted = stop_on_exhausted
        self.exhausted_stagnation_mult = exhausted_stagnation_mult

        self.rng = random.Random(seed)

        # rolling state
        self._bucket = []  # (loss, grad_norm) entries of the in-progress epoch
        self.epoch_stats = []  # finalized per-epoch dicts
        self._last_lr_change_epoch = -10**9
        self.boosts = 0
        self.decays = 0
        self.lr_mult = 1.0
        self.stagnation_count = 0
        self._last_diag_epoch = -10**9
        self.success_epoch: Optional[int] = None
        self.success_streak = 0
        self.best_eval_loss: Optional[float] = None
        self.eval_bad_streak = 0
        self.stop_reason: Optional[str] = None
        self.history = []

    # ------------------------------------------------------------------ hooks

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Accumulate logged losses into the current epoch bucket."""
        try:
            if not logs:
                return
            if "loss" in logs:
                self._bucket.append(
                    (float(logs["loss"]),
                     float(logs["grad_norm"]) if logs.get("grad_norm") is not None else None)
                )
        except Exception:
            pass

    def on_epoch_end(self, args, state, control, model=None, optimizer=None,
                     lr_scheduler=None, **kwargs):
        try:
            self._on_epoch_end_impl(args, state, control, model=model,
                                    optimizer=optimizer, lr_scheduler=lr_scheduler)
        except Exception as e:
            print(f"[ADAPTIVE] controller error at epoch {state.epoch}: {e}")

    # ------------------------------------------------------------------ core

    def _on_epoch_end_impl(self, args, state, control, model=None,
                           optimizer=None, lr_scheduler=None):
        epoch_no = int(round(state.epoch)) or len(self.epoch_stats) + 1

        bucket, self._bucket = self._bucket, []
        losses = [b[0] for b in bucket]
        grads = [b[1] for b in bucket if b[1] is not None]
        mean_loss = sum(losses) / len(losses) if losses else None
        grad_med = sorted(grads)[len(grads) // 2] if grads else None

        a_mem = float(getattr(self.lambda_scheduler, "current_a_mem", 0.0) or 0.0)
        lam = float(self.lambda_scheduler.get_lambda())

        current_lr = None
        if optimizer is not None and getattr(optimizer, "param_groups", None):
            current_lr = optimizer.param_groups[0].get("lr")

        record = {
            "epoch": epoch_no,
            "mean_train_loss": mean_loss,
            "grad_median": grad_med,
            "a_mem": a_mem,
            "lambda": lam,
            "lr": current_lr,
            "lr_mult": self.lr_mult,
        }

        action = ACTION_NONE
        info = {}

        # ---- finalize eval-loss tracking (trainer logged eval_loss this epoch)
        eval_loss = None
        for entry in reversed(getattr(state, "log_history", [])):
            if "eval_loss" in entry:
                eval_loss = float(entry["eval_loss"])
                break
        record["eval_loss"] = eval_loss
        if eval_loss is not None:
            if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.eval_bad_streak = 0
            elif eval_loss > self.best_eval_loss * (1 + self.divergence_tolerance):
                self.eval_bad_streak += 1
            else:
                self.eval_bad_streak = 0

        # ---- success: the gate opened (memorization saturated) -------------
        if lam > 0:
            self.success_streak += 1
            if self.success_epoch is None:
                self.success_epoch = epoch_no
                print(f"[ADAPTIVE] *** PATTERN FOUND: memorization saturated and the "
                      f"lambda gate opened at epoch {epoch_no} (A_mem={a_mem:.3f}, "
                      f"lambda={lam:.4f}) ***")
        else:
            self.success_streak = 0

        # ---- stagnation tracking (before boost logic) ----------------------
        if a_mem < self.stagnation_floor:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0

        # ---- decisions ------------------------------------------------------
        prev = self.epoch_stats[-1] if self.epoch_stats else None

        # 1) instability -> decay (never right after a boost/decay; never when
        #    the probe loss is active, because a rising total loss is then
        #    EXPECTED, not instability)
        if lam == 0 and prev is not None and mean_loss is not None and prev["mean_train_loss"]:
            cooldown_ok = epoch_no - self._last_lr_change_epoch >= self.window_epochs
            jump = (mean_loss - prev["mean_train_loss"]) / prev["mean_train_loss"]
            spike = False
            if grad_med is not None and len(self.epoch_stats) >= 3:
                hist_grads = sorted(
                    s["grad_median"] for s in self.epoch_stats if s["grad_median"]
                )
                if hist_grads:
                    run_med = hist_grads[len(hist_grads) // 2]
                    spike = grad_med > self.grad_spike_mult * max(run_med, 1e-8)
            if cooldown_ok and (jump > self.jump_threshold or spike):
                action = ACTION_DECAY
                ok = self._apply_lr_mult(self.decay_factor, lr_scheduler,
                                         optimizer, floor=self.min_lr_mult)
                if ok:
                    self.decays += 1
                    info["jump"] = jump
                    info["spike"] = spike

        # 1b) eval-aware damping: held-out loss degrading past the best value
        #     by > divergence_tolerance for eval_decay_patience consecutive
        #     evaluations (while lambda == 0) -> DECAY the LR. This is the
        #     direct countermeasure to the observed controller mismatch, where
        #     eval loss rose 25-40% after its minimum while the LR multiplier
        #     was driven UP to 5.06x by plateau boosts.
        #     BOUNDED by eval_decay_floor (default 0.5): rising eval IS the
        #     memorization signature the lambda gate waits for, so damping
        #     must never crush memorization to a crawl. The half50v3 run
        #     showed the unbounded spiral lr_mult 1.0 -> 0.49 -> 0.34 -> 0.24
        #     -> 0.17 (breaching min_lr_mult) with A_mem_lf pinned at ~0.02
        #     -- tau_0=0.8 unreachable. Floor at 0.5 = "damp, don't crush";
        #     the eval_boost_block guard remains the primary anti-instability
        #     mechanism.
        if (action == ACTION_NONE and lam == 0 and self.success_epoch is None
                and self.eval_bad_streak >= self.eval_decay_patience
                and epoch_no - self._last_lr_change_epoch >= max(1, self.window_epochs // 2)):
            ok = self._apply_lr_mult(self.decay_factor, lr_scheduler,
                                     optimizer, floor=self.eval_decay_floor)
            if ok:
                action = ACTION_DECAY
                self.decays += 1
                info["decay_reason"] = "eval_degrading"
                info["eval_bad_streak"] = self.eval_bad_streak
                # a decay releases eval pressure; restart the bad streak so
                # repeated decays only happen after a NEW degradation episode
                self.eval_bad_streak = 0
            else:
                info["eval_decay_blocked"] = "at_eval_decay_floor"

        # 2) plateau -> boost (only pre-saturation: post-activation dynamics
        #    belong to the probe loss and must not be accelerated).
        #    BLOCKED while held-out loss is actively degrading: boosting a
        #    memorizing model accelerates exactly the overfitting we observe.
        if action == ACTION_NONE and lam == 0 and self.success_epoch is None:
            if len(self.epoch_stats) >= self.window_epochs - 1 and mean_loss is not None:
                base = self.epoch_stats[-(self.window_epochs - 1)]["mean_train_loss"]
                if base:
                    slope = (mean_loss - base) / self.window_epochs
                    improving = slope < -self.plateau_min_improvement
                    record["loss_slope"] = slope
                    if not improving and self.boosts < self.max_boosts \
                            and self.lr_mult < self.max_lr_mult \
                            and epoch_no - self._last_lr_change_epoch >= self.window_epochs:
                        if self.eval_bad_streak >= max(1, self.eval_boost_block):
                            info["boost_blocked"] = "eval_degrading"
                            info["eval_bad_streak"] = self.eval_bad_streak
                        else:
                            mult = min(self.boost_factor, self.max_lr_mult / self.lr_mult)
                            if self._apply_lr_mult(mult, lr_scheduler, optimizer):
                                action = ACTION_BOOST
                                self.boosts += 1
                                info["boost_mult"] = mult

        # 3) stagnation diagnosis ---------------------------------------------
        if (self.stagnation_count >= self.stagnation_epochs
                and epoch_no - self._last_diag_epoch >= self.diagnose_every
                and self.tokenizer is not None and self.mem_dataset is not None):
            ok, match_rate = self._run_diagnosis(model)
            if ok:
                action = ACTION_DIAGNOSE if action == ACTION_NONE else action
                info["diag_exact_match"] = match_rate
                self._last_diag_epoch = epoch_no

        # 4) stopping rules ----------------------------------------------------
        if self.stop_on_success and self.success_streak >= self.success_patience \
                and self.eval_bad_streak >= 2:
            action = ACTION_STOP_SUCCESS
            control.should_training_stop = True
            self.stop_reason = "success_sustained_then_degrading"
        elif (self.stop_on_exhausted
              and self.success_epoch is None
              and self.boosts >= self.max_boosts
              and self.stagnation_count >= self.exhausted_stagnation_mult * self.stagnation_epochs):
            action = ACTION_STOP_EXHAUSTED
            control.should_training_stop = True
            self.stop_reason = "boosts_exhausted_without_saturation"
        elif (self.eval_bad_streak >= self.divergence_patience and lam == 0
              and self.success_epoch is None):
            # divergence is logged every epoch; only STOP if explicitly enabled
            # via divergence_patience < 0 semantics is not used -- instead the
            # stop_on_exhausted rule above guards wasted compute.
            info["eval_diverging"] = True

        if action == ACTION_NONE and self.eval_bad_streak >= self.divergence_patience:
            print(f"[ADAPTIVE] eval_loss diverging for {self.eval_bad_streak} epochs "
                  f"(best {self.best_eval_loss:.4f}) without gate activation -- "
                  f"memorization overfitting phase, continuing")

        self._last_lr_change_epoch = (
            epoch_no if action in (ACTION_BOOST, ACTION_DECAY) else self._last_lr_change_epoch
        )

        record["action"] = ACTION_NAMES.get(action, str(action))
        record["lr_mult_pre"] = record["lr_mult"]
        record["lr_mult"] = self.lr_mult  # post-action value
        record["eval_bad_streak"] = self.eval_bad_streak
        record["eval_best"] = self.best_eval_loss
        record.update(info)
        self.epoch_stats.append(record)
        self.history.append(record)

        # ---- reporting -------------------------------------------------------
        msg = (f"[ADAPTIVE] epoch {epoch_no}: loss={mean_loss if mean_loss is None else round(mean_loss, 4)} "
               f"a_mem={a_mem:.4f} lambda={lam:.4f} lr_mult={self.lr_mult:.2f} "
               f"action={record['action']}")
        if action == ACTION_BOOST:
            msg += (f" x{info.get('boost_mult')} -> lr_mult={self.lr_mult:.2f} "
                    f"({self.boosts}/{self.max_boosts} boosts)")
        elif action == ACTION_DECAY:
            msg += (f" -> lr_mult={self.lr_mult:.2f}"
                    f" ({info.get('decay_reason', 'instability')})")
        if record.get("boost_blocked"):
            msg += f" [boost blocked: {record['boost_blocked']}]"
        if record.get("diag_exact_match") is not None:
            msg += f" diag_match={record['diag_exact_match']:.2f}"
        print(msg)

        self._log_wandb(record)

    # ------------------------------------------------------------------ utils

    def _apply_lr_mult(self, mult: float, lr_scheduler, optimizer,
                       floor: Optional[float] = None,
                       ceil: Optional[float] = None) -> bool:
        """Scale the LR schedule base LRs (and current LRs) by `mult`.

        `floor`/`ceil` clamp the resulting *cumulative* multiplier. A floor
        REFUSES the adjustment when the multiplier is already at/below it
        (a floor must never raise the LR back up); a ceil caps boosts.
        """
        try:
            new_mult = self.lr_mult * mult
            if floor is not None and self.lr_mult <= floor + 1e-9:
                return False  # already damped to the floor: refuse, don't raise
            if floor is not None:
                new_mult = max(floor, new_mult)
            if ceil is not None:
                new_mult = min(ceil, new_mult)
            # Effective multiplier actually applied to the LRs: differs from
            # `mult` when a floor/ceil clamp bites (LR must move to the
            # clamped target, not one `mult` step past it).
            eff = new_mult / self.lr_mult
            if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
                lr_scheduler.base_lrs = [b * eff for b in lr_scheduler.base_lrs]
            if optimizer is not None and getattr(optimizer, "param_groups", None):
                for g in optimizer.param_groups:
                    if "lr" in g and g["lr"] is not None:
                        g["lr"] = g["lr"] * eff
            self.lr_mult = round(new_mult, 6)
            return True
        except Exception as e:
            print(f"[ADAPTIVE] LR adjustment failed: {e}")
            return False

    def _run_diagnosis(self, model):
        """Generate answers for a few D_mem prompts and log them to W&B.

        Returns (ok, exact_match_rate). Tells a learning failure (garbage
        generations) apart from an extraction artifact (right answer, wrong
        format) when A_mem stagnates near zero.
        """
        try:
            from ..evaluation.gen_eval import (
                _strip_answer_suffix,
                extract_answer_from_generation,
                compute_exact_match,
            )

            n = min(self.diagnose_samples, len(self.mem_dataset))
            rows = [self.mem_dataset[i] for i in range(n)]
            prompts = [_strip_answer_suffix(r["text"]) for r in rows]
            targets = [r.get("label_text", r.get("entity", r.get("target", ""))) for r in rows]

            prev_side = getattr(self.tokenizer, "padding_side", "right")
            self.tokenizer.padding_side = "left"
            try:
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                        truncation=True, max_length=512).to(self.device)
                outputs = model.generate(
                    **inputs, max_new_tokens=24, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id)
                gens = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            finally:
                self.tokenizer.padding_side = prev_side

            table_rows = []
            correct = 0
            for prompt, gen, tgt in zip(prompts, gens, targets):
                pred = extract_answer_from_generation(gen)
                hit = compute_exact_match(pred, tgt)
                correct += int(hit)
                table_rows.append([prompt, gen[-200:], tgt, pred, bool(hit)])
            match_rate = correct / max(n, 1)

            try:
                import wandb
                if wandb.run is not None:
                    table = wandb.Table(
                        columns=["prompt", "generation", "target", "prediction", "exact_match"],
                        data=table_rows)
                    wandb.log({"adaptive/diagnosis_table": table,
                               "adaptive/diag_exact_match": match_rate})
            except Exception:
                pass
            return True, match_rate
        except Exception as e:
            print(f"[ADAPTIVE] diagnosis failed: {e}")
            return False, None

    def _log_wandb(self, record):
        try:
            import wandb
            if wandb.run is None:
                return
            metrics = {
                "adaptive/epoch": record["epoch"],
                "adaptive/action": record["action"],
                "adaptive/lr_mult": record["lr_mult"],
                "adaptive/a_mem": record["a_mem"],
                "adaptive/lambda": record["lambda"],
                "adaptive/stagnation_epochs": self.stagnation_count,
                "adaptive/boosts": self.boosts,
                "adaptive/decays": self.decays,
                "adaptive/eval_bad_streak": self.eval_bad_streak,
            }
            if self.best_eval_loss is not None:
                metrics["adaptive/eval_best"] = self.best_eval_loss
            if record.get("mean_train_loss") is not None:
                metrics["adaptive/mean_train_loss"] = record["mean_train_loss"]
            if record.get("eval_loss") is not None:
                metrics["adaptive/eval_loss"] = record["eval_loss"]
            if record.get("loss_slope") is not None:
                metrics["adaptive/loss_slope"] = record["loss_slope"]
            if record.get("grad_median") is not None:
                metrics["adaptive/grad_median"] = record["grad_median"]
            if self.success_epoch is not None:
                metrics["adaptive/pattern_found_epoch"] = self.success_epoch
            wandb.log(metrics)
        except Exception:
            pass

    def on_train_end(self, args, state, control, model=None, **kwargs):
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "adaptive/summary_boosts": self.boosts,
                    "adaptive/summary_decays": self.decays,
                    "adaptive/summary_final_lr_mult": self.lr_mult,
                    "adaptive/summary_pattern_found_epoch": self.success_epoch or -1,
                    "adaptive/summary_stop_reason": self.stop_reason or "budget_reached",
                })
            print(f"[ADAPTIVE] run summary: boosts={self.boosts} decays={self.decays} "
                  f"final_lr_mult={self.lr_mult} pattern_found_at_epoch={self.success_epoch} "
                  f"stop_reason={self.stop_reason or 'budget_reached'}")
        except Exception:
            pass
