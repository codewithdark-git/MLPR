"""Lambda scheduler for gating the probe loss."""

from typing import Optional

# Gate metrics:
#   "generation" - free-generation exact match on D_mem (idea.md v1).
#     Diagnosed UNREACHABLE in this regime: plateaus at 0-12% vs tau_0=0.9
#     because it conflates *knowing* a fact with *saying* it verbatim.
#   "likelihood" - teacher-forced greedy answer-token match (ParaRel-style
#     recall probe). One forward pass; measures knowing, not saying.
#   "max"        - max(generation, likelihood): opens if the model can
#     recall the fact under ANY recall mode. Recommended (v2 default).
GATE_GENERATION = "generation"
GATE_LIKELIHOOD = "likelihood"
GATE_MAX = "max"
_VALID_GATE_METRICS = (GATE_GENERATION, GATE_LIKELIHOOD, GATE_MAX)


class LambdaScheduler:
    """
    Scheduler that computes lambda(t) based on memorization accuracy.
    
    The gating function is:
    λ(t) = λ₀ * min(1, max(0, A_mem(t) - τ₀) / Δ₀)
    
    This ensures the probe loss is only activated after memorization
    reaches a certain threshold (τ₀).

    v2: A_mem(t) can be measured in two recall modes -- free-generation
    exact match ("saying") and teacher-forced likelihood match ("knowing").
    `gate_metric` selects which signal (or their max) drives the gate;
    both are always recorded for the paper curves.
    """
    
    def __init__(
        self,
        lambda_0: float = 0.3,
        tau_0: float = 0.9,
        delta_0: float = 0.05,
        gate_metric: str = GATE_GENERATION,
    ):
        """
        Initialize the lambda scheduler.
        
        Args:
            lambda_0: Maximum lambda value (λ₀)
            tau_0: Memorization threshold (τ₀) - probe activates when A_mem > τ₀
            delta_0: Smoothing factor (Δ₀) for gradual activation
            gate_metric: which A_mem signal drives the gate --
                "generation" (v1, free-generation EM), "likelihood"
                (v2, teacher-forced EM) or "max" (v2, both).
        """
        if gate_metric not in _VALID_GATE_METRICS:
            raise ValueError(
                f"gate_metric must be one of {_VALID_GATE_METRICS}, "
                f"got {gate_metric!r}")
        self.lambda_0 = lambda_0
        self.tau_0 = tau_0
        self.delta_0 = delta_0
        self.gate_metric = gate_metric
        
        # Raw per-mode memorization accuracies
        self.current_a_mem_gen: float = 0.0
        self.current_a_mem_lf: Optional[float] = None
        
        # Effective gate signal (per gate_metric) -- this is what the
        # adaptive controller also reads via .current_a_mem
        self.current_a_mem: float = 0.0
        
        # Current lambda value
        self._current_lambda: float = 0.0
        
        # Track if memorization has saturated
        self.memorization_saturated: bool = False
        
        # Track if probe has been activated
        self.probe_activated: bool = False
    
    def update_a_mem(self, a_mem: float, a_mem_likelihood: Optional[float] = None) -> None:
        """
        Update the memorization accuracies and recalculate lambda.
        
        Args:
            a_mem: Free-generation exact-match accuracy on D_mem (0..1)
            a_mem_likelihood: Teacher-forced likelihood EM on D_mem (0..1);
                None when the likelihood evaluator is unavailable (falls back
                to the generation signal regardless of gate_metric).
        """
        self.current_a_mem_gen = float(a_mem)
        self.current_a_mem_lf = (
            None if a_mem_likelihood is None else float(a_mem_likelihood)
        )
        
        # Effective gate signal per the configured metric
        if self.gate_metric == GATE_LIKELIHOOD and self.current_a_mem_lf is not None:
            signal = self.current_a_mem_lf
        elif self.gate_metric == GATE_MAX and self.current_a_mem_lf is not None:
            signal = max(self.current_a_mem_gen, self.current_a_mem_lf)
        else:
            signal = self.current_a_mem_gen
        self.current_a_mem = signal
        
        # Check if memorization has saturated
        if signal >= self.tau_0:
            self.memorization_saturated = True
        
        # Calculate lambda using the gating function
        self._current_lambda = self._compute_lambda(signal)
        
        # Check if probe is activated
        if self._current_lambda > 0:
            self.probe_activated = True
    
    def _compute_lambda(self, a_mem: float) -> float:
        """
        Compute lambda using the gating function.
        
        λ(t) = λ₀ * min(1, max(0, A_mem(t) - τ₀) / Δ₀)
        
        Args:
            a_mem: Effective memorization-accuracy signal
            
        Returns:
            Lambda value between 0 and λ₀
        """
        # Compute the gating factor
        numerator = max(0.0, a_mem - self.tau_0)
        gating_factor = min(1.0, numerator / self.delta_0)
        
        return self.lambda_0 * gating_factor
    
    def get_lambda(self) -> float:
        """
        Get the current lambda value.
        
        Returns:
            Current lambda value
        """
        return self._current_lambda
    
    def reset(self) -> None:
        """Reset the scheduler state."""
        self.current_a_mem = 0.0
        self.current_a_mem_gen = 0.0
        self.current_a_mem_lf = None
        self._current_lambda = 0.0
        self.memorization_saturated = False
        self.probe_activated = False
    
    def should_activate_probe(self) -> bool:
        """
        Check if the probe loss should be activated.
        
        Returns:
            True if lambda > 0
        """
        return self._current_lambda > 0
    
    def is_memorization_saturated(self) -> bool:
        """
        Check if memorization has reached saturation.
        
        Returns:
            True if A_mem >= τ₀
        """
        return self.memorization_saturated
    
    def get_status(self) -> dict:
        """
        Get the current status of the scheduler.
        
        Returns:
            Dictionary with scheduler status information
        """
        return {
            "lambda": self._current_lambda,
            "a_mem": self.current_a_mem,
            "a_mem_gen": self.current_a_mem_gen,
            "a_mem_likelihood": self.current_a_mem_lf,
            "gate_metric": self.gate_metric,
            "tau_0": self.tau_0,
            "lambda_0": self.lambda_0,
            "delta_0": self.delta_0,
            "memorization_saturated": self.memorization_saturated,
            "probe_activated": self.probe_activated,
        }
