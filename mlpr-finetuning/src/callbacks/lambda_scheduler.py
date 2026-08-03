"""Lambda scheduler for gating the probe loss."""

from typing import Optional


class LambdaScheduler:
    """
    Scheduler that computes lambda(t) based on memorization accuracy.
    
    The gating function is:
    λ(t) = λ₀ * min(1, max(0, A_mem(t) - τ₀) / Δ₀)
    
    This ensures the probe loss is only activated after memorization
    reaches a certain threshold (τ₀).
    """
    
    def __init__(
        self,
        lambda_0: float = 0.3,
        tau_0: float = 0.9,
        delta_0: float = 0.05,
    ):
        """
        Initialize the lambda scheduler.
        
        Args:
            lambda_0: Maximum lambda value (λ₀)
            tau_0: Memorization threshold (τ₀) - probe activates when A_mem > τ₀
            delta_0: Smoothing factor (Δ₀) for gradual activation
        """
        self.lambda_0 = lambda_0
        self.tau_0 = tau_0
        self.delta_0 = delta_0
        
        # Current memorization accuracy
        self.current_a_mem: float = 0.0
        
        # Current lambda value
        self._current_lambda: float = 0.0
        
        # Track if memorization has saturated
        self.memorization_saturated: bool = False
        
        # Track if probe has been activated
        self.probe_activated: bool = False
    
    def update_a_mem(self, a_mem: float) -> None:
        """
        Update the memorization accuracy and recalculate lambda.
        
        Args:
            a_mem: Current memorization accuracy (0.0 to 1.0)
        """
        self.current_a_mem = a_mem
        
        # Check if memorization has saturated
        if a_mem >= self.tau_0:
            self.memorization_saturated = True
        
        # Calculate lambda using the gating function
        self._current_lambda = self._compute_lambda(a_mem)
        
        # Check if probe is activated
        if self._current_lambda > 0:
            self.probe_activated = True
    
    def _compute_lambda(self, a_mem: float) -> float:
        """
        Compute lambda using the gating function.
        
        λ(t) = λ₀ * min(1, max(0, A_mem(t) - τ₀) / Δ₀)
        
        Args:
            a_mem: Current memorization accuracy
            
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
            "tau_0": self.tau_0,
            "lambda_0": self.lambda_0,
            "delta_0": self.delta_0,
            "memorization_saturated": self.memorization_saturated,
            "probe_activated": self.probe_activated,
        }
