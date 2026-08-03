"""LoRA setup for PEFT fine-tuning."""

from typing import List, Optional
from transformers import PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType


def setup_lora_model(
    model: PreTrainedModel,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
) -> PreTrainedModel:
    """
    Configure and apply LoRA to a pretrained model.
    
    Args:
        model: The pretrained transformer model
        r: LoRA rank (dimension of low-rank matrices)
        alpha: LoRA scaling factor
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        bias: Bias configuration for LoRA ("none", "all", or "lora_only")
        
    Returns:
        Model with LoRA adapters applied
    """
    # Default target modules for transformer attention layers
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
    )
    
    # Apply LoRA to model
    peft_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    peft_model.print_trainable_parameters()
    
    return peft_model


def create_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
) -> LoraConfig:
    """
    Create a LoRA configuration without applying it to a model.
    
    Args:
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability
        target_modules: Target module names
        bias: Bias configuration
        
    Returns:
        LoraConfig instance
    """
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
    )
