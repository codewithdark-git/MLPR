"""
Main entry point for MLPR fine-tuning.

This script orchestrates the entire training pipeline:
1. Load configuration
2. Initialize W&B logging
3. Load model from HuggingFace
4. Load dataset from HuggingFace
5. Setup LoRA and probe head
6. Train with MLPR loss
7. Push checkpoints to HuggingFace
8. Save logs to W&B
"""

import os
import argparse
import yaml
from typing import Optional

import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from src.data import MLPDataset, MLPRTargetCollator
from src.models import setup_lora_model, LinearProbe
from src.trainer import MLPRTrainer
from src.callbacks import LambdaScheduler, LifecycleCheckpointCallback, WnBMatrixLockCallback
from src.evaluation import evaluate_memorization, evaluate_generalization


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MLPR Fine-tuning")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen2.5_7b.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override model name from config"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Override dataset name from config"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push final model to HuggingFace Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID for pushing"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/organization name"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config: dict, args) -> None:
    """Initialize Weights & Biases logging."""
    wandb_entity = args.wandb_entity or config.get("wandb_entity", "default")
    wandb_project = args.wandb_project or config.get("wandb_project", "mlpr-finetuning")
    
    # Check for API key
    if not os.environ.get("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not set. W&B logging will be disabled.")
        os.environ["WANDB_MODE"] = "disabled"
    
    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        config=config,
        name=f"mlpr-{config.get('model_name', 'model').split('/')[-1]}",
    )


def load_model_and_tokenizer(config: dict, args) -> tuple:
    """Load model and tokenizer from HuggingFace."""
    model_name = args.model_name or config.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    print(f"Model loaded successfully")
    print(f"Model has {model.config.num_hidden_layers} layers")
    
    return model, tokenizer


def setup_probe(model, num_entities: int, config: dict) -> LinearProbe:
    """Setup the linear probe head."""
    hidden_size = model.config.hidden_size
    
    probe = LinearProbe(
        hidden_size=hidden_size,
        num_entities=num_entities,
        bias=True,
        dropout=0.0,
    )
    
    print(f"Probe created: {hidden_size} -> {num_entities}")
    return probe


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    # This is called during evaluation
    # For now, return empty dict - actual metrics computed separately
    return {}


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Override config with command line args
    if args.model_name:
        config["model_name"] = args.model_name
    if args.dataset_name:
        config["dataset_name"] = args.dataset_name
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.hub_model_id:
        config["hub_model_id"] = args.hub_model_id
    
    # Setup W&B
    setup_wandb(config, args)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer from HuggingFace
    model, tokenizer = load_model_and_tokenizer(config, args)
    
    # Load dataset from HuggingFace
    dataset_name = config.get("dataset_name", "mlpr-kb-dataset")
    entity_vocab_path = config.get("entity_vocab_path")
    
    print(f"Loading dataset: {dataset_name}")
    mlpr_dataset = MLPDataset(
        dataset_name=dataset_name,
        entity_vocab_path=entity_vocab_path,
        tokenizer=tokenizer,
        max_length=512,
    )
    
    # Get number of entities for probe
    num_entities = mlpr_dataset.candidate_set_size
    print(f"Candidate set size: {num_entities}")
    
    # Prepare datasets
    train_dataset = mlpr_dataset.get_train_dataset()
    eval_dataset = mlpr_dataset.get_eval_dataset()
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = mlpr_dataset.prepare_dataset_for_tokenizer(train_dataset, tokenizer)
    eval_dataset = mlpr_dataset.prepare_dataset_for_tokenizer(eval_dataset, tokenizer)
    
    # Setup LoRA
    print("Setting up LoRA...")
    lora_config = {
        "r": config.get("lora_r", 16),
        "alpha": config.get("lora_alpha", 32),
        "dropout": config.get("lora_dropout", 0.05),
        "target_modules": config.get("lora_target_modules"),
    }
    model = setup_lora_model(model, **lora_config)
    
    # Setup probe head
    l_star = config.get("l_star", 14)
    probe_head = setup_probe(model, num_entities, config)
    
    # Setup lambda scheduler
    lambda_scheduler = LambdaScheduler(
        lambda_0=config.get("lambda_0", 0.3),
        tau_0=config.get("tau_0", 0.9),
        delta_0=config.get("delta_0", 0.05),
    )
    
    # Create data collator
    collator = MLPRTargetCollator(
        tokenizer=tokenizer,
        padding="longest",
    )
    
    # Setup training arguments
    output_dir = args.output_dir or config.get("output_dir", "./outputs")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        num_train_epochs=config.get("num_train_epochs", 3),
        learning_rate=config.get("learning_rate", 2e-5),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        weight_decay=config.get("weight_decay", 0.01),
        evaluation_strategy=config.get("evaluation_strategy", "epoch"),
        save_strategy=config.get("save_strategy", "epoch"),
        logging_steps=config.get("logging_steps", 10),
        fp16=False,  # Use bf16 if available
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        push_to_hub=args.push_to_hub,
        hub_model_id=config.get("hub_model_id"),
        report_to="wandb",
        remove_unused_columns=False,
    )
    
    # Setup huggingface-lifecycle manager for checkpoint push/pull
    hf_manager = None
    if config.get("lifecycle_enabled", True):
        try:
            from hf_lifecycle import HFManager, KeepLastN
            
            output_dir = args.output_dir or config.get("output_dir", "./outputs")
            hub_model_id = config.get("hub_model_id")
            
            hf_manager = HFManager(
                repo_id=hub_model_id,
                local_dir=output_dir,
                checkpoint_dir=f"{output_dir}/checkpoints",
                hf_token=os.environ.get("HF_TOKEN"),
                retention_policy=KeepLastN(3),
                auto_push=False,  # We control push timing via callback
            )
            print("HFManager initialized for lifecycle checkpoint management")
        except ImportError:
            print("Warning: huggingface-lifecycle not installed. Install with:")
            print("  pip install git+https://github.com/codewithdark-git/huggingface-lifecycle.git")
        except Exception as e:
            print(f"Warning: Could not initialize HFManager: {e}")

    # Setup callbacks
    lifecycle_callback = LifecycleCheckpointCallback(
        hf_manager=hf_manager,
        enabled=config.get("lifecycle_enabled", True),
        push_every_n_epochs=config.get("lifecycle_push_every_n_epochs", 1),
    )
    lifecycle_callback.set_lambda_scheduler(lambda_scheduler)
    
    wandb_callback = WnBMatrixLockCallback(
        log_probe_matrix=True,
        log_lora_matrices=True,
        lock_checkpoints=True,
        l_star=l_star,
    )
    
    # Create trainer
    print("Creating MLPR Trainer...")
    trainer = MLPRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        probe_head=probe_head,
        lambda_scheduler=lambda_scheduler,
        l_star=l_star,
        callbacks=[lifecycle_callback, wandb_callback],
    )
    
    # Run baseline evaluation before training (Condition A)
    print("\n" + "="*50)
    print("Running Baseline Evaluation (Pre-training)")
    print("="*50)
    
    baseline_a_mem, _ = evaluate_memorization(model, tokenizer, eval_dataset, device)
    baseline_a_gen, _ = evaluate_generalization(model, tokenizer, eval_dataset, device)
    
    print(f"Baseline A_mem: {baseline_a_mem:.4f}")
    print(f"Baseline A_gen: {baseline_a_gen:.4f}")
    
    lifecycle_callback.set_baseline_a_gen(baseline_a_gen)
    
    # Start training
    print("\n" + "="*50)
    print("Starting MLPR Training")
    print("="*50)
    
    trainer.train()
    
    # Final evaluation
    print("\n" + "="*50)
    print("Running Final Evaluation (Post-training)")
    print("="*50)
    
    final_a_mem, mem_results = evaluate_memorization(model, tokenizer, eval_dataset, device)
    final_a_gen, gen_results = evaluate_generalization(model, tokenizer, eval_dataset, device)
    
    print(f"\nFinal A_mem: {final_a_mem:.4f}")
    print(f"Final A_gen: {final_a_gen:.4f}")
    print(f"Final Gap: {final_a_mem - final_a_gen:.4f}")
    
    # Log final metrics to W&B
    wandb.log({
        "final_a_mem": final_a_mem,
        "final_a_gen": final_a_gen,
        "final_gap": final_a_mem - final_a_gen,
        "baseline_a_mem": baseline_a_mem,
        "baseline_a_gen": baseline_a_gen,
        "improvement_a_gen": final_a_gen - baseline_a_gen,
    })
    
    # Save final model locally
    print(f"\nSaving final model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        print("\nPushing model to HuggingFace Hub...")
        hub_model_id = config.get("hub_model_id", "mlpr-finetuned-model")
        trainer.push_to_hub(hub_model_id)
        print(f"Model pushed to: {hub_model_id}")
    
    # Finish W&B run
    wandb.finish()
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)


if __name__ == "__main__":
    main()
