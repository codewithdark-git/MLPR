# MLPR Fine-Tuning: Closing the Knowing–Using Gap

**Mid-Layer Probe Regularization Strategy for Open Pretrained Models (Qwen / LLaMA)**

## Overview

This repository implements **Mid-Layer Probe Regularization (MLPR)**, a training-time auxiliary loss technique that closes the "Knowing–Using Gap" in large language models during fine-tuning. While standard supervised fine-tuning (SFT) achieves near-perfect memorization, models often fail to utilize memorized facts for compositional, multi-hop reasoning. MLPR addresses this by forcing mid-layer representations to remain "causally usable" through an auxiliary probe loss.

## Key Features

- **Composite Loss Function**: Combines standard cross-entropy loss with an auxiliary mid-layer probe loss
- **Gated Activation**: Probe loss activates only after memorization saturation ($A_{mem} > \tau_0$)
- **LoRA Integration**: Parameter-efficient fine-tuning with PEFT LoRA adapters
- **HuggingFace Integration**: Load datasets and models from HF Hub, push checkpoints back
- **W&B Logging**: Matrix visualization, checkpoint artifacts, and experiment tracking
- **Lifecycle Dashboard**: Real-time event emission for training milestones

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd mlpr-finetuning

# Install dependencies
pip install -r requirements.txt

# Optional: Install lifecycle dashboard package
pip install git+https://github.com/codewith-dark-git/huggingface-lifecycle.git
```

### Requirements

- Python 3.10+
- PyTorch >= 2.0
- Transformers >= 4.37.0
- PEFT >= 0.8.0
- Accelerate >= 0.27.0
- Weights & Biases >= 0.16.0
- Datasets >= 2.14.0

## Quick Start

### Basic Training Run

```bash
python main.py \
    --config configs/qwen2.5_7b.yaml \
    --output_dir ./outputs \
    --wandb_entity your-entity \
    --wandb_project knowing_using_gap
```

### Advanced Options

```bash
python main.py \
    --config configs/qwen2.5_7b.yaml \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset_name your-hf-dataset \
    --output_dir ./outputs \
    --push_to_hub \
    --hub_model_id your-username/mlpr-finetuned \
    --wandb_entity your-entity \
    --wandb_project knowing_using_gap
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to configuration YAML file | `configs/qwen2.5_7b.yaml` |
| `--model_name` | Override model name from config | None |
| `--dataset_name` | Override dataset name from config | None |
| `--output_dir` | Output directory for checkpoints | From config |
| `--push_to_hub` | Push final model to HuggingFace Hub | False |
| `--hub_model_id` | HuggingFace Hub model ID | None |
| `--wandb_entity` | W&B entity/organization name | From config |
| `--wandb_project` | W&B project name | From config |
| `--local_rank` | Local rank for distributed training | -1 |

## Methodology

### The Knowing–Using Gap

Standard SFT on structured knowledge bases drives single-hop memorization to near-zero CE loss, but creates a gap where:
- **Memorization Accuracy ($A_{mem}$)**: High (near 100%)
- **Generalization Accuracy ($A_{gen}$)**: Low (fails multi-hop reasoning)

### MLPR Solution

The training objective is a composite loss:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda(t) \cdot \mathcal{L}_{probe}$$

Where:
- $\mathcal{L}_{CE}$: Standard cross-entropy loss
- $\mathcal{L}_{probe}$: Auxiliary mid-layer probe loss at layer $l^* \approx 0.5L$
- $\lambda(t)$: Gating function based on memorization accuracy

### Gating Function

$$\lambda(t) = \lambda_0 \cdot \min\left(1, \frac{\max(0, A_{mem}(t)-\tau_0)}{\Delta_0}\right)$$

Default parameters:
- $\lambda_0 = 0.3$ (maximum probe weight)
- $\tau_0 = 0.9$ (memorization threshold)
- $\Delta_0 = 0.05$ (smoothing factor)

## Dataset Format

The dataset should be a closed Knowledge Base (KB) with fact triplets $(n_1, e_{12}, n_2)$:

### Memorization Set (~1,000 samples)
Single-hop QA pairs for training:
```json
{
    "text": "Who is the regulator of Bank X? Answer: FinancialAuthority",
    "entity": "FinancialAuthority",
    "relation": "regulator_of",
    "head_entity": "Bank X",
    "entity_end_char_idx": 52,
    "entity_class_id": 0,
    "type": "mem"
}
```

### Generalization Set (~500 samples)
Multi-hop QA pairs for evaluation only:
```json
{
    "text": "Who regulates the issuer of Product Y? Answer: FinancialAuthority",
    "entity": "FinancialAuthority",
    "relations": ["issues", "regulator_of"],
    "head_entity": "Product Y",
    "entity_end_char_idx": 58,
    "entity_class_id": 0,
    "type": "gen"
}
```

### Loading from HuggingFace

The dataset can be loaded directly from HuggingFace Hub:

```python
from datasets import load_dataset

dataset = load_dataset("your-username/mlpr-kb-dataset")
```

Or use the built-in sample dataset for testing.

## Configuration

Edit `configs/qwen2.5_7b.yaml` to customize:

```yaml
# Model settings
model_name: "Qwen/Qwen2.5-7B-Instruct"
l_star: 14  # Mid-layer index (0.5 * num_layers)

# LoRA settings
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

# Dataset settings
dataset_name: "mlpr-kb-dataset"
entity_vocab_path: "configs/entity_vocab.json"

# Training settings
per_device_train_batch_size: 4
num_train_epochs: 3
learning_rate: 2e-5

# MLPR settings
lambda_0: 0.3
tau_0: 0.9
delta_0: 0.05

# W&B settings
wandb_entity: "your-entity"
wandb_project: "knowing_using_gap"
```

## Architecture

```
mlpr-finetuning/
├── main.py                    # Entry point
├── configs/
│   └── qwen2.5_7b.yaml       # Hyperparameters
├── src/
│   ├── data/
│   │   ├── dataset.py        # HF dataset loading
│   │   └── collator.py       # Custom collator with entity_pos
│   ├── models/
│   │   ├── lora_setup.py     # PEFT LoRA configuration
│   │   └── probe.py          # Linear probe module
│   ├── trainer/
│   │   └── mlpr_trainer.py   # Custom HF Trainer
│   ├── callbacks/
│   │   ├── lambda_scheduler.py   # λ(t) gating function
│   │   ├── lifecycle_hooks.py    # Dashboard events
│   │   └── wandb_callback.py     # Matrix logging
│   └── evaluation/
│       └── gen_eval.py       # Multi-hop evaluation
├── tests/
│   └── test_mlpr.py          # Unit tests
└── requirements.txt
```

## Evaluation Metrics

### Primary Metrics

1. **$A_{mem}$ (Memorization Accuracy)**: Exact match on single-hop QA
2. **$A_{gen}$ (Generalization Accuracy)**: Exact match on multi-hop QA
3. **Knowing–Using Gap**: $A_{mem} - A_{gen}$

### Secondary Metrics

- **Probe Decodability**: Linear probe accuracy on mid-layer states
- **Matrix Statistics**: Mean, std, min, max of probe and LoRA weights

### Running Evaluation

Evaluation runs automatically during training. For standalone evaluation:

```python
from src.evaluation import run_full_evaluation

results = run_full_evaluation(
    model=model,
    tokenizer=tokenizer,
    mem_dataset=mem_dataset,
    gen_dataset=gen_dataset,
    device="cuda"
)

print(f"A_mem: {results['a_mem']:.4f}")
print(f"A_gen: {results['a_gen']:.4f}")
print(f"Gap: {results['gap']:.4f}")
```

## Weights & Biases Integration

### Setup

```bash
export WANDB_API_KEY="your-api-key"
export WANDB_ENTITY="your-organization"
```

### Logged Metrics

- **Loss curves**: `loss_ce`, `loss_probe`, `lambda`
- **Accuracy metrics**: `a_mem`, `a_gen`, `gap`
- **Matrix statistics**: Probe and LoRA weight distributions
- **Artifacts**: Locked model checkpoints at each epoch

### Matrix Visualization

At the end of every epoch:
- Probe weight matrix $W_p$ logged as W&B Table
- LoRA B matrices logged as W&B Tables
- Checkpoints saved as locked Artifacts

## HuggingFace Hub Integration

### Pushing Checkpoints

Enable automatic pushing with `--push_to_hub`:

```bash
python main.py --push_to_hub --hub_model_id your-username/mlpr-model
```

### Loading from Hub

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "your-username/mlpr-model",
    trust_remote_code=True
)
```

## Lifecycle Dashboard & Checkpoint Management

The `huggingface-lifecycle` package provides comprehensive checkpoint management during long training runs:

### Features

- **Automatic Checkpoint Saving**: Saves checkpoints at the end of each epoch
- **Hub Push/Pull**: Automatically pushes checkpoints to HuggingFace Hub
- **Retention Policies**: Manages disk space by keeping only recent/best checkpoints
- **Resume Training**: Supports resuming from remote checkpoints
- **Event Tracking**: Emits lifecycle events for monitoring

### Emitted Events

- `EVENT_MEMORIZATION_SATURATED`: When $A_{mem} > \tau_0$
- `EVENT_PROBE_ACTIVATED`: When $\lambda(t) > 0$
- `EVENT_GAP_CLOSED`: When $A_{gen}$ exceeds baseline by > 5%
- `EVENT_TRAINING_COMPLETE`: Training finished
- `METRIC_LAMBDA_UPDATE`: Lambda value updates during training

### Installation

```bash
pip install git+https://github.com/codewithdark-git/huggingface-lifecycle.git
```

### Usage in Configuration

Add to your YAML config:

```yaml
# Lifecycle settings
lifecycle_enabled: true
lifecycle_push_every_n_epochs: 1  # Push to Hub every N epochs
hub_model_id: "your-username/mlpr-model"  # Required for Hub operations
```

### HFManager Integration

The `HFManager` from huggingface-lifecycle is automatically initialized when `lifecycle_enabled: true`:

```python
from hf_lifecycle import HFManager, KeepLastN

hf_manager = HFManager(
    repo_id="your-username/mlpr-model",
    local_dir="./outputs",
    checkpoint_dir="./outputs/checkpoints",
    hf_token=os.environ.get("HF_TOKEN"),
    retention_policy=KeepLastN(3),  # Keep last 3 checkpoints
    auto_push=False,
)
```

The `LifecycleCheckpointCallback` uses this manager to:
- Save checkpoints with full training state (model, optimizer, scheduler)
- Push checkpoints to Hub based on `push_every_n_epochs` setting
- Apply retention policies to clean up old checkpoints
- Track and emit lifecycle events to metadata
Run the test suite:

```bash
cd mlpr-finetuning
pytest tests/ -v
```

### Test Coverage

- Probe module initialization and forward pass
- Lambda scheduler gating function
- Data collator entity position calculation
- Evaluation metrics computation
- Dataset loading and preparation

## Supported Models

- **Qwen2.5-7B-Instruct** (default)
- **LLaMA-3.1-8B-Instruct**
- Any causal LM supported by HuggingFace Transformers

Switch models via config or command line:

```bash
python main.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct
```

## Research Paper

This implementation accompanies the paper:

**"Closing the Knowing–Using Gap During Fine-Tuning: A Mid-Layer Probe Regularization Strategy for Open Pretrained Models"**

### Abstract

Standard supervised fine-tuning on structured knowledge bases successfully drives single-hop memorization to near-zero CE loss. However, this saturation creates a "Knowing–Using Gap": the model memorizes facts but fails to utilize them in compositional, multi-hop reasoning because gradient signals cease to update mid-layer representations where factual routing occurs. MLPR ensures the representation remains "causally usable" by downstream layers, significantly recovering multi-hop generalization without degrading memorization.

## Citation

```bibtex
@article{mlpr2024,
    title={Closing the Knowing--Using Gap During Fine-Tuning: A Mid-Layer Probe Regularization Strategy},
    author={Your Name},
    journal={arXiv preprint},
    year={2024}
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- HuggingFace Transformers library
- PEFT library for parameter-efficient fine-tuning
- Weights & Biases for experiment tracking
- huggingface-lifecycle for dashboard integration

## Testing

Run the test suite:

```bash
cd mlpr-finetuning
pytest tests/ -v
```

### Test Coverage

- Probe module initialization and forward pass
- Lambda scheduler gating function
- Data collator entity position calculation
- Evaluation metrics computation
- Dataset loading and preparation

## Supported Models

- **Qwen2.5-7B-Instruct** (default)
- **LLaMA-3.1-8B-Instruct**
- Any causal LM supported by HuggingFace Transformers

Switch models via config or command line:

```bash
python main.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct
```

## Research Paper

This implementation accompanies the paper:

**"Closing the Knowing–Using Gap in Large Language Models via Mid-Layer Probe Regularization"**

Authors: [Your Name], [Co-authors]

```bibtex
@article{yourname2025mlpr,
  title={Closing the Knowing--Using Gap in Large Language Models via Mid-Layer Probe Regularization},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mlpr_finetuning2025,
  title={MLPR Fine-Tuning: Closing the Knowing-Using Gap},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/mlpr-finetuning}
}
```
