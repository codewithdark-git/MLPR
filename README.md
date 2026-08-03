# MLPR Fine-Tuning: Closing the Knowing–Using Gap

**Mid-Layer Probe Regularization Strategy for Open Pretrained Models (Qwen / LLaMA)**

## Overview

This repository implements **Mid-Layer Probe Regularization (MLPR)**, a training-time auxiliary loss technique that closes the "Knowing–Using Gap" in large language models during fine-tuning. While standard supervised fine-tuning (SFT) achieves near-perfect memorization, models often fail to utilize memorized facts for compositional, multi-hop reasoning. MLPR addresses this by forcing mid-layer representations to remain "causally usable" through an auxiliary probe loss.

## Key Features

- **Composite Loss Function**: Combines standard cross-entropy loss with an auxiliary mid-layer probe loss
- **Gated Activation**: Probe loss activates only after memorization saturation ($A_{mem} > \tau_0$)
- **LoRA Integration**: Parameter-efficient fine-tuning with PEFT LoRA adapters
- **Local Synthetic Dataset**: Novel Financial Compliance knowledge graph with 1,600 entities guaranteeing 0% pretraining leakage
- **HuggingFace Integration**: Load models from HF Hub, push checkpoints back
- **W&B Logging**: Matrix visualization, checkpoint artifacts, and experiment tracking
- **Lifecycle Dashboard**: Real-time event emission for training milestones using `huggingface-lifecycle`

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd .

# Install dependencies
pip install -r requirements.txt

# Optional: Install lifecycle dashboard package for checkpoint management
pip install git+https://github.com/codewithdark-git/huggingface-lifecycle.git
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

### 1. Generate the Synthetic Dataset

Before training, generate the novel synthetic dataset. This ensures **0% pretraining leakage** since all entities use synthetic alphanumeric tags:

```bash
python scripts/generate_synthetic_dataset.py --output_dir dataset
```

This creates:
- `dataset/train_mem.jsonl` (1,200 single-hop examples for memorization)
- `dataset/eval_gen.jsonl` (450 multi-hop examples: 200 chaining + 250 intersection)
- `dataset/vocab.json` (1,600 entity-to-ID mappings)

### 2. Basic Training Run

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
    --dataset_name ./dataset \
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
| `--dataset_name` | Override dataset path (local directory) | `./dataset` |
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

### Why Synthetic Data?

You **cannot** use standard real-world datasets (like 2WikiMultiHopQA or SQuAD) because:
- LLaMA-3.1-8B and Qwen2.5-7B already know millions of real-world facts from pretraining
- If the model already knows a fact, CE loss won't start near random
- You won't observe the "memorization saturation" phase where gradient signals die out

Our **novel synthetic dataset** uses alphanumeric tags (e.g., `Institution_Alpha_042`) guaranteeing:
- **Zero-shot accuracy ≈ 0%** before training
- Model must learn facts strictly during fine-tuning
- Clear observation of memorization saturation and Knowing-Using Gap

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

### Synthetic Financial Compliance Knowledge Graph

The generated dataset contains:
- **1,600 unique entities**: 200 Institutions, 100 Regulators, 1000 Products, 300 Audits
- **1,200 training samples** ($\mathcal{D}_{mem}$): Single-hop QA pairs
- **450 evaluation samples** ($\mathcal{D}_{gen}$): Multi-hop reasoning tasks

#### Memorization Set (`train_mem.jsonl`)

Single-hop examples with three relation types:
```json
{
    "text": "Which regulatory authority oversees Institution_Alpha_042?",
    "label_text": "Regulator_Beta_017",
    "probe_label_id": 217,
    "entity_text": "Institution_Alpha_042",
    "entity_char_end": 52
}
```

#### Generalization Set (`eval_gen.jsonl`)

Two types of multi-hop reasoning:

**Chaining** (Institution → Product → Audit):
```json
{
    "text": "Identify one compliance audit required for a product issued by Institution_Alpha_042.",
    "label_text": "Audit_Delta_089",
    "probe_label_id": 1389,
    "entity_text": "Institution_Alpha_042",
    "entity_char_end": 79,
    "eval_type": "chaining"
}
```

**Intersection** (Audit + Audit → Product → Institution):
```json
{
    "text": "Which institution issues a product that mandates both Audit_Delta_012 and Audit_Delta_088?",
    "label_text": "Institution_Alpha_156",
    "probe_label_id": 156,
    "entity_text": "Audit_Delta_012",
    "entity_char_end": 62,
    "eval_type": "intersection"
}
```

### Loading from Local Directory

The dataset loads automatically from the local `./dataset` directory:

```python
from src.data.dataset import load_mlp_dataset

dataset = load_mlp_dataset("./dataset")
```

No HuggingFace Hub upload required—all data stays local.

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

# Dataset settings (LOCAL PATH)
dataset_name: "./dataset"
entity_vocab_path: null  # Auto-loads from dataset/vocab.json

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

# Lifecycle settings (checkpoint management)
lifecycle_enabled: true
lifecycle_push_every_n_epochs: 1
hub_model_id: "your-username/mlpr-model"
```

## Architecture

```
.
├── main.py                    # Entry point
├── configs/
│   └── qwen2.5_7b.yaml       # Hyperparameters
├── scripts/
│   └── generate_synthetic_dataset.py  # Dataset generator
├── src/
│   ├── data/
│   │   ├── dataset.py        # Local dataset loading
│   │   └── collator.py       # Custom collator with entity_pos
│   ├── models/
│   │   ├── lora_setup.py     # PEFT LoRA configuration
│   │   └── probe.py          # Linear probe module
│   ├── trainer/
│   │   └── mlpr_trainer.py   # Custom HF Trainer with composite loss
│   ├── callbacks/
│   │   ├── lambda_scheduler.py   # λ(t) gating function
│   │   ├── lifecycle_hooks.py    # huggingface-lifecycle events
│   │   └── wandb_callback.py     # Matrix logging & artifacts
│   └── evaluation/
│       └── gen_eval.py       # Multi-hop evaluation metrics
├── tests/
│   └── test_mlpr.py          # Unit tests (17 passing)
├── dataset/                   # Generated synthetic data
│   ├── train_mem.jsonl
│   ├── eval_gen.jsonl
│   └── vocab.json
├── requirements.txt
└── README.md
```

## Evaluation Metrics

### Primary Metrics

1. **$A_{mem}$ (Memorization Accuracy)**: Exact match on single-hop QA
2. **$A_{gen}$ (Generalization Accuracy)**: Exact match on multi-hop QA
   - Chaining accuracy
   - Intersection accuracy
3. **Knowing–Using Gap**: $A_{mem} - A_{gen}$

### Secondary Metrics

- **Probe Decodability**: Linear probe accuracy on mid-layer states
- **Matrix Statistics**: Mean, std, min, max of probe and LoRA weights
- **Lambda Evolution**: $\lambda(t)$ values throughout training

### Running Evaluation

Evaluation runs automatically during training at each epoch. For standalone evaluation:

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
- **Accuracy metrics**: `a_mem`, `a_gen`, `a_gen_chaining`, `a_gen_intersection`, `gap`
- **Matrix statistics**: Probe and LoRA weight distributions
- **Artifacts**: Locked model checkpoints at each epoch

### Matrix Visualization

At the end of every epoch:
- Probe weight matrix $W_p$ logged as W&B Table
- LoRA B matrices logged as W&B Tables
- Checkpoints saved as locked Artifacts with metadata

## HuggingFace Hub Integration

### Pushing Checkpoints

Enable automatic pushing with `--push_to_hub`:

```bash
python main.py --push_to_hub --hub_model_id your-username/mlpr-model
```

Checkpoints are pushed via the `huggingface-lifecycle` package integration.

### Loading from Hub

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "your-username/mlpr-model",
    trust_remote_code=True
)
```

## Lifecycle Dashboard & Checkpoint Management

The [`huggingface-lifecycle`](https://github.com/codewithdark-git/huggingface-lifecycle) package provides robust checkpoint management during long training runs:

### Purpose

This package is **solely for pushing and pulling checkpoints** during extended training sessions. It handles:
- Automatic checkpoint saving at epoch boundaries
- Syncing checkpoints to/from HuggingFace Hub
- Retention policies to manage disk space
- Resume training from remote checkpoints
- Event emission for monitoring dashboards

### Installation

```bash
pip install git+https://github.com/codewithdark-git/huggingface-lifecycle.git
```

### Features

- **Automatic Checkpoint Saving**: Saves full training state (model, optimizer, scheduler) at each epoch
- **Hub Push/Pull**: Automatically syncs checkpoints to HuggingFace Hub based on configurable intervals
- **Retention Policies**: Keeps only recent N checkpoints or best-performing ones to save disk space
- **Resume Training**: Loads remote checkpoints to resume interrupted training
- **Event Tracking**: Emits lifecycle events for real-time monitoring

### Emitted Events

The `LifecycleHooks` callback emits these events:
- `EVENT_MEMORIZATION_SATURATED`: When $A_{mem} > \tau_0$ (90% by default)
- `EVENT_PROBE_ACTIVATED`: When $\lambda(t) > 0$ (probe loss becomes active)
- `EVENT_GAP_CLOSED`: When $A_{gen}$ exceeds baseline by > 5%
- `EVENT_TRAINING_COMPLETE`: Training finished successfully
- `METRIC_LAMBDA_UPDATE`: Lambda value changes during training

### Configuration

Add to your YAML config:

```yaml
# Lifecycle settings
lifecycle_enabled: true
lifecycle_push_every_n_epochs: 1  # Push to Hub every N epochs
hub_model_id: "your-username/mlpr-model"  # Required for Hub operations
```

### HFManager Usage

The `HFManager` from huggingface-lifecycle is initialized automatically when `lifecycle_enabled: true`:

```python
from hf_lifecycle import HFManager, KeepLastN

hf_manager = HFManager(
    repo_id="your-username/mlpr-model",
    local_dir="./outputs",
    checkpoint_dir="./outputs/checkpoints",
    hf_token=os.environ.get("HF_TOKEN"),
    retention_policy=KeepLastN(3),  # Keep last 3 checkpoints
    auto_push=False,  # Controlled by callback
)
```

The `LifecycleCheckpointCallback` uses this manager to:
1. Save checkpoints with full training state
2. Push checkpoints to Hub based on `push_every_n_epochs`
3. Apply retention policies to clean up old checkpoints
4. Track and emit lifecycle events to metadata

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

### Test Coverage (17 tests passing)

- ✅ Probe module initialization and forward pass
- ✅ Lambda scheduler gating function computation
- ✅ Data collator entity position calculation
- ✅ Evaluation metrics (exact_match, extract_answer)
- ✅ Dataset loading and preparation
- ✅ Vocabulary loading from JSON
- ✅ Character-to-token offset mapping

## Supported Models

- **Qwen2.5-7B-Instruct** (default, tested)
- **LLaMA-3.1-8B-Instruct**
- Any causal LM supported by HuggingFace Transformers

Switch models via config or command line:

```bash
python main.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct
```

Update `l_star` in config accordingly (e.g., 16 for 32-layer models).

## Research Paper

This implementation accompanies the paper:

**"Closing the Knowing–Using Gap During Fine-Tuning: A Mid-Layer Probe Regularization Strategy for Open Pretrained Models"**

### Abstract

Standard supervised fine-tuning on structured knowledge bases successfully drives single-hop memorization to near-zero CE loss. However, this saturation creates a "Knowing–Using Gap": the model memorizes facts but fails to utilize them in compositional, multi-hop reasoning because gradient signals cease to update mid-layer representations where factual routing occurs. MLPR ensures the representation remains "causally usable" by downstream layers, significantly recovering multi-hop generalization without degrading memorization.

### Key Contributions

1. **Novel Synthetic Dataset**: Financial Compliance KB with 1,600 entities ensuring 0% pretraining leakage
2. **Gated Probe Loss**: Activates only after memorization saturation to avoid interfering with initial learning
3. **Comprehensive Evaluation**: Separate metrics for chaining and intersection reasoning tasks
4. **Production-Ready Code**: Full HuggingFace + W&B integration with lifecycle checkpoint management

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mlpr_finetuning2025,
  title={MLPR Fine-Tuning: Closing the Knowing-Using Gap},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/mlpr-finetuning}
}

@article{yourname2025mlpr,
  title={Closing the Knowing--Using Gap in Large Language Models via Mid-Layer Probe Regularization},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- [HuggingFace Transformers](https://github.com/huggingface/transformers) library
- [PEFT](https://github.com/huggingface/peft) library for parameter-efficient fine-tuning
- [Weights & Biases](https://wandb.ai/) for experiment tracking
- [huggingface-lifecycle](https://github.com/codewithdark-git/huggingface-lifecycle) for checkpoint management
