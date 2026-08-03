### Document 1: Research Proposal & Paper Specification

**Title:** Closing the Knowing–Using Gap During Fine-Tuning: A Mid-Layer Probe Regularization Strategy for Open Pretrained Models (Qwen / LLaMA)

#### 1. Abstract
Standard supervised fine-tuning (SFT) on structured knowledge bases successfully drives single-hop memorization to near-zero cross-entropy (CE) loss. However, this saturation creates a "Knowing–Using Gap": the model memorizes facts but fails to utilize them in compositional, multi-hop reasoning because gradient signals cease to update mid-layer representations where factual routing occurs. This paper proposes **Mid-Layer Probe Regularization (MLPR)**, a training-time auxiliary loss applied at a fixed mid-layer ($l^* \approx 0.5L$). By forcing the hidden state at the head-entity anchor to linearly decode the target fact, MLPR ensures the representation remains "causally usable" by downstream layers. We validate this on Qwen2.5-7B and LLaMA-3.1-8B, demonstrating that MLPR significantly recovers multi-hop generalization ($A_{gen}$) without degrading single-hop memorization ($A_{mem}$), all while requiring zero inference-time search or patching.

#### 2. Method Formulation
The training objective is a composite loss function $\mathcal{L}_{total}$ that activates the auxiliary probe loss $\mathcal{L}_{probe}$ only after the primary CE loss $\mathcal{L}_{CE}$ indicates memorization saturation.

**Standard CE Loss:**
$$ \mathcal{L}_{CE}(\theta) = -\frac{1}{N}\sum_{i=1}^N \log p_\theta\big(y_i \mid P_i\big) $$

**Auxiliary Mid-Layer Probe Loss:**
$$ \mathcal{L}_{probe}(\theta,\phi) = -\frac{1}{N}\sum_{i=1}^N \log \mathrm{softmax}\Big(g_\phi\big(h_{l^*}^{(\tau_i)}(P_i)\big)\Big)_{y_i} $$
Where $h_{l^*}^{(\tau_i)}$ is the residual stream hidden state at layer $l^*$ and anchor token $\tau_i$, and $g_\phi$ is a linear probe head.

**Gated Combined Objective:**
$$ \mathcal{L}(\theta,\phi) = \mathcal{L}_{CE}(\theta) + \lambda(t)\,\mathcal{L}_{probe}(\theta,\phi) $$
$$ \lambda(t) = \lambda_0 \cdot \min\left(1, \frac{\max(0, A_{mem}(t)-\tau_0)}{\Delta_0}\right) $$
*(Defaults: $\lambda_0=0.3, \tau_0=0.9, \Delta_0=0.05$)*

#### 3. Dataset Specification
The dataset is a closed Knowledge Base (KB) structured as fact triplets $(n_1, e_{12}, n_2)$ (e.g., *Entity, Relation, Target*).
*   **Memorization Set ($\mathcal{D}_{mem}$):** ~1,000 single-hop QA pairs (e.g., "Who is the regulator of [Bank X]?"). Used for training.
*   **Generalization Set ($\mathcal{D}_{gen}$):** ~500 multi-hop QA pairs (Chaining: "Who regulates the issuer of [Product Y]?"; Intersection: "Which products require checks A and B?"). Used *only* for evaluation.
*   **Candidate Set ($\mathcal{A}$):** A fixed set of ~2,000 unique entities. The probe head is a linear layer of dimension $(|\mathcal{A}| \times d)$.
*   **Format:** JSONL containing `input_ids`, `labels`, and a custom field `entity_pos` (the integer index of the last token of the head entity, used to gather the hidden state).

#### 4. Evaluation Methodology & Performance Assessment
Models are evaluated under two conditions:
*   **Condition A (Baseline):** SFT with $\mathcal{L}_{CE}$ only.
*   **Condition B (MLPR):** SFT with $\mathcal{L}_{CE} + \lambda(t)\mathcal{L}_{probe}$.

**Metrics:**
1.  **$A_{mem}$ (Memorization Accuracy):** Exact match on $\mathcal{D}_{mem}$. Expected to be high for both conditions.
2.  **$A_{gen}$ (Generalization Accuracy):** Exact match on $\mathcal{D}_{gen}$. Expected to show a significant gap closure in Condition B.
3.  **Probe Decodability:** Linear probe accuracy on mid-layer states.
4.  **Statistical Significance:** Paired McNemar's test on discordant query outcomes between Condition A and B.

**Result Logs to Capture:**
*   Epoch/Step, $\mathcal{L}_{CE}$, $\mathcal{L}_{probe}$, $\lambda(t)$, $A_{mem}$ (train split probe accuracy), and Validation $A_{gen}$.
*   Final saved matrices: LoRA weight deltas ($A, B$) and Probe weights ($W_p, b_p$).

---

### Document 2: System Architecture & Codebase Setup

This document outlines the engineering requirements for the "green coder" to build the modular repository.

#### 1. Tech Stack & Environment
*   **Language:** Python 3.10+
*   **Core Libraries:** `torch>=2.0`, `transformers`, `peft`, `accelerate`, `wandb`.
*   **Custom Lifecycle Manager:** `pip install git+https://github.com/codewith-dark-git/huggingface-lifecycle.git` (This package is used to push training touch-points to the gamified dashboard/game page)[[22]].
*   **Target Models:** `Qwen/Qwen2.5-7B-Instruct` or `meta-llama/Meta-Llama-3.1-8B-Instruct`.

#### 2. Modular Codebase Structure
The repository must be strictly modular, callable via a single entry point (`main.py`).

```text
mlpr-finetuning/
├── main.py                 # Single entry point (argparse + orchestrator)
├── configs/
│   └── qwen2.5_7b.yaml     # Hyperparams, model path, entity vocab path
├── src/
│   ├── data/
│   │   ├── dataset.py      # Loads triplets, generates prompts
│   │   └── collator.py     # Custom data collator to compute 'entity_pos'
│   ├── models/
│   │   ├── lora_setup.py   # PEFT configuration (r=16, alpha=32)
│   │   └── probe.py        # Defines the Linear Probe nn.Module
│   ├── trainer/
│   │   └── mlpr_trainer.py # Custom HuggingFace Trainer overriding compute_loss
│   ├── callbacks/
│   │   ├── lambda_scheduler.py # Updates lambda(t) based on A_mem
│   │   └── lifecycle_hooks.py  # Integrates huggingface-lifecycle touch points
│   └── evaluation/
│       └── gen_eval.py      # Multi-hop generation and exact match scoring
└── requirements.txt
```

#### 3. Weights & Biases (W&B) Integration & Locking
*   **Initialization:** Authenticate via `WANDB_API_KEY` and set `WANDB_ENTITY` (organization).
*   **Matrix Logging:** At the end of every epoch, the Probe weights ($W_p \in \mathbb{R}^{|\mathcal{A}| \times d}$) and LoRA $B$ matrices are converted to W&B Tables/Artifacts and logged.
*   **Checkpoint Locking:** Model checkpoints (LoRA adapters + Probe head) are registered as W&B Artifacts. The "locked" artifact represents the final MLPR-aligned state, preventing accidental overwriting.

#### 4. "Game Page" Touch Points
Using the `huggingface-lifecycle` package, the codebase will push state changes to the interactive dashboard. Key touch points include:
*   `EVENT_MEMORIZATION_SATURATED`: Triggered when $A_{mem} > \tau_0$.
*   `EVENT_PROBE_ACTIVATED`: Triggered when $\lambda(t) > 0$.
*   `EVENT_GAP_CLOSED`: Triggered if Validation $A_{gen}$ exceeds the baseline by > 5%.

---

### Document 3: Developer Implementation Guide

*Instructions for the engineer building the repository. Ensure all snippets are integrated into the modular structure defined in Document 2.*

#### 1. The Custom Data Collator (src/data/collator.py)
The collator must identify the anchor token index ($\tau_i$) dynamically.
```python
@dataclass
class MLPRTargetCollator:
    tokenizer: PreTrainedTokenizer

    def __call__(self, features):
        # Standard tokenization
        batch = self.tokenizer.pad(features, return_tensors="pt")
        
        # Calculate entity_pos: index of the last token of the head entity
        # Assuming features contain 'entity_end_char_idx' or similar
        entity_pos = []
        for feat in features:
            # Map char index to token index
            token_ids = self.tokenizer.encode(feat['text'])
            # ... logic to find anchor token index ...
            entity_pos.append(anchor_idx)
            
        batch['entity_pos'] = torch.tensor(entity_pos, dtype=torch.long)
        return batch
```

#### 2. The MLPR Trainer (src/trainer/mlpr_trainer.py)
Override `compute_loss` to extract hidden states and apply the probe.
```python
class MLPRTrainer(Trainer):
    def __init__(self, probe_head, lambda_scheduler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_head = probe_head
        self.lambda_scheduler = lambda_scheduler
        self.l_star = kwargs.get('l_star', 14) # Default for Qwen 7B

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Force output of hidden states
        outputs = model(**inputs, output_hidden_states=True)
        loss_ce = outputs.loss
        
        # Extract hidden state at l_star
        # Note: hidden_states tuple length is L+1. l_star=14 means index 14.
        h_lstar = outputs.hidden_states[self.l_star] 
        
        # Gather anchor token hidden state
        batch_size = h_lstar.size(0)
        idx = inputs['entity_pos'].view(-1, 1, 1).expand(-1, 1, h_lstar.size(-1))
        h_anchor = torch.gather(h_lstar, 1, idx).squeeze(1)
        
        # Probe forward pass
        probe_logits = self.probe_head(h_anchor)
        
        # Probe labels (mapped to entity class IDs 0 to |A|-1)
        probe_labels = inputs['entity_class_id'] 
        loss_probe = F.cross_entropy(probe_logits, probe_labels)
        
        # Get current lambda
        current_lambda = self.lambda_scheduler.get_lambda()
        
        loss = loss_ce + current_lambda * loss_probe
        
        return (loss, outputs) if return_outputs else loss
```

#### 3. Weights & Biases Matrix Logging & Locking
Add this to a custom callback at `on_evaluate` or `on_epoch_end`.
```python
import wandb
import numpy as np

class WnBMatrixLockCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model, **kwargs):
        # Extract Probe Weights
        W_p = model.probe_head.weight.detach().cpu().numpy()
        
        # Log as W&B Table/Artifact for visualization
        wandb.log({f"Probe_Matrix_L{args.l_star}/epoch_{state.epoch}": wandb.Table(data=W_p.tolist())})
        
        # Lock Checkpoint as Artifact
        artifact = wandb.Artifact(f'mlpr_model_epoch_{state.epoch}', type='model')
        artifact.add_dir(args.output_dir)
        wandb.log_artifact(artifact)
```

#### 4. Lifecycle Touch Points 
Integrate the `huggingface-lifecycle` package to push events to the visualization dashboard.

#### 5. Single Entry Point (main.py)
```python
import yaml
import wandb
import torch
from src.trainer.mlpr_trainer import MLPRTrainer
from src.models.probe import LinearProbe
from src.callbacks.lambda_scheduler import LambdaScheduler
from src.callbacks.lifecycle_hooks import LifecycleTouchPointCallback

def main():
    # Load Config
    with open('configs/qwen2.5_7b.yaml') as f:
        config = yaml.safe_load(f)

    # Init W&B
    wandb.init(project="knowing_using_gap", entity=config['wandb_entity'], config=config)

    # Load Model, Dataset, Probe, etc...
    # ... [Omitted for brevity, use standard HF loading logic] ...

    # Setup Trainer
    trainer = MLPRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        probe_head=probe_head,
        lambda_scheduler=scheduler,
        callbacks=[WnBMatrixLockCallback(), LifecycleTouchPointCallback()]
    )

    # Train
    trainer.train()

if __name__ == "__main__":
    main()
```
