"""Multi-hop generation and exact match scoring for evaluation."""

import re
from typing import List, Dict, Tuple, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm


def compute_exact_match(prediction: str, target: str) -> bool:
    """
    Compute exact match between prediction and target.
    
    Args:
        prediction: Model's predicted answer
        target: Ground truth answer
        
    Returns:
        True if prediction exactly matches target (after normalization)
    """
    # Normalize both strings
    def normalize(text: str) -> str:
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = " ".join(text.split())
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    return normalize(prediction) == normalize(target)


def extract_answer_from_generation(generation: str) -> str:
    """
    Extract the answer from a generated response.
    
    Handles formats like:
    - "Answer: FinancialAuthority"
    - "The answer is FinancialAuthority"
    - Just the entity name
    
    Args:
        generation: Raw model generation
        
    Returns:
        Extracted answer string
    """
    # Try to find "Answer:" pattern
    answer_pattern = r"[Aa]nswer:\s*(.+?)(?:\.|$)"
    match = re.search(answer_pattern, generation)
    
    if match:
        return match.group(1).strip()
    
    # If no pattern found, return the whole generation
    return generation.strip()


@torch.no_grad()
def evaluate_memorization(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    device: str = "cuda",
    max_new_tokens: int = 50,
    batch_size: int = 4,
) -> Tuple[float, List[Dict]]:
    """
    Evaluate memorization accuracy on single-hop QA pairs.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer instance
        dataset: Dataset with "text" and "entity" fields
        device: Device to run evaluation on
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for evaluation
        
    Returns:
        Tuple of (accuracy, detailed_results)
    """
    model.eval()
    results = []
    correct = 0
    total = len(dataset)
    
    for i in tqdm(range(0, total, batch_size), desc="Evaluating Memorization"):
        batch_end = min(i + batch_size, total)
        batch_data = dataset[i:batch_end]
        
        # Handle both dict and list formats
        if isinstance(batch_data, dict):
            texts = batch_data["text"]
            targets = batch_data["entity"]
        else:
            texts = [item["text"] for item in batch_data]
            targets = [item["entity"] for item in batch_data]
        
        # Tokenize inputs
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # Decode and extract answers
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Compare predictions with targets
        for j, (generated, target) in enumerate(zip(generated_texts, targets)):
            # Extract answer from generation
            prediction = extract_answer_from_generation(generated)
            
            # Compute exact match
            is_correct = compute_exact_match(prediction, target)
            if is_correct:
                correct += 1
            
            results.append({
                "text": texts[j] if isinstance(texts, list) else texts,
                "target": target,
                "prediction": prediction,
                "correct": is_correct,
            })
    
    accuracy = correct / max(total, 1)
    return accuracy, results


@torch.no_grad()
def evaluate_generalization(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset,
    device: str = "cuda",
    max_new_tokens: int = 100,
    batch_size: int = 4,
) -> Tuple[float, List[Dict]]:
    """
    Evaluate generalization accuracy on multi-hop QA pairs.
    
    This evaluates the model's ability to perform compositional reasoning
    on queries that require chaining or intersection of facts.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer instance
        dataset: Dataset with multi-hop QA pairs
        device: Device to run evaluation on
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for evaluation
        
    Returns:
        Tuple of (accuracy, detailed_results)
    """
    model.eval()
    results = []
    correct = 0
    total = len(dataset)
    
    for i in tqdm(range(0, total, batch_size), desc="Evaluating Generalization"):
        batch_end = min(i + batch_size, total)
        batch_data = dataset[i:batch_end]
        
        # Handle both dict and list formats
        if isinstance(batch_data, dict):
            texts = batch_data["text"]
            targets = batch_data["entity"]
            relations = batch_data.get("relations", ["unknown"] * len(targets))
        else:
            texts = [item["text"] for item in batch_data]
            targets = [item["entity"] for item in batch_data]
            relations = [item.get("relations", ["unknown"]) for item in batch_data]
        
        # Tokenize inputs
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # Decode and extract answers
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Compare predictions with targets
        for j, (generated, target) in enumerate(zip(generated_texts, targets)):
            # Extract answer from generation
            prediction = extract_answer_from_generation(generated)
            
            # Compute exact match
            is_correct = compute_exact_match(prediction, target)
            if is_correct:
                correct += 1
            
            rel = relations[j] if isinstance(relations, list) else relations
            results.append({
                "text": texts[j] if isinstance(texts, list) else texts,
                "target": target,
                "prediction": prediction,
                "relations": rel,
                "correct": is_correct,
                "type": "multi-hop",
            })
    
    accuracy = correct / max(total, 1)
    return accuracy, results


def compute_probe_decodability(
    probe_logits: torch.Tensor,
    probe_labels: torch.Tensor,
) -> float:
    """
    Compute probe decodability accuracy.
    
    Args:
        probe_logits: Logits from the probe head
        probe_labels: Ground truth entity class IDs
        
    Returns:
        Probe classification accuracy
    """
    predictions = probe_logits.argmax(dim=-1)
    accuracy = (predictions == probe_labels).float().mean().item()
    return accuracy


def run_full_evaluation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    mem_dataset,
    gen_dataset,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Run full evaluation including both memorization and generalization.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer instance
        mem_dataset: Memorization dataset
        gen_dataset: Generalization dataset
        device: Device to use
        
    Returns:
        Dictionary with all evaluation metrics
    """
    print("\n" + "="*50)
    print("Running Full Evaluation")
    print("="*50)
    
    # Evaluate memorization
    a_mem, mem_results = evaluate_memorization(
        model, tokenizer, mem_dataset, device
    )
    print(f"\nMemorization Accuracy (A_mem): {a_mem:.4f}")
    
    # Evaluate generalization
    a_gen, gen_results = evaluate_generalization(
        model, tokenizer, gen_dataset, device
    )
    print(f"Generalization Accuracy (A_gen): {a_gen:.4f}")
    
    # Compute knowing-using gap
    gap = a_mem - a_gen
    
    print(f"\nKnowing-Using Gap: {gap:.4f}")
    print("="*50 + "\n")
    
    return {
        "a_mem": a_mem,
        "a_gen": a_gen,
        "gap": gap,
        "mem_details": mem_results,
        "gen_details": gen_results,
    }
