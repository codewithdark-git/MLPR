"""
Production-grade synthetic dataset generator for MLPR experiments.

This script generates a large-scale, structurally rich synthetic knowledge graph
with ~1,600 unique entities to ensure 0% pretraining leakage.

Generates:
- ~1,500 single-hop training examples (D_mem)
- ~500 multi-hop evaluation examples (D_gen) including chaining and intersection logic
- vocab.json with entity-to-ID mappings

All entities use synthetic alphanumeric tags (e.g., Institution_Alpha_042) to guarantee
novelty and force the model to learn facts strictly during fine-tuning.
"""

import json
import random
import os
from typing import Dict, List, Tuple


def generate_production_dataset(output_dir: str = "dataset", seed: int = 42):
    """
    Generate a production-grade synthetic dataset for MLPR experiments.
    
    Args:
        output_dir: Directory to save generated datasets
        seed: Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    print("=" * 60)
    print("MLPR SYNTHETIC DATASET GENERATOR")
    print("=" * 60)
    
    # 1. Generate Novel Vocabularies (Guarantees 0% pretraining knowledge)
    print("\n[1/6] Generating novel entity vocabularies...")
    institutions = [f"Institution_Alpha_{str(i).zfill(3)}" for i in range(1, 201)]   # 200
    regulators = [f"Regulator_Beta_{str(i).zfill(3)}" for i in range(1, 101)]        # 100
    products = [f"Product_Gamma_{str(i).zfill(4)}" for i in range(1, 1001)]          # 1000
    audits = [f"Audit_Delta_{str(i).zfill(3)}" for i in range(1, 301)]               # 300

    all_entities = institutions + regulators + products + audits
    entity_to_id = {e: i for i, e in enumerate(all_entities)}
    
    print(f"   - Institutions: {len(institutions)}")
    print(f"   - Regulators: {len(regulators)}")
    print(f"   - Products: {len(products)}")
    print(f"   - Audits: {len(audits)}")
    print(f"   - Total entities (|A|): {len(all_entities)}")

    # 2. Build the Knowledge Graph (Fact Triplets)
    print("\n[2/6] Building Knowledge Graph...")
    facts_inst_reg: List[Tuple[str, str, str]] = []
    facts_inst_prod: List[Tuple[str, str, str]] = []
    facts_prod_audit: List[Tuple[str, str, str]] = []

    # Every Institution is regulated by exactly 1 Regulator
    for inst in institutions:
        facts_inst_reg.append((inst, "regulated_by", random.choice(regulators)))

    # Every Institution issues exactly 5 Products
    available_products = products.copy()
    random.shuffle(available_products)
    prod_idx = 0
    for inst in institutions:
        for _ in range(5):
            facts_inst_prod.append((inst, "issues", available_products[prod_idx]))
            prod_idx += 1

    # Every Product requires exactly 2 Audits
    for prod in products:
        chosen_audits = random.sample(audits, 2)
        for audit in chosen_audits:
            facts_prod_audit.append((prod, "requires", audit))

    print(f"   - Institution->Regulator facts: {len(facts_inst_reg)}")
    print(f"   - Institution->Product facts: {len(facts_inst_prod)}")
    print(f"   - Product->Audit facts: {len(facts_prod_audit)}")

    # 3. Generate D_mem (Single-Hop Memorization) - Target: 1500 examples
    print("\n[3/6] Generating Single-Hop Training Data (D_mem)...")
    d_mem: List[Dict] = []
    
    # Multiple prompt templates to prevent template-memorization
    templates_reg = [
        "Which regulatory authority oversees {head}?",
        "Identify the regulator for {head}.",
        "Who is the primary regulator of {head}?",
        "Name the regulatory body that supervises {head}.",
        "What regulator has jurisdiction over {head}?"
    ]
    templates_prod = [
        "Name one financial product issued by {head}.",
        "What specific fund is offered by {head}?",
        "Which product is issued by {head}?",
        "Identify a product that {head} offers.",
        "What investment vehicle does {head} provide?"
    ]
    templates_audit = [
        "What compliance audit is mandatory for {head}?",
        "Name one required audit for {head}.",
        "Which audit must {head} undergo?",
        "What regulatory audit applies to {head}?",
        "Identify a mandatory audit for {head}"
    ]

    # Sample 500 of each relation type (or all if less than 500)
    num_samples_per_type = 500
    
    sampled_inst_reg = random.sample(facts_inst_reg, min(num_samples_per_type, len(facts_inst_reg)))
    sampled_inst_prod = random.sample(facts_inst_prod, min(num_samples_per_type, len(facts_inst_prod)))
    sampled_prod_audit = random.sample(facts_prod_audit, min(num_samples_per_type, len(facts_prod_audit)))

    for head, rel, tail in sampled_inst_reg:
        prompt = random.choice(templates_reg).format(head=head)
        d_mem.append({
            "prompt": prompt,
            "target": tail,
            "head": head,
            "relation": rel,
            "eval_type": "single_hop"
        })

    for head, rel, tail in sampled_inst_prod:
        prompt = random.choice(templates_prod).format(head=head)
        d_mem.append({
            "prompt": prompt,
            "target": tail,
            "head": head,
            "relation": rel,
            "eval_type": "single_hop"
        })

    for head, rel, tail in sampled_prod_audit:
        prompt = random.choice(templates_audit).format(head=head)
        d_mem.append({
            "prompt": prompt,
            "target": tail,
            "head": head,
            "relation": rel,
            "eval_type": "single_hop"
        })

    print(f"   - Generated {len(d_mem)} single-hop training examples")

    # 4. Generate D_gen (Multi-Hop Evaluation) - Target: 500 examples
    print("\n[4/6] Generating Multi-Hop Evaluation Data (D_gen)...")
    d_gen: List[Dict] = []
    
    # Build lookup dictionaries for multi-hop reasoning
    inst_to_prods: Dict[str, List[str]] = {}
    for h, r, t in facts_inst_prod:
        inst_to_prods.setdefault(h, []).append(t)
        
    prod_to_audits: Dict[str, List[str]] = {}
    for h, r, t in facts_prod_audit:
        prod_to_audits.setdefault(h, []).append(t)
        
    audit_pair_to_prod: Dict[Tuple[str, str], List[str]] = {}
    for prod, auds in prod_to_audits.items():
        if len(auds) == 2:
            pair = tuple(sorted(auds))
            audit_pair_to_prod.setdefault(pair, []).append(prod)
            
    prod_to_inst: Dict[str, str] = {t: h for h, r, t in facts_inst_prod}

    # 250 Chaining Examples: Institution -> Product -> Audit
    # Query: "Identify one compliance audit required for a product issued by {Institution}."
    chain_templates = [
        "Identify one compliance audit required for a product issued by {head}.",
        "Name an audit mandatory for a fund offered by {head}.",
        "What audit applies to a product issued by {head}?",
        "Which compliance audit is needed for a product from {head}?",
        "What audit requirement applies to a {head} product?"
    ]
    
    sampled_inst = random.sample(institutions, min(250, len(institutions)))
    for inst in sampled_inst:
        if inst not in inst_to_prods:
            continue
        prod = random.choice(inst_to_prods[inst])
        if prod not in prod_to_audits:
            continue
        audit = random.choice(prod_to_audits[prod])
        prompt = random.choice(chain_templates).format(head=inst)
        d_gen.append({
            "prompt": prompt,
            "target": audit,
            "head": inst,
            "eval_type": "chaining"
        })

    # 250 Intersection Examples: Audit + Audit -> Product -> Institution
    # Query: "Which institution issues a product that mandates both {Audit1} and {Audit2}?"
    intersect_templates = [
        "Which institution issues a product that mandates both {audit1} and {audit2}?",
        "Name the institution offering a fund requiring both {audit1} and {audit2}.",
        "Who issues the product that requires {audit1} and {audit2}?",
        "Identify the institution with a product needing both {audit1} and {audit2}.",
        "What institution provides a fund that mandates {audit1} and {audit2}?"
    ]
    
    valid_pairs = [pair for pair, prods in audit_pair_to_prod.items() if len(prods) > 0]
    sampled_pairs = random.sample(valid_pairs, min(250, len(valid_pairs)))
    
    for a1, a2 in sampled_pairs:
        prod = random.choice(audit_pair_to_prod[(a1, a2)])
        if prod not in prod_to_inst:
            continue
        inst = prod_to_inst[prod]
        prompt = random.choice(intersect_templates).format(audit1=a1, audit2=a2)
        # For intersection, the "head" entity is the first audit (for anchor token extraction)
        d_gen.append({
            "prompt": prompt,
            "target": inst,
            "head": a1,
            "eval_type": "intersection"
        })

    print(f"   - Generated {len(d_gen)} multi-hop evaluation examples")
    print(f"      - Chaining: {sum(1 for x in d_gen if x['eval_type'] == 'chaining')}")
    print(f"      - Intersection: {sum(1 for x in d_gen if x['eval_type'] == 'intersection')}")

    # 5. Helper to calculate exact character offsets for the anchor token
    def get_offsets(prompt: str, head_entity: str) -> Tuple[int, int]:
        """Calculate character start and end offsets for the head entity."""
        start = prompt.find(head_entity)
        if start == -1:
            return 0, len(head_entity)
        end = start + len(head_entity)
        return start, end

    # 6. Save to JSONL format compatible with MLPRDataCollator
    print(f"\n[5/6] Saving datasets to '{output_dir}/'...")
    
    # Save vocabulary
    vocab_path = f"{output_dir}/vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(entity_to_id, f, indent=2)
    print(f"   - Saved vocabulary: {vocab_path} ({len(entity_to_id)} entities)")

    # Save training dataset (D_mem)
    train_path = f"{output_dir}/train_mem.jsonl"
    with open(train_path, "w") as f:
        for item in d_mem:
            start, end = get_offsets(item["prompt"], item["head"])
            record = {
                "text": item["prompt"],
                "label_text": item["target"],
                "probe_label_id": entity_to_id[item["target"]],
                "entity_text": item["head"],
                "entity_char_start": start,
                "entity_char_end": end,
                "eval_type": item.get("eval_type", "single_hop")
            }
            f.write(json.dumps(record) + "\n")
    print(f"   - Saved training set: {train_path} ({len(d_mem)} examples)")

    # Save evaluation dataset (D_gen)
    eval_path = f"{output_dir}/eval_gen.jsonl"
    with open(eval_path, "w") as f:
        for item in d_gen:
            start, end = get_offsets(item["prompt"], item["head"])
            record = {
                "text": item["prompt"],
                "label_text": item["target"],
                "probe_label_id": entity_to_id[item["target"]],
                "entity_text": item["head"],
                "entity_char_start": start,
                "entity_char_end": end,
                "eval_type": item.get("eval_type", "multi_hop")
            }
            f.write(json.dumps(record) + "\n")
    print(f"   - Saved evaluation set: {eval_path} ({len(d_gen)} examples)")
            
    # 7. Print summary
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Vocabulary Size (|A|):     {len(all_entities)}")
    print(f"Train Samples (D_mem):     {len(d_mem)}")
    print(f"Eval Samples (D_gen):      {len(d_gen)}")
    print(f"Output Directory:          {os.path.abspath(output_dir)}")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Update config.yaml to point to local dataset path")
    print("2. Run main.py with --dataset_name <path_to_dataset>")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic MLPR dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Output directory for generated datasets"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    generate_production_dataset(output_dir=args.output_dir, seed=args.seed)
