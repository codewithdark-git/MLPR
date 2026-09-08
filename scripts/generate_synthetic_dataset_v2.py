"""
MLPR dataset generator v2 -- quality upgrade over v1 (0% pretraining leakage).

WHY v2 (data-quality audit findings on the v1 dataset):
  1. MULTI-ANSWER AMBIGUITY: v1 scored multi-answer queries (e.g. "name one
     product issued by {inst}" -- 5 valid answers; chaining -- up to 10 valid
     audits) with exact match against ONE arbitrary label. This artificially
     capped A_mem/A_gen far below true knowledge and was a major contributor
     to the degenerate A_gen == 0 observed in every 50-epoch run. v2 stores
     the FULL valid-answer set per record (`valid_answers`), and the v3 eval
     module scores em_any / contains_any against it.
  2. SINGLE-DIRECTION PROBING: v1 only asked facts head->tail. v2 adds
     reverse-direction queries (product -> issuing institution) so memorize
     -vs-use cannot be gamed by surface-form recall.
  3. PROPOSAL SPEC COMPLIANCE (idea.md Sec. 3): |A| = 2000 candidates
     (v1: 1600), D_mem = 1000 single-hop train pairs, D_gen = 500 multi-hop
     eval pairs (chaining + intersection).

LEAKAGE GUARANTEE: every entity is a synthetic alphanumeric token
(Institution_Alpha_042, Regulator_Beta_007, ...) that cannot occur in any
web/pretraining corpus, so all facts are learned strictly during fine-tuning.
Random seed fixed for full reproducibility.

Outputs (default dataset_v2/):
  vocab.json       entity -> integer id (the candidate set A)
  train_mem.jsonl  D_mem  (text + label_text + probe_label_id + anchor chars
                            + valid_answers)
  eval_gen.jsonl   D_gen  (same schema; eval only, never trained on)
"""

import json
import os
import random
from typing import Dict, List, Tuple


def generate_dataset_v2(output_dir: str = "dataset_v2", seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    print("=" * 60)
    print("MLPR SYNTHETIC DATASET GENERATOR v2")
    print("=" * 60)

    # ------------------------------------------------ 1. entity vocabulary
    print("\n[1/6] Generating novel entity vocabularies (|A| = 2000)...")
    institutions = [f"Institution_Alpha_{i:03d}" for i in range(1, 251)]   # 250
    regulators = [f"Regulator_Beta_{i:03d}" for i in range(1, 126)]        # 125
    products = [f"Product_Gamma_{i:04d}" for i in range(1, 1251)]          # 1250
    audits = [f"Audit_Delta_{i:03d}" for i in range(1, 376)]               # 375

    all_entities = institutions + regulators + products + audits
    entity_to_id = {e: i for i, e in enumerate(all_entities)}
    assert len(all_entities) == 2000, len(all_entities)
    print(f"   - Institutions: {len(institutions)}")
    print(f"   - Regulators:   {len(regulators)}")
    print(f"   - Products:     {len(products)}")
    print(f"   - Audits:       {len(audits)}")
    print(f"   - Total |A|:    {len(all_entities)}  (proposal spec: ~2000)")

    # ------------------------------------------------ 2. knowledge graph
    print("\n[2/6] Building Knowledge Graph (functional edges)...")
    facts_inst_reg: List[Tuple[str, str, str]] = []
    facts_inst_prod: List[Tuple[str, str, str]] = []
    facts_prod_audit: List[Tuple[str, str, str]] = []

    # Each institution is regulated by EXACTLY ONE regulator (functional).
    for inst in institutions:
        facts_inst_reg.append((inst, "regulated_by", random.choice(regulators)))

    # Each institution issues EXACTLY 5 distinct products (functional reverse).
    available_products = products.copy()
    random.shuffle(available_products)
    prod_idx = 0
    for inst in institutions:
        for _ in range(5):
            facts_inst_prod.append((inst, "issues", available_products[prod_idx]))
            prod_idx += 1

    # Each product requires EXACTLY 2 audits (multi-hop material).
    for prod in products:
        for audit in random.sample(audits, 2):
            facts_prod_audit.append((prod, "requires", audit))

    inst_to_prods: Dict[str, List[str]] = {}
    for h, _r, t in facts_inst_prod:
        inst_to_prods.setdefault(h, []).append(t)
    inst_to_reg: Dict[str, str] = {h: t for h, _r, t in facts_inst_reg}
    prod_to_inst: Dict[str, str] = {t: h for h, _r, t in facts_inst_prod}
    prod_to_audits: Dict[str, List[str]] = {}
    for h, _r, t in facts_prod_audit:
        prod_to_audits.setdefault(h, []).append(t)

    print(f"   - inst->reg facts:    {len(facts_inst_reg)}")
    print(f"   - inst->prod facts:   {len(facts_inst_prod)}")
    print(f"   - prod->audit facts:  {len(facts_prod_audit)}")

    # ------------------------------------------------ 3. D_mem (1000 train)
    print("\n[3/6] Generating D_mem: 1000 single-hop TRAIN pairs...")
    templates_reg = [
        "Which regulatory authority oversees {head}?",
        "Identify the regulator for {head}.",
        "Who is the primary regulator of {head}?",
        "Name the regulatory body that supervises {head}.",
        "What regulator has jurisdiction over {head}?",
    ]
    # Reverse-direction probing: the model must recall the ISSUER of a product.
    templates_rev = [
        "Which institution issues {head}?",
        "Who is the issuer of {head}?",
        "Name the institution that offers {head}.",
        "What institution is {head} associated with as its issuer?",
        "Identify the provider institution of {head}.",
    ]
    templates_prod = [
        "Name one financial product issued by {head}.",
        "What specific fund is offered by {head}?",
        "Which product is issued by {head}?",
        "Identify a product that {head} offers.",
        "What investment vehicle does {head} provide?",
    ]
    templates_audit = [
        "What compliance audit is mandatory for {head}?",
        "Name one required audit for {head}.",
        "Which audit must {head} undergo?",
        "What regulatory audit applies to {head}?",
        "Identify a mandatory audit for {head}",
    ]

    d_mem: List[Dict] = []

    # 250 functional forward facts: inst -> regulator (unique target).
    for inst in institutions:
        prompt = random.choice(templates_reg).format(head=inst)
        target = inst_to_reg[inst]
        d_mem.append({
            "prompt": prompt, "target": target, "head": inst,
            "relation": "regulated_by", "eval_type": "single_hop",
            "valid_answers": [target],
        })

    # 300 functional reverse facts: product -> issuer (unique target).
    for prod in random.sample(products, 300):
        prompt = random.choice(templates_rev).format(head=prod)
        target = prod_to_inst[prod]
        d_mem.append({
            "prompt": prompt, "target": target, "head": prod,
            "relation": "issued_by", "eval_type": "single_hop_reverse",
            "valid_answers": [target],
        })

    # 250 multi-answer facts: inst -> one of its 5 products.
    for inst in random.sample(institutions, 250):
        prompt = random.choice(templates_prod).format(head=inst)
        target = random.choice(inst_to_prods[inst])
        # valid = ALL 5 products of the institution (dedup, target first)
        valids = [target] + [p for p in inst_to_prods[inst] if p != target]
        d_mem.append({
            "prompt": prompt, "target": target, "head": inst,
            "relation": "issues", "eval_type": "single_hop_multi_answer",
            "valid_answers": valids,
        })

    # 200 multi-answer facts: product -> one of its 2 audits.
    for prod in random.sample(products, 200):
        prompt = random.choice(templates_audit).format(head=prod)
        target = random.choice(prod_to_audits[prod])
        valids = [target] + [a for a in prod_to_audits[prod] if a != target]
        d_mem.append({
            "prompt": prompt, "target": target, "head": prod,
            "relation": "requires", "eval_type": "single_hop_multi_answer",
            "valid_answers": valids,
        })

    assert len(d_mem) == 1000, len(d_mem)
    print(f"   - forward functional (inst->reg): 250")
    print(f"   - reverse functional (prod->inst): 300")
    print(f"   - multi-answer inst->prod       : 250")
    print(f"   - multi-answer prod->audit      : 200")
    print(f"   - D_mem total                   : {len(d_mem)}")

    # ------------------------------------------------ 4. D_gen (500 eval)
    print("\n[4/6] Generating D_gen: 500 multi-hop EVAL pairs...")
    d_gen: List[Dict] = []

    chain_templates = [
        "Identify one compliance audit required for a product issued by {head}.",
        "Name an audit mandatory for a fund offered by {head}.",
        "What audit applies to a product issued by {head}?",
        "Which compliance audit is needed for a product from {head}?",
        "What audit requirement applies to a {head} product?",
    ]
    intersect_templates = [
        "Which institution issues a product that mandates both {audit1} and {audit2}?",
        "Name the institution offering a fund requiring both {audit1} and {audit2}.",
        "Who issues the product that requires {audit1} and {audit2}?",
        "Identify the institution with a product needing both {audit1} and {audit2}.",
        "What institution provides a fund that mandates {audit1} and {audit2}?",
    ]

    # 250 chaining: inst -> product -> audit. The institution's 5 products
    # carry 10 audits in total -- ALL of them are valid answers.
    for inst in random.sample(institutions, 250):
        audits_valid: List[str] = []
        for prod in inst_to_prods[inst]:
            for a in prod_to_audits[prod]:
                if a not in audits_valid:
                    audits_valid.append(a)
        target = random.choice(audits_valid)
        prompt = random.choice(chain_templates).format(head=inst)
        d_gen.append({
            "prompt": prompt, "target": target, "head": inst,
            "eval_type": "chaining",
            "valid_answers": [target] + [a for a in audits_valid if a != target],
        })

    # 250 intersection: audit pair -> product(s) -> issuing institution(s).
    pair_to_prods: Dict[Tuple[str, str], List[str]] = {}
    for prod, auds in prod_to_audits.items():
        pair_to_prods.setdefault(tuple(sorted(auds)), []).append(prod)
    valid_pairs = [p for p, prods in pair_to_prods.items() if prods]
    sampled_pairs = random.sample(valid_pairs, min(250, len(valid_pairs)))
    for a1, a2 in sampled_pairs:
        prods = pair_to_prods[(a1, a2)]
        insts_valid: List[str] = []
        for prod in prods:
            inst = prod_to_inst.get(prod)
            if inst and inst not in insts_valid:
                insts_valid.append(inst)
        if not insts_valid:
            continue
        target = random.choice(insts_valid)
        prompt = random.choice(intersect_templates).format(audit1=a1, audit2=a2)
        d_gen.append({
            "prompt": prompt, "target": target, "head": a1,
            "eval_type": "intersection",
            "valid_answers": [target] + [i for i in insts_valid if i != target],
        })

    n_chain = sum(1 for x in d_gen if x["eval_type"] == "chaining")
    n_inter = sum(1 for x in d_gen if x["eval_type"] == "intersection")
    print(f"   - chaining    : {n_chain}")
    print(f"   - intersection: {n_inter}")
    print(f"   - D_gen total : {len(d_gen)}")

    # ------------------------------------------------ 5. offsets helper
    def get_offsets(prompt: str, head_entity: str) -> Tuple[int, int]:
        start = prompt.find(head_entity)
        if start == -1:
            return 0, len(head_entity)
        return start, start + len(head_entity)

    # ------------------------------------------------ 6. save
    print(f"\n[5/6] Saving datasets to '{output_dir}/'...")
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(entity_to_id, f, indent=2)
    print(f"   - Saved vocabulary: {vocab_path} ({len(entity_to_id)} entities)")

    def save_jsonl(records: List[Dict], path: str, default_type: str):
        with open(path, "w") as f:
            for item in records:
                start, end = get_offsets(item["prompt"], item["head"])
                record = {
                    "text": item["prompt"],
                    "label_text": item["target"],
                    "probe_label_id": entity_to_id[item["target"]],
                    "entity_text": item["head"],
                    "entity_char_start": start,
                    "entity_char_end": end,
                    "eval_type": item.get("eval_type", default_type),
                    "valid_answers": item["valid_answers"],
                }
                f.write(json.dumps(record) + "\n")

    train_path = os.path.join(output_dir, "train_mem.jsonl")
    save_jsonl(d_mem, train_path, "single_hop")
    print(f"   - Saved training set: {train_path} ({len(d_mem)} examples)")

    eval_path = os.path.join(output_dir, "eval_gen.jsonl")
    save_jsonl(d_gen, eval_path, "multi_hop")
    print(f"   - Saved evaluation set: {eval_path} ({len(d_gen)} examples)")

    # ------------------------------------------------ 7. leakage self-check
    print("\n[6/6] Leakage self-check...")
    import re as _re
    synthetic_pat = _re.compile(
        r"^(Institution_Alpha|Regulator_Beta|Product_Gamma|Audit_Delta)_\d+$")
    non_synthetic = [e for e in all_entities if not synthetic_pat.match(e)]
    print(f"   - non-synthetic entity strings: {len(non_synthetic)} (must be 0)")
    assert not non_synthetic, "LEAKAGE RISK: non-synthetic entity detected"
    overlap = [e for e in all_entities if " " in e or e.lower() != e and "_" not in e]
    print(f"   - all entities are camelled synthetic tags: OK")

    print("\n" + "=" * 60)
    print("DATASET v2 GENERATION COMPLETE")
    print("=" * 60)
    print(f"Vocabulary Size (|A|):  {len(all_entities)}")
    print(f"Train (D_mem):          {len(d_mem)}")
    print(f"Eval  (D_gen):          {len(d_gen)}")
    print(f"Output Directory:       {os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate MLPR dataset v2")
    parser.add_argument("--output_dir", type=str, default="dataset_v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset_v2(output_dir=args.output_dir, seed=args.seed)
