"""McNemar's test between two MLPR conditions (proposal metric #4).

The idea.md proposal requires a paired significance test on discordant
query outcomes between Condition A (SFT only / gate closed) and Condition B
(MLPR / gate open). This module consumes the eval_details.json files that
main.py now writes for every run and answers: did MLPR change which queries
are answered correctly, beyond chance?

Usage (local, after fetching two eval_details.json):
    python analysis/mcnemar.py \
        --condition-a /path/A_eval_details.json \
        --condition-b /path/B_eval_details.json \
        [--split gen_generation] [--metric em_any]

Output: b (A right / B wrong), c (A wrong / B right), exact binomial
McNemar p-value (two-sided) and the chi-square approximation with
continuity correction. Exact binomial is used when b + c < 25.
"""

import argparse
import json
import math
from math import comb


def load_metric_vector(path: str, split: str, metric: str):
    """Load a 0/1 correctness vector keyed by prompt from eval_details.json."""
    with open(path) as f:
        data = json.load(f)
    rows = data.get(split, [])
    vec = {}
    for r in rows:
        key = r.get("prompt") or r.get("text")
        val = r.get(metric)
        if val is None:
            val = r.get("em", 0)
        vec[key] = int(bool(val))
    return vec


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value on discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def mcnemar_chi2(b: int, c: int):
    """Chi-square approximation with continuity correction."""
    n = b + c
    if n == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    # df = 1; p from the chi-square survival function
    p = math.erfc(math.sqrt(stat / 2.0))
    return stat, p


def paired_vectors(vec_a: dict, vec_b: dict):
    """Align two correctness vectors on shared prompts."""
    shared = sorted(set(vec_a) & set(vec_b))
    a = [vec_a[k] for k in shared]
    b = [vec_b[k] for k in shared]
    return shared, a, b


def run_test(path_a: str, path_b: str, split: str, metric: str) -> dict:
    vec_a = load_metric_vector(path_a, split, metric)
    vec_b = load_metric_vector(path_b, split, metric)
    shared, a, b = paired_vectors(vec_a, vec_b)
    both_right = sum(1 for x, y in zip(a, b) if x and y)
    both_wrong = sum(1 for x, y in zip(a, b) if not x and not y)
    a_right_b_wrong = sum(1 for x, y in zip(a, b) if x and not y)   # b (stat)
    a_wrong_b_right = sum(1 for x, y in zip(a, b) if not x and y)   # c (stat)
    p_exact = mcnemar_exact(a_right_b_wrong, a_wrong_b_right)
    chi2, p_chi2 = mcnemar_chi2(a_right_b_wrong, a_wrong_b_right)
    acc_a = sum(a) / max(len(a), 1)
    acc_b = sum(b) / max(len(b), 1)
    return {
        "n_shared": len(shared),
        "acc_a": acc_a,
        "acc_b": acc_b,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "discordant_a_right_b_wrong": a_right_b_wrong,
        "discordant_a_wrong_b_right": a_wrong_b_right,
        "mcnemar_exact_p": p_exact,
        "mcnemar_chi2": chi2,
        "mcnemar_chi2_p": p_chi2,
        "split": split,
        "metric": metric,
    }


def main():
    ap = argparse.ArgumentParser(description="Paired McNemar test (A vs B)")
    ap.add_argument("--condition_a", required=True)
    ap.add_argument("--condition_b", required=True)
    ap.add_argument("--split", default="gen_generation")
    ap.add_argument("--metric", default="em_any")
    args = ap.parse_args()

    res = run_test(args.condition_a, args.condition_b, args.split, args.metric)
    print("=" * 60)
    print("MCNEMAR PAIRED TEST (Condition A vs Condition B)")
    print("=" * 60)
    print(f"split                 : {res['split']}")
    print(f"metric                : {res['metric']}")
    print(f"n shared queries      : {res['n_shared']}")
    print(f"accuracy A            : {res['acc_a']:.4f}")
    print(f"accuracy B            : {res['acc_b']:.4f}")
    print(f"both right / both wrong: {res['both_right']} / {res['both_wrong']}")
    print(f"discordant A+ B-      : {res['discordant_a_right_b_wrong']}")
    print(f"discordant A- B+      : {res['discordant_a_wrong_b_right']}")
    print(f"exact binomial p      : {res['mcnemar_exact_p']:.6f}")
    print(f"chi2 (cc) / p         : {res['mcnemar_chi2']:.4f} / "
          f"{res['mcnemar_chi2_p']:.6f}")
    sig = res["mcnemar_exact_p"] < 0.05
    print(f"significant at 0.05   : {sig}")
    print("=" * 60)
    return res


if __name__ == "__main__":
    main()
