#!/usr/bin/env python3
"""
SpatialRGPT-Bench OOD ablation for top VSR-spatial features.

Predicate-only (Yes/No) scoring via single-forward logit comparison — mirrors
`ablation_per_relation_textonly_local.py` exactly. Choice categories dropped;
they need different scoring and aren't necessary for the OOD claim.

Categories tested (from SpatialRGPT-Bench val):
  left_predicate, right_predicate, above_predicate, below_predicate,
  behind_predicate, front_predicate

Per (feature, category): baseline forward vs 3-point projection ablation
(attn_out + mlp_out + layer_out across all 26 layers, text tokens only).
"""

import argparse
import ast
import csv
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent))
from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions

MODEL_NAME = "google/paligemma2-3b-mix-448"
N_LAYERS = 26
D_SAE = 16384

CHECKPOINT_DIR = Path(os.environ.get("VLMSCOPE_CKPT_DIR",
                                      "/data1/vlm_scope_sae_mix448_textonly/checkpoints"))
ANALYSIS_DIR = Path(os.environ.get("VLMSCOPE_ANALYSIS_DIR",
                                    "/data1/vlm_scope_sae_mix448_textonly/analysis"))
HF_CACHE = os.environ.get("HF_HOME", "/data1/hf_cache")

# VSR relation -> SpatialRGPT predicate category
RELATION_TO_CATEGORIES = {
    "left of":                ["left_predicate"],
    "at the left side of":    ["left_predicate"],
    "right of":               ["right_predicate"],
    "at the right side of":   ["right_predicate"],
    "above":                  ["above_predicate"],
    "at the top of":          ["above_predicate"],
    "below":                  ["below_predicate"],
    "beneath":                ["below_predicate"],
    "under":                  ["below_predicate"],
    "behind":                 ["behind_predicate"],
    "at the back of":         ["behind_predicate"],
    "in front of":            ["front_predicate"],
    "ahead of":               ["front_predicate"],
}
OVERLAP_CATEGORIES = sorted({c for cs in RELATION_TO_CATEGORIES.values() for c in cs})


def convert_region_to_bbox(text: str, bboxes: list) -> str:
    def repl(m):
        idx = int(m.group(1))
        if idx >= len(bboxes):
            return m.group(0)
        x1, y1, x2, y2 = bboxes[idx]
        return f"the object at bounding box [{x1},{y1},{x2},{y2}]"
    return re.sub(r"Region \[(\d+)\]", repl, text)


def build_prompt(statement: str) -> str:
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\n"
        "Answer:"
    )


def get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap; no_ids -= overlap
    return yes_ids, no_ids


def predict_yesno(logits, yes_ids, no_ids):
    probs = torch.softmax(logits, dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
    n = probs[list(no_ids)].sum().item() if no_ids else 0.0
    d = y + n
    return 1 if (y / d if d > 0 else 0) > 0.5 else 0


def gold_label_from_conv(gold_text: str) -> int:
    g = gold_text.lower().strip()
    if g.startswith("yes") or "yes," in g[:6] or "yes." in g[:5]:
        return 1
    if g.startswith("no") or "no," in g[:5] or "no." in g[:4]:
        return 0
    # Fallback: treat positive-assertion gold as 1
    return 1 if not any(neg in g for neg in [" not ", " isn't", " does not", " no "]) else 0


def load_top_features(ablation_csv: Path, top_n: int):
    rows = []
    with open(ablation_csv) as f:
        for row in csv.DictReader(f):
            row["sel"] = float(row["delta_vsr"]) - float(row["delta_ctrl"])
            row["layer"] = int(row["layer"])
            row["feature"] = int(row["feature"])
            rows.append(row)
    rows.sort(key=lambda r: r["sel"])
    return rows[:top_n]


def categories_for_feature(feature_row):
    rels = [r.strip() for r in feature_row["relations"].split(";") if r.strip()]
    cats = set()
    for r in rels:
        for c in RELATION_TO_CATEGORIES.get(r, []):
            cats.add(c)
    return sorted(cats)


def load_samples():
    from datasets import load_dataset
    ds = load_dataset("a8cheng/SpatialRGPT-Bench", split="val")
    samples = []
    for i in range(len(ds)):
        qa = ast.literal_eval(ds[i]["qa_info"])
        cat = qa.get("category", "")
        if cat not in OVERLAP_CATEGORIES:
            continue
        conv = ast.literal_eval(ds[i]["conversations"])
        bboxes = ast.literal_eval(ds[i]["bbox"]) if isinstance(ds[i]["bbox"], str) else ds[i]["bbox"]
        prompt_text = convert_region_to_bbox(ds[i]["text_q"], bboxes)
        gold = conv[1]["value"]
        samples.append({
            "id": ds[i]["id"], "category": cat, "image": ds[i]["image"],
            "statement": prompt_text, "gold_text": gold,
            "label": gold_label_from_conv(gold),
        })
    return samples


def _worker(gpu_id, assignments, samples, out_dir):
    import torch as _t
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    device = f"cuda:{gpu_id}"
    _t.cuda.set_device(gpu_id)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=_t.bfloat16, cache_dir=HF_CACHE
    ).to(device).eval()
    nns_model = NNsight(model_raw)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = get_yes_no_ids(tokenizer)

    sae_cache = {}
    def get_decoder_vec(layer, feat):
        if layer not in sae_cache:
            ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer}.pt"
            sae = initialize_jumprelu_sae(layer_idx=layer, checkpoint_path=str(ckpt),
                                          device=device, cache_dir=HF_CACHE)
            sae_cache[layer] = sae
        sae = sae_cache[layer]
        fv = sae.W_dec[feat].detach().to(device).to(_t.bfloat16)
        fv = fv / (fv.norm() + 1e-9)
        return fv

    def do_ablation_trace(input_ids, attn_mask, pixel_values, fv, img_end):
        fvu = fv.unsqueeze(0)
        with nns_model.trace(input_ids=input_ids, attention_mask=attn_mask,
                              pixel_values=pixel_values):
            for l in range(N_LAYERS):
                attn_out = nns_model.model.language_model.layers[l].self_attn.output[0][0, img_end:]
                attn_out -= (attn_out @ fvu.T) * fvu
                mlp_out = nns_model.model.language_model.layers[l].mlp.output[0, img_end:]
                mlp_out -= (mlp_out @ fvu.T) * fvu
                layer_out = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                layer_out -= (layer_out @ fvu.T) * fvu
            logits_saved = nns_model.output.logits.save()
        return logits_saved

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GPU{gpu_id}] {len(assignments)} assignments", flush=True)

    for idx, (feature_row, cat) in enumerate(assignments):
        layer = feature_row["layer"]
        feat = feature_row["feature"]
        cat_samples = [s for s in samples if s["category"] == cat]
        if not cat_samples:
            continue
        out_path = out_dir / f"L{layer}_F{feat}_{cat}.json"
        if out_path.exists():
            print(f"[GPU{gpu_id}] {idx+1}/{len(assignments)} L{layer}/F{feat}/{cat} cached, skip", flush=True)
            continue

        fv = get_decoder_vec(layer, feat)

        base_c = abl_c = 0
        per_samples = []
        for samp in cat_samples:
            img = samp["image"]
            if img is None: continue
            try:
                img = img.convert("RGB")
                prompt = build_prompt(samp["statement"])
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, model_raw, device=device
                )
                # baseline
                with _t.inference_mode():
                    out = model_raw(input_ids=input_ids, attention_mask=attn_mask,
                                     pixel_values=pixel_values, use_cache=False)
                base_pred = predict_yesno(out.logits[0, -1, :], yes_ids, no_ids)
                # ablated
                _, img_end = get_image_token_positions(input_ids)
                logits_saved = do_ablation_trace(input_ids, attn_mask, pixel_values, fv, img_end)
                abl_pred = predict_yesno(logits_saved[0, -1, :], yes_ids, no_ids)

                bs = int(base_pred == samp["label"])
                as_ = int(abl_pred == samp["label"])
                base_c += bs; abl_c += as_
                per_samples.append({"id": samp["id"], "base_pred": base_pred,
                                     "abl_pred": abl_pred, "label": samp["label"],
                                     "base_score": bs, "abl_score": as_})
            except Exception as e:
                per_samples.append({"id": samp["id"], "error": str(e)[:200]})

        n = len(cat_samples)
        result = {
            "layer": layer, "feature": feat, "category": cat, "n": n,
            "baseline_acc": 100 * base_c / max(n, 1),
            "ablated_acc": 100 * abl_c / max(n, 1),
            "delta_acc": 100 * (abl_c - base_c) / max(n, 1),
            "samples": per_samples,
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[GPU{gpu_id}] {idx+1}/{len(assignments)} L{layer}/F{feat}/{cat} "
              f"base={result['baseline_acc']:.1f} abl={result['ablated_acc']:.1f} "
              f"∆={result['delta_acc']:+.2f} (n={n})", flush=True)


def main():
    global CHECKPOINT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-csv", type=str,
                        default=str(ANALYSIS_DIR / "ablation_per_relation_full" / "ablation_summary.csv"))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--out-dir", type=str, default=str(ANALYSIS_DIR / "spatialrgpt_ood"))
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    args = parser.parse_args()

    CHECKPOINT_DIR = Path(args.checkpoint_dir)
    os.environ["VLMSCOPE_CKPT_DIR"] = args.checkpoint_dir

    print(f"Loading top-{args.top_n} features from {args.ablation_csv}")
    top_features = load_top_features(Path(args.ablation_csv), args.top_n)
    assignments = []
    for r in top_features:
        for cat in categories_for_feature(r):
            assignments.append((r, cat))
        print(f"  L{r['layer']}/F{r['feature']:<6} sel={r['sel']:+6.2f}  "
              f"rels=[{r['relations'][:50]}]  -> cats={categories_for_feature(r)}")
    print(f"\nTotal (feature, category) assignments: {len(assignments)}")
    if not assignments:
        print("No matching predicate categories for any top feature — exiting.")
        return

    print(f"\nLoading SpatialRGPT-Bench val (predicates: {OVERLAP_CATEGORIES})...")
    samples = load_samples()
    from collections import Counter
    cat_counts = Counter(s["category"] for s in samples)
    print(f"  Loaded {len(samples)} samples")
    for c, n in cat_counts.most_common():
        print(f"    {c}: {n}")

    n_gpus = len(args.gpus)
    chunks = [[] for _ in range(n_gpus)]
    for i, asn in enumerate(assignments):
        chunks[i % n_gpus].append(asn)

    mp.set_start_method("spawn", force=True)
    procs = []
    for i, gpu_id in enumerate(args.gpus):
        p = mp.Process(target=_worker, args=(gpu_id, chunks[i], samples, args.out_dir))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    rows = []
    out_dir = Path(args.out_dir)
    for jp in sorted(out_dir.glob("L*_F*.json")):
        with open(jp) as f:
            d = json.load(f)
        rows.append({k: d.get(k) for k in ["layer", "feature", "category", "n",
                                            "baseline_acc", "ablated_acc", "delta_acc"]})
    csv_path = out_dir / "spatialrgpt_ablation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
    print(f"\nWrote summary: {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
