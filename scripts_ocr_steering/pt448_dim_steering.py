#!/usr/bin/env python3
"""
DIM (Difference-in-Means) steering for pt-448 using mix-448 hidden states.

For each top-10 feature (layer_L, feature_F, relations_R):
  1. Filter VSR → subset S_R (relation-filtered examples)
  2. Run mix-448 forward, collect h[layer_L] for each example
  3. Split by ground-truth label:
       S_pos = label==1 (statement is true)
       S_neg = label==0 (statement is false)
  4. DIM_vec = mean(h[layer_L] | S_pos) - mean(h[layer_L] | S_neg)
  5. Normalize DIM_vec → unit vector
  6. Inject into pt-448 at layer_L (residual stream, text tokens only):
       layer_out += alpha * ones_proxy * DIM_vec
  7. Evaluate VSR accuracy

Contrast with pt448_feature_injection.py which uses W_dec[F] as direction.
Here the direction is empirical: whatever full-hidden-state direction separates
true-spatial from false-spatial in mix-448 at layer L.

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_dim_steering/

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 pt448_dim_steering.py
"""

import os, sys, json, gc, hashlib, warnings
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MIX_MODEL    = "google/paligemma2-3b-mix-448"
PT_MODEL     = "google/paligemma2-3b-pt-448"
N_LAYERS     = 26
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_dim_steering")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

INJECTION_ALPHAS = [-1.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

TOP10 = [
    (9,  387,   ["at the right side of"]),
    (14, 10561, ["close to"]),
    (11, 12278, ["touching"]),
    (9,  7540,  ["consists of"]),
    (4,  14233, ["ahead of"]),
    (6,  7539,  ["left of", "right of"]),
    (11, 9639,  ["in", "inside", "on"]),
    (13, 15219, ["behind"]),
    (15, 220,   ["across from", "at the left side of"]),
    (12, 2257,  ["facing"]),
]


def _build_vsr_prompt(statement):
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\nAnswer:"
    )


def _get_yes_no_ids(tokenizer):
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


def _predict(logits, yes_ids, no_ids):
    probs = torch.softmax(logits, dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
    n = probs[list(no_ids)].sum().item() if no_ids else 0.0
    d = y + n
    return 1 if (y / d if d > 0 else 0) > 0.5 else 0


def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"):
        return None
    h = hashlib.md5(url.encode()).hexdigest()
    cp = IMAGE_CACHE / f"{h}.jpg"
    try:
        if cp.exists():
            return Image.open(cp).convert("RGB")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(cp, "JPEG")
        return img
    except Exception:
        return None


def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    # Load both models
    print(f"[INFO] Loading {MIX_MODEL} (for DIM extraction)...", flush=True)
    processor = AutoProcessor.from_pretrained(MIX_MODEL)
    mix_model = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    mix_nns = NNsight(mix_model)

    print(f"[INFO] Loading {PT_MODEL} (for steering)...", flush=True)
    pt_processor = AutoProcessor.from_pretrained(PT_MODEL)
    pt_model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    pt_nns = NNsight(pt_model)

    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(mix_model.parameters()).dtype

    print("[INFO] Loading VSR (all splits)...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    print(f"[INFO] VSR total: {len(vsr_all)}", flush=True)

    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation", "")].append(vi)

    def get_filtered_indices(relations):
        idxs = []
        for r in relations:
            idxs.extend(relation_indices.get(r, []))
        return idxs

    def extract_dim_vec(indices, layer_idx):
        """Collect mix-448 hidden states at layer_idx, compute DIM by ground-truth label."""
        pos_vecs, neg_vecs = [], []
        for vi in indices:
            ex = vsr_all[vi]
            img = _load_image(ex)
            if img is None:
                continue
            label = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, mix_nns._module, device=device
                )
                _, img_end = get_image_token_positions(input_ids)
                with mix_nns.trace(input_ids=input_ids, attention_mask=attn_mask,
                                   pixel_values=pixel_values):
                    # Save last text token hidden state at layer_idx output
                    h = mix_nns.model.language_model.layers[layer_idx].output[0][0, -1].save()
                h_val = h.detach().float().cpu()
                if label == 1:
                    pos_vecs.append(h_val)
                else:
                    neg_vecs.append(h_val)
            except Exception:
                continue

        if not pos_vecs or not neg_vecs:
            return None, 0, 0

        pos_mean = torch.stack(pos_vecs).mean(0)
        neg_mean = torch.stack(neg_vecs).mean(0)
        dim_vec  = pos_mean - neg_mean
        dim_vec  = dim_vec / dim_vec.norm().clamp(min=1e-8)
        return dim_vec.to(model_dtype).to(device), len(pos_vecs), len(neg_vecs)

    def run_vsr(indices, dim_vec=None, layer_idx=None, alpha=0.0):
        correct = total = 0
        for vi in indices:
            ex = vsr_all[vi]
            img = _load_image(ex)
            if img is None:
                continue
            label = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, pt_processor, pt_model if dim_vec is None else pt_nns._module,
                    device=device
                )
                if dim_vec is None:
                    with torch.inference_mode():
                        out = pt_model(input_ids=input_ids, attention_mask=attn_mask,
                                       pixel_values=pixel_values, use_cache=False)
                    pred = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                else:
                    _, img_end = get_image_token_positions(input_ids)
                    fv_col = dim_vec.unsqueeze(1)
                    with pt_nns.trace(input_ids=input_ids, attention_mask=attn_mask,
                                      pixel_values=pixel_values):
                        layer_out = pt_nns.model.language_model.layers[layer_idx].output[0][0, img_end:]
                        ones = (layer_out @ fv_col) * 0.0 + 1.0
                        layer_out += alpha * ones * dim_vec
                        logits_s = pt_nns.output.logits.save()
                    pred = _predict(logits_s[0, -1, :], yes_ids, no_ids)
            except Exception:
                pred = 0
            total += 1
            correct += (pred == label)
        return correct / max(total, 1) * 100, total

    baseline_cache = {}
    all_results = []

    for layer_idx, feature_idx, relations in TOP10:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"dim_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key} already done", flush=True)
            with open(result_path) as f:
                all_results.append(json.load(f))
            continue

        indices = get_filtered_indices(relations)
        if not indices:
            print(f"[WARN] {key}: no VSR samples for {relations}", flush=True)
            continue

        rel_key = ";".join(sorted(relations))
        if rel_key not in baseline_cache:
            print(f"[BASE] pt-448 baseline for [{rel_key}] (N={len(indices)})...", flush=True)
            acc, n = run_vsr(indices)
            baseline_cache[rel_key] = (acc, n)
            print(f"[BASE] Baseline: {acc:.2f}% (N={n})", flush=True)
        base_acc, base_n = baseline_cache[rel_key]

        # Extract DIM vector from mix-448
        print(f"[DIM] Extracting DIM vector for {key} at layer {layer_idx} (N={len(indices)})...", flush=True)
        dim_vec, n_pos, n_neg = extract_dim_vec(indices, layer_idx)
        if dim_vec is None:
            print(f"[WARN] {key}: could not compute DIM vec (pos={n_pos} neg={n_neg})", flush=True)
            continue
        print(f"[DIM] {key}: n_pos={n_pos} n_neg={n_neg} vec_norm={dim_vec.norm().item():.4f}", flush=True)

        result = {
            "layer": layer_idx, "feature": feature_idx,
            "relations": relations, "n_samples": len(indices),
            "n_pos": n_pos, "n_neg": n_neg,
            "baseline_vsr_acc": base_acc,
            "method": "DIM_layer_hidden_state",
            "alphas": {}
        }

        for alpha in INJECTION_ALPHAS:
            print(f"[STEER] {key} α={alpha:+g} (N={len(indices)})...", flush=True)
            acc, n = run_vsr(indices, dim_vec, layer_idx, alpha)
            delta = acc - base_acc
            result["alphas"][str(alpha)] = {"acc": acc, "delta": delta, "n": n}
            print(f"  α={alpha:+g}: {acc:.2f}% (Δ={delta:+.2f}%)", flush=True)

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)
        torch.cuda.empty_cache(); gc.collect()

    # Summary
    print(f"\n{'='*110}")
    print("pt-448 DIM Steering — Layer hidden-state contrast (mix-448 pos/neg by ground-truth label)")
    print(f"{'='*110}")
    alphas = INJECTION_ALPHAS
    header = f"{'L/F':<12} {'Relations':<32} {'N':>5} {'Base':>7}"
    for a in alphas:
        header += f"  {a:>+6}"
    print(header)
    print("-" * 110)
    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        rels = "; ".join(r["relations"])[:31]
        row = f"{key:<12} {rels:<32} {r['n_samples']:>5} {r['baseline_vsr_acc']:>6.1f}%"
        for a in alphas:
            d = r["alphas"].get(str(a), {}).get("delta")
            row += f"  {d:>+6.1f}" if d is not None else f"  {'--':>6}"
        print(row)

    import csv
    csv_path = OUT_DIR / "dim_steering_summary.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["layer", "feature", "relations", "n_samples", "n_pos", "n_neg", "baseline_vsr_acc"]
        for a in alphas:
            fieldnames.append(f"delta_alpha_{a:+g}")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            row = {
                "layer": r["layer"], "feature": r["feature"],
                "relations": "; ".join(r["relations"]),
                "n_samples": r["n_samples"],
                "n_pos": r.get("n_pos", ""), "n_neg": r.get("n_neg", ""),
                "baseline_vsr_acc": f"{r['baseline_vsr_acc']:.2f}",
            }
            for a in alphas:
                d = r["alphas"].get(str(a), {}).get("delta", "")
                row[f"delta_alpha_{a:+g}"] = f"{d:.2f}" if d != "" else ""
            w.writerow(row)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
