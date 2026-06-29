#!/usr/bin/env python3
"""
Projection-based steering on mix-448 for the top-10 spatial features.

For each feature, amplifies its existing projection in the residual stream:
    act += alpha * (act @ fv.T) * fv   (at all 3 taps, all 26 layers)

alpha=-1  ≈ ablation (sanity check — should reproduce ablation drops)
alpha>0   amplifies the existing feature → tests if more spatial signal helps

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 mix448_projection_steering.py
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

MODEL_NAME   = "google/paligemma2-3b-mix-448"
N_LAYERS     = 26
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix448_projection_steering")
LOG_PATH       = Path("/data1/vlm_scope_sae_mix448_textonly/logs/mix448_projection_steering.log")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# alpha=-1 ≈ ablation (sanity), positives = amplify
STEERING_ALPHAS = [-1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0]

# Top-10 canonical spatial features (layer, feature, relations)
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
    import torch.multiprocessing
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {MODEL_NAME}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_model = NNsight(model_raw)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    print("[INFO] Loading VSR (all splits)...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    print(f"[INFO] VSR total: {len(vsr_all)}", flush=True)

    # Pre-index by relation
    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation", "")].append(vi)

    def get_filtered_indices(relations):
        idxs = []
        for r in relations:
            idxs.extend(relation_indices.get(r, []))
        return idxs

    def run_vsr(indices, feature_vec=None, alpha=0.0):
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
                    img, prompt, processor,
                    model_raw if feature_vec is None else nns_model._module,
                    device=device
                )
                if feature_vec is None:
                    with torch.inference_mode():
                        out = model_raw(input_ids=input_ids, attention_mask=attn_mask,
                                        pixel_values=pixel_values, use_cache=False)
                    pred = _predict(out.logits[0, -1, :], yes_ids, no_ids)
                else:
                    _, img_end = get_image_token_positions(input_ids)
                    fv = feature_vec.unsqueeze(0)  # (1, d)
                    with nns_model.trace(input_ids=input_ids, attention_mask=attn_mask,
                                         pixel_values=pixel_values):
                        for l in range(N_LAYERS):
                            attn_out = nns_model.model.language_model.layers[l].self_attn.output[0][0, img_end:]
                            attn_out += alpha * (attn_out @ fv.T) * fv
                            mlp_out = nns_model.model.language_model.layers[l].mlp.output[0, img_end:]
                            mlp_out += alpha * (mlp_out @ fv.T) * fv
                            layer_out = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                            layer_out += alpha * (layer_out @ fv.T) * fv
                        logits_s = nns_model.output.logits.save()
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
        result_path = OUT_DIR / f"steering_{key}.json"
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
            print(f"[BASE] Computing baseline for [{rel_key}] (N={len(indices)})...", flush=True)
            acc, n = run_vsr(indices)
            baseline_cache[rel_key] = (acc, n)
            print(f"[BASE] Baseline: {acc:.2f}% (N={n})", flush=True)
        base_acc, base_n = baseline_cache[rel_key]

        # Load SAE feature direction
        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                       device=device, cache_dir=HF_CACHE)
        sae.eval()
        fv = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        fv = fv / fv.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()

        result = {
            "layer": layer_idx, "feature": feature_idx,
            "relations": relations, "n_samples": len(indices),
            "baseline_vsr_acc": base_acc,
            "alphas": {}
        }

        for alpha in STEERING_ALPHAS:
            print(f"[STEER] {key} α={alpha:+g} (N={len(indices)})...", flush=True)
            acc, n = run_vsr(indices, fv, alpha)
            delta = acc - base_acc
            result["alphas"][str(alpha)] = {"acc": acc, "delta": delta, "n": n}
            print(f"  α={alpha:+g}: {acc:.2f}% (Δ={delta:+.2f}%)", flush=True)

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        all_results.append(result)
        torch.cuda.empty_cache(); gc.collect()

    # Print summary table
    print(f"\n{'='*100}")
    print(f"mix-448 Projection Steering — Top-10 Spatial Features")
    print(f"{'='*100}")
    alphas = STEERING_ALPHAS
    header = f"{'L/F':<12} {'Relations':<32} {'N':>5} {'Base':>7}"
    for a in alphas:
        header += f"  {a:>+6}"
    print(header)
    print("-" * 100)
    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        rels = "; ".join(r["relations"])[:31]
        row = f"{key:<12} {rels:<32} {r['n_samples']:>5} {r['baseline_vsr_acc']:>6.1f}%"
        for a in alphas:
            d = r["alphas"].get(str(a), {}).get("delta")
            row += f"  {d:>+6.1f}" if d is not None else f"  {'--':>6}"
        print(row)

    # Save summary CSV
    import csv
    csv_path = OUT_DIR / "steering_summary.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["layer", "feature", "relations", "n_samples", "baseline_vsr_acc"]
        for a in alphas:
            fieldnames.append(f"delta_alpha_{a:+g}")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            row = {
                "layer": r["layer"], "feature": r["feature"],
                "relations": "; ".join(r["relations"]),
                "n_samples": r["n_samples"],
                "baseline_vsr_acc": f"{r['baseline_vsr_acc']:.2f}",
            }
            for a in alphas:
                d = r["alphas"].get(str(a), {}).get("delta", "")
                row[f"delta_alpha_{a:+g}"] = f"{d:.2f}" if d != "" else ""
            w.writerow(row)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
