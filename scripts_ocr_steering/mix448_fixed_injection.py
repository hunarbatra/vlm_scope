#!/usr/bin/env python3
"""
Fixed-vector injection into mix-448 to further boost spatial VSR accuracy.

Unlike pt-448 injection (bridging a gap), this asks: can we make the already
instruction-tuned model even MORE accurate by amplifying the spatial feature
directions it already uses?

Same W_dec[F] directions, same proxy trick, but target = mix-448.
Since mix-448 already fires these features strongly, we expect the optimal
alpha to be very small (avoid overpowering).

Also tests NEGATIVE alpha = partial feature suppression in mix-448
(confirming ablation direction from Exp 1).

Baseline is mix-448 VSR accuracy (60-86% range).

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/mix448_fixed_injection/

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 mix448_fixed_injection.py
"""

import os, sys, json, gc, hashlib, warnings, math
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL_NAME     = "google/paligemma2-3b-mix-448"
N_LAYERS       = 26
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/mix448_fixed_injection")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
HF_CACHE       = "/data1/hf_cache/hub"
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# Small range for mix-448 (features already present); negative = partial ablation sanity
INJECTION_ALPHAS = [-2.0, -1.0, -0.5, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

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


def _predict_and_margin(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item() if no_ids else 1e-9
    d = y + n
    p_yes = max(y / d if d > 0 else 0.5, 1e-7)
    p_no  = max(1.0 - p_yes, 1e-7)
    return (1 if p_yes > 0.5 else 0), math.log(p_yes / p_no)


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

    relation_indices = defaultdict(list)
    for vi in range(len(vsr_all)):
        relation_indices[vsr_all[vi].get("relation", "")].append(vi)

    def get_filtered_indices(relations):
        idxs = []
        for r in relations:
            idxs.extend(relation_indices.get(r, []))
        return idxs

    def run_vsr_baseline(indices):
        correct = total = 0; margins = []
        for vi in indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            label = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, model_raw, device=device)
                with torch.inference_mode():
                    out = model_raw(input_ids=input_ids, attention_mask=attn_mask,
                                   pixel_values=pixel_values, use_cache=False)
                pred, margin = _predict_and_margin(out.logits[0, -1, :], yes_ids, no_ids)
                margins.append(margin if label == 1 else -margin)
            except Exception:
                pred = 0; margins.append(0.0)
            total += 1; correct += (pred == label)
        acc = correct / max(total, 1) * 100
        return acc, sum(margins) / max(len(margins), 1), total

    def run_vsr_injected(indices, fv, layer_idx, alpha):
        correct = total = 0; margins = []
        fv_col = fv.unsqueeze(1)
        for vi in indices:
            ex = vsr_all[vi]; img = _load_image(ex)
            if img is None: continue
            label = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))
            try:
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, nns_model._module, device=device)
                _, img_end = get_image_token_positions(input_ids)
                with nns_model.trace(input_ids=input_ids, attention_mask=attn_mask,
                                     pixel_values=pixel_values):
                    layer_out = nns_model.model.language_model.layers[layer_idx].output[0][0, img_end:]
                    ones = (layer_out @ fv_col) * 0.0 + 1.0
                    layer_out += alpha * ones * fv
                    logits_s = nns_model.output.logits.save()
                pred, margin = _predict_and_margin(logits_s[0, -1, :], yes_ids, no_ids)
                margins.append(margin if label == 1 else -margin)
            except Exception:
                pred = 0; margins.append(0.0)
            total += 1; correct += (pred == label)
        acc = correct / max(total, 1) * 100
        return acc, sum(margins) / max(len(margins), 1), total

    baseline_cache = {}
    all_results = []

    for layer_idx, feature_idx, relations in TOP10:
        key = f"L{layer_idx}_F{feature_idx}"
        result_path = OUT_DIR / f"mix_inject_{key}.json"
        if result_path.exists():
            print(f"[SKIP] {key}", flush=True)
            with open(result_path) as f: all_results.append(json.load(f))
            continue

        indices = get_filtered_indices(relations)
        if not indices: continue

        rel_key = ";".join(sorted(relations))
        if rel_key not in baseline_cache:
            print(f"[BASE] mix-448 baseline [{rel_key}] (N={len(indices)})...", flush=True)
            acc, margin, n = run_vsr_baseline(indices)
            baseline_cache[rel_key] = (acc, margin, n)
            print(f"[BASE] {acc:.2f}% margin={margin:.3f} (N={n})", flush=True)
        base_acc, base_margin, base_n = baseline_cache[rel_key]

        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        fv = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        fv = fv / fv.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()

        result = {
            "layer": layer_idx, "feature": feature_idx, "relations": relations,
            "n_samples": len(indices), "baseline_vsr_acc": base_acc,
            "baseline_margin": base_margin, "model": "mix-448", "alphas": {}
        }

        for alpha in INJECTION_ALPHAS:
            print(f"[INJECT] {key} α={alpha:+g} (N={len(indices)})...", flush=True)
            acc, margin, n = run_vsr_injected(indices, fv, layer_idx, alpha)
            da = acc - base_acc; dm = margin - base_margin
            result["alphas"][str(alpha)] = {"acc": acc, "delta_acc": da,
                                            "margin": margin, "delta_margin": dm, "n": n}
            print(f"  α={alpha:+g}: {acc:.2f}% (Δ={da:+.2f}%) margin={margin:.3f} (Δ={dm:+.3f})", flush=True)

        with open(result_path, "w") as f: json.dump(result, f, indent=2)
        all_results.append(result)
        torch.cuda.empty_cache(); gc.collect()

    print(f"\n{'='*110}")
    print("mix-448 Fixed Injection (can fine-tuned model be boosted further?)")
    print(f"{'='*110}")
    header = f"{'L/F':<12} {'Relations':<32} {'N':>5} {'Base':>7}"
    for a in INJECTION_ALPHAS: header += f"  {a:>+6}"
    print(header); print("-" * 110)
    for r in all_results:
        key = f"L{r['layer']}/F{r['feature']}"
        rels = "; ".join(r["relations"])[:31]
        row = f"{key:<12} {rels:<32} {r['n_samples']:>5} {r['baseline_vsr_acc']:>6.1f}%"
        for a in INJECTION_ALPHAS:
            d = r["alphas"].get(str(a), {}).get("delta_acc")
            row += f"  {d:>+6.2f}" if d is not None else f"  {'--':>6}"
        print(row)

    import csv
    csv_path = OUT_DIR / "mix_injection_summary.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["layer", "feature", "relations", "n_samples", "baseline_vsr_acc"]
        for a in INJECTION_ALPHAS: fieldnames += [f"delta_acc_{a}", f"delta_margin_{a}"]
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in all_results:
            row = {"layer": r["layer"], "feature": r["feature"],
                   "relations": "; ".join(r["relations"]), "n_samples": r["n_samples"],
                   "baseline_vsr_acc": f"{r['baseline_vsr_acc']:.2f}"}
            for a in INJECTION_ALPHAS:
                v = r["alphas"].get(str(a), {})
                row[f"delta_acc_{a}"] = f"{v.get('delta_acc', ''):.2f}" if v else ""
                row[f"delta_margin_{a}"] = f"{v.get('delta_margin', ''):.3f}" if v else ""
            w.writerow(row)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
