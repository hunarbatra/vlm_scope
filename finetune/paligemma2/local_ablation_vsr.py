#!/usr/bin/env python3
"""
Local ablation: evaluates feature ablation on VSR, VQA, and Control.
Adapted from modal_ablation.py for local A5000 GPUs.

Usage:
    HF_TOKEN=hf_... CUDA_VISIBLE_DEVICES=0 python local_ablation_vsr.py \
        --features-csv analysis_results/firing_pertoken/spatial_adapted_pertoken.csv \
        --max-vqa 500 --max-vsr 2000 \
        --results-dir analysis_results/ablation_vsr
"""
import argparse, os, sys, json, hashlib, warnings
from pathlib import Path
from collections import defaultdict

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from PIL import Image
from io import BytesIO
import requests
from nnsight import NNsight

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    initialize_vlm_model, process_vlm_inputs,
    get_image_token_positions, initialize_jumprelu_sae,
)

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

MODEL_NAME = os.environ.get("VLM_MODEL_NAME", "google/paligemma2-3b-ft-docci-448")
CHECKPOINT_DIR = Path(os.environ.get("SAE_CHECKPOINT_DIR", "/data1/vlm_scope_sae_docci/checkpoints"))
CONTROL_RELATIONS = {"has", "wears", "holds", "made of", "part of", "contains"}


def get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    return yes_ids - overlap, no_ids - overlap


def predict_yesno(logits_saved, yes_ids, no_ids):
    logits = logits_saved[:, -1, :]
    probs = torch.softmax(logits, dim=-1)[0]
    yes_mass = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
    no_mass = probs[list(no_ids)].sum().item() if no_ids else 0.0
    denom = yes_mass + no_mass
    p_yes = yes_mass / denom if denom > 0 else 0.0
    return 1 if p_yes > 0.5 else 0


def load_vsr_samples(max_samples=2000):
    """Load VSR with image downloading (or from cache)."""
    import pickle
    cache_path = Path("analysis_results/data_cache/vsr_samples.pkl")
    if cache_path.exists():
        print(f"Loading VSR from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            all_samples = pickle.load(f)
        samples = all_samples[:max_samples]
        print(f"  {len(samples)} VSR samples (from cache)")
        return samples

    from concurrent.futures import ThreadPoolExecutor, as_completed
    print(f"Loading VSR (max {max_samples})...")
    raw = []
    for split in ["train", "validation", "test"]:
        ds = load_dataset("cambridgeltl/vsr_random", split=split)
        for item in ds:
            raw.append(item)
            if len(raw) >= max_samples:
                break
        if len(raw) >= max_samples:
            break

    def dl(item):
        try:
            url = item.get("image_link", "")
            if not url.startswith("http"): return None
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            return {
                "image": img, "caption": item.get("caption", ""),
                "label": int(item.get("label", 0)),
                "relation": item.get("relation", ""),
            }
        except: return None

    samples = []
    with ThreadPoolExecutor(max_workers=64) as ex:
        futs = {ex.submit(dl, it): i for i, it in enumerate(raw)}
        for f in tqdm(as_completed(futs), total=len(futs), desc="VSR images"):
            r = f.result()
            if r: samples.append(r)
    print(f"  {len(samples)} VSR samples ready")
    return samples


def load_vqa_yesno(max_samples=500):
    import pickle
    cache_path = Path("analysis_results/data_cache/vqa_samples.pkl")
    if cache_path.exists():
        print(f"Loading VQA from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            all_samples = pickle.load(f)
        # Cache uses 'answer' key; normalize to 'multiple_choice_answer'
        samples = []
        for s in all_samples:
            ans = str(s.get("multiple_choice_answer", s.get("answer", ""))).strip().lower()
            if ans in ("yes", "no"):
                if "multiple_choice_answer" not in s:
                    s["multiple_choice_answer"] = s.get("answer", "")
                samples.append(s)
                if len(samples) >= max_samples:
                    break
        print(f"  {len(samples)} VQA yes/no samples (from cache)")
        return samples

    print(f"Loading VQA yes/no (max {max_samples})...")
    ds = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True)
    samples = []
    for item in tqdm(ds, desc="VQA scan", total=max_samples * 3):
        mc = str(item.get("multiple_choice_answer", "")).strip().lower()
        if mc in ("yes", "no"):
            samples.append(item)
            if len(samples) >= max_samples:
                break
    print(f"  {len(samples)} VQA yes/no samples")
    return samples


def do_ablation_trace(nns_model, input_ids, attention_mask, pixel_values,
                      feature_vec, img_end, n_layers):
    """Ablate feature direction from all layers (text tokens only).

    Projects out the feature direction at 3 points per layer:
    1. Self-attention output
    2. MLP output
    3. Layer output (residual stream)

    Matches modal_ablation.py / original ablate_sae_feature_vsr.py.
    """
    fv = feature_vec.unsqueeze(0)  # (1, d_in)
    with nns_model.trace(
        input_ids=input_ids, attention_mask=attention_mask,
        pixel_values=pixel_values, use_cache=False,
    ) as tr:
        for l in range(n_layers):
            # 1. Self-attention output
            attn_out = nns_model.model.language_model.layers[l].self_attn.output[0][0, img_end:]
            attn_proj = (attn_out @ fv.T) * fv
            attn_out -= attn_proj

            # 2. MLP output
            mlp_out = nns_model.model.language_model.layers[l].mlp.output[0, img_end:]
            mlp_proj = (mlp_out @ fv.T) * fv
            mlp_out -= mlp_proj

            # 3. Layer output (residual stream)
            layer_out = nns_model.model.language_model.layers[l].output[0][img_end:]
            layer_proj = (layer_out @ fv.T) * fv
            layer_out -= layer_proj

        logits_saved = nns_model.output.logits.save()
    return logits_saved


def eval_baseline_vsr(vsr_samples, processor, model, yes_ids, no_ids, device):
    correct = ctrl_correct = 0
    total = ctrl_total = 0
    for s in tqdm(vsr_samples, desc="VSR baseline"):
        prompt = f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s['caption'].strip()}\nAnswer:"
        inp = process_vlm_inputs(s["image"], prompt, processor, model, device=device)
        with torch.inference_mode():
            out = model(input_ids=inp[0], attention_mask=inp[1], pixel_values=inp[2], use_cache=False)
        probs = torch.softmax(out.logits[:, -1, :], dim=-1)[0]
        yes_m = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
        no_m = probs[list(no_ids)].sum().item() if no_ids else 0.0
        d = yes_m + no_m
        pred = 1 if (yes_m / d if d > 0 else 0) > 0.5 else 0
        total += 1
        if pred == s["label"]: correct += 1
        if s["relation"] in CONTROL_RELATIONS:
            ctrl_total += 1
            if pred == s["label"]: ctrl_correct += 1
    return {
        "vsr_acc": correct / total * 100 if total else 0,
        "vsr_total": total,
        "ctrl_acc": ctrl_correct / ctrl_total * 100 if ctrl_total else 0,
        "ctrl_total": ctrl_total,
    }


def eval_baseline_vqa(vqa_samples, processor, model, yes_ids, no_ids, device):
    correct = total = 0
    for s in tqdm(vqa_samples, desc="VQA baseline"):
        img = s["image"].convert("RGB")
        q = s.get("question", "").strip()
        label = 1 if s.get("multiple_choice_answer", "").strip().lower() == "yes" else 0
        prompt = f"Answer the following question with only 'Yes' or 'No':\nQuestion: {q}\nAnswer:"
        inp = process_vlm_inputs(img, prompt, processor, model, device=device)
        with torch.inference_mode():
            out = model(input_ids=inp[0], attention_mask=inp[1], pixel_values=inp[2], use_cache=False)
        probs = torch.softmax(out.logits[:, -1, :], dim=-1)[0]
        yes_m = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
        no_m = probs[list(no_ids)].sum().item() if no_ids else 0.0
        d = yes_m + no_m
        pred = 1 if (yes_m / d if d > 0 else 0) > 0.5 else 0
        total += 1
        if pred == label: correct += 1
    return {"vqa_acc": correct / total * 100 if total else 0, "vqa_total": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--max-vqa", type=int, default=500)
    parser.add_argument("--max-vsr", type=int, default=2000)
    parser.add_argument("--results-dir", default="analysis_results/ablation_vsr")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    df = pd.read_csv(args.features_csv)
    if "odds_ratio" in df.columns:
        df = df.sort_values("odds_ratio", ascending=False)
    print(f"Features to ablate: {len(df)}")

    # Load data
    vsr_samples = load_vsr_samples(args.max_vsr)
    vqa_samples = load_vqa_yesno(args.max_vqa)

    # Load model (use local cache to avoid gated repo auth issues)
    print("Loading PaliGemma2...")
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=False)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, local_files_only=False
    ).to(device).eval()
    nns_model = NNsight(model_raw)
    yes_ids, no_ids = get_yes_no_ids(processor.tokenizer)
    n_layers = model_raw.config.text_config.num_hidden_layers

    # Baselines (cached)
    baseline_path = results_dir / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"Cached baseline: VSR={baseline['vsr_acc']:.2f}% VQA={baseline['vqa_acc']:.2f}% Ctrl={baseline['ctrl_acc']:.2f}%")
    else:
        print("Computing baselines...")
        b_vsr = eval_baseline_vsr(vsr_samples, processor, model_raw, yes_ids, no_ids, device)
        b_vqa = eval_baseline_vqa(vqa_samples, processor, model_raw, yes_ids, no_ids, device)
        baseline = {**b_vsr, **b_vqa}
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"Baseline: VSR={baseline['vsr_acc']:.2f}% VQA={baseline['vqa_acc']:.2f}% Ctrl={baseline['ctrl_acc']:.2f}%")

    # Ablate each feature
    results = []
    for _, row in df.iterrows():
        layer_idx = int(row["layer"])
        feature_idx = int(row["feature"])

        # Check if already done
        feat_path = results_dir / f"ablation_L{layer_idx}_F{feature_idx}.json"
        if feat_path.exists():
            with open(feat_path) as f:
                result = json.load(f)
            results.append(result)
            print(f"  L{layer_idx} F{feature_idx}: SKIP (cached)")
            continue

        # Load SAE, get feature vec
        ckpt_path = CHECKPOINT_DIR / f"pretrained_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path), device=device)
        sae.eval()
        fv = sae.W_dec[feature_idx].detach().to(torch.bfloat16).to(device)
        fv = fv / fv.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()

        # --- Ablated VSR ---
        vsr_correct = ctrl_correct = 0
        vsr_total = ctrl_total = 0
        for si, s in enumerate(tqdm(vsr_samples, desc=f"L{layer_idx}F{feature_idx} VSR")):
            prompt = f"Is the following statement correct? Answer only with 'Yes' or 'No'.\nStatement: {s['caption'].strip()}\nAnswer:"
            inp_ids, attn, pv = process_vlm_inputs(s["image"], prompt, processor, model_raw, device=device)
            _, img_end = get_image_token_positions(inp_ids)
            try:
                logits = do_ablation_trace(nns_model, inp_ids, attn, pv, fv, img_end, n_layers)
                pred = predict_yesno(logits, yes_ids, no_ids)
            except Exception as e:
                if si < 3: print(f"  [ERR] VSR {si}: {e}")
                pred = 0
            vsr_total += 1
            if pred == s["label"]: vsr_correct += 1
            if s["relation"] in CONTROL_RELATIONS:
                ctrl_total += 1
                if pred == s["label"]: ctrl_correct += 1

        # --- Ablated VQA ---
        vqa_correct = vqa_total = 0
        for si, s in enumerate(tqdm(vqa_samples, desc=f"L{layer_idx}F{feature_idx} VQA")):
            img = s["image"].convert("RGB")
            q = s.get("question", "").strip()
            label = 1 if s.get("multiple_choice_answer", "").strip().lower() == "yes" else 0
            prompt = f"Answer the following question with only 'Yes' or 'No':\nQuestion: {q}\nAnswer:"
            inp_ids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
            _, img_end = get_image_token_positions(inp_ids)
            try:
                logits = do_ablation_trace(nns_model, inp_ids, attn, pv, fv, img_end, n_layers)
                pred = predict_yesno(logits, yes_ids, no_ids)
            except Exception as e:
                if si < 3: print(f"  [ERR] VQA {si}: {e}")
                pred = 0
            vqa_total += 1
            if pred == label: vqa_correct += 1

        abl_vsr = vsr_correct / vsr_total * 100 if vsr_total else 0
        abl_ctrl = ctrl_correct / ctrl_total * 100 if ctrl_total else 0
        abl_vqa = vqa_correct / vqa_total * 100 if vqa_total else 0

        result = {
            "layer": layer_idx, "feature": feature_idx,
            "odds_ratio": float(row.get("odds_ratio", 0)),
            "freq_diff": float(row.get("freq_diff", 0)),
            "abl_vsr": abl_vsr, "abl_ctrl": abl_ctrl, "abl_vqa": abl_vqa,
            "delta_vsr": abl_vsr - baseline["vsr_acc"],
            "delta_ctrl": abl_ctrl - baseline["ctrl_acc"],
            "delta_vqa": abl_vqa - baseline["vqa_acc"],
            "vsr_total": vsr_total, "ctrl_total": ctrl_total, "vqa_total": vqa_total,
        }
        results.append(result)

        with open(feat_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n  L{layer_idx} F{feature_idx}: ΔVSR={result['delta_vsr']:+.2f}pp  ΔCtrl={result['delta_ctrl']:+.2f}pp  ΔVQA={result['delta_vqa']:+.2f}pp")

        # Save aggregate
        pd.DataFrame(results).to_csv(results_dir / "ablation_results.csv", index=False)
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print(f"{'Layer':<6} {'Feat':<7} {'OR':>6} {'ΔVSR':>8} {'ΔCtrl':>8} {'ΔVQA':>8}")
    print("-"*70)
    for r in sorted(results, key=lambda x: x["delta_vsr"]):
        print(f"  L{r['layer']:<4} F{r['feature']:<5} {r['odds_ratio']:>6.1f} "
              f"{r['delta_vsr']:>+7.2f}pp {r['delta_ctrl']:>+7.2f}pp {r['delta_vqa']:>+7.2f}pp")
    print("="*70)
    print(f"Baseline: VSR={baseline['vsr_acc']:.2f}%  Ctrl={baseline['ctrl_acc']:.2f}%  VQA={baseline['vqa_acc']:.2f}%")


if __name__ == "__main__":
    main()
