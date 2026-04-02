"""
Modal deployment for PaliGemma2 SAE feature ablation experiments.

Evaluates the causal impact of individual SAE features on:
  - VSR (Visual Spatial Reasoning) accuracy
  - VQA (yes/no subset) accuracy
  - Control VSR (non-spatial relations)

Produces a results table matching the paper's Table 1 format:
  Layer | Feature | ∆VSR Acc | ∆VQA Acc | ∆Ctrl VSR | Relation | VSR OR

Usage:
    cd finetune/paligemma2
    MODAL_PROFILE=hunar-oxford modal run modal_ablation.py
"""

import os
import json
import modal
from pathlib import Path

VOLUME_NAME = "vlm-scope-data-v2"
GPU_TYPE = "A100"
TIMEOUT = 86400

app = modal.App("vlm-scope-ablation")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.44",
        "sae-lens>=4.0",
        "nnsight>=0.3",
        "datasets",
        "h5py",
        "tqdm",
        "huggingface-hub",
        "Pillow",
        "numpy",
        "scipy",
        "pandas",
        "accelerate",
        "requests",
    )
    .env({
        "HF_HOME": "/vol/cache/huggingface",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
        "WANDB_MODE": "disabled",
    })
    .add_local_file(
        local_path=str(Path(__file__).parent / "utils.py"),
        remote_path="/root/paligemma2/utils.py",
    )
)

# --------------- Constants ---------------

N_LAYERS = 26
D_SAE = 16384
RESULTS_BASE = "/vol/results/paligemma2"
MODEL_NAME = "google/paligemma2-3b-pt-224"
SAE_TYPE = "jumprelu"
N_GPUS = 8

# VSR evaluation config
VSR_DATASET = "cambridgeltl/vsr_random"
VSR_SPLITS = ["train", "dev", "test"]
VQA_MAX_SAMPLES = 1000  # yes/no VQA samples for ∆VQA

# Image cache on volume
IMAGE_CACHE_DIR = "/vol/cache/vsr_images"

# Spatial relations to use as control (non-spatial)
CONTROL_RELATIONS = {"has", "wears", "holds", "made of", "part of", "contains"}


# --------------- Helper functions ---------------

def _build_vsr_prompt(statement: str) -> str:
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\n"
        "Answer:"
    )


def _get_yes_no_ids(tokenizer):
    """Get token IDs for yes/no answers."""
    yes_ids = set()
    no_ids = set()
    for text in [" Yes", "Yes", " yes", "YES"]:
        toks = tokenizer.encode(text, add_special_tokens=False)
        if toks:
            yes_ids.add(toks[0])
    for text in [" No", "No", " no", "NO"]:
        toks = tokenizer.encode(text, add_special_tokens=False)
        if toks:
            no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap
    no_ids -= overlap
    return yes_ids, no_ids


# --------------- Baseline evaluation (no ablation) ---------------

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/vol": volume},
    timeout=TIMEOUT,
)
def compute_baseline():
    """Compute baseline VSR + VQA accuracy without any ablation."""
    import sys
    import hashlib
    import torch
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    from datasets import load_dataset, concatenate_datasets
    from PIL import Image
    from io import BytesIO
    import requests

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_vlm_model, process_vlm_inputs, get_image_token_positions

    out_dir = Path(RESULTS_BASE) / "analysis" / "ablation_jumprelu"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "baseline_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"[Baseline] Loaded from cache: VSR={cached['vsr_acc']:.2f}%, VQA={cached['vqa_acc']:.2f}%")
        return cached

    print("[Baseline] Loading PaliGemma2...")
    processor, model = initialize_vlm_model(MODEL_NAME, device="cuda")
    model.eval()
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)

    # Load VSR (all splits)
    print("[Baseline] Loading VSR dataset...")
    vsr_splits = []
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    for split in VSR_SPLITS:
        vsr_splits.append(load_dataset(VSR_DATASET, data_files=data_files, split=split))
    vsr = concatenate_datasets(vsr_splits)
    print(f"[Baseline] VSR: {len(vsr)} samples")

    # Evaluate VSR baseline
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    vsr_correct = 0
    vsr_total = 0
    vsr_per_relation = {}  # relation -> {correct, total}
    vsr_ctrl_correct = 0
    vsr_ctrl_total = 0

    for i in tqdm(range(len(vsr)), desc="VSR baseline"):
        ex = vsr[i]
        url = ex.get("image_link", "")
        statement = str(ex.get("caption", "")).strip()
        label = int(ex.get("label", 0))
        relation = ex.get("relation", "")

        if not statement or not url:
            continue

        # Load/cache image
        url_hash = hashlib.md5(url.encode()).hexdigest()
        img_cache = os.path.join(IMAGE_CACHE_DIR, f"{url_hash}.jpg")
        try:
            if os.path.exists(img_cache):
                img = Image.open(img_cache).convert("RGB")
            else:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                img.save(img_cache, "JPEG")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        prompt = _build_vsr_prompt(statement)
        input_ids, attention_mask, pixel_values = process_vlm_inputs(img, prompt, processor, model)

        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            logits = out.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)[0]

        yes_mass = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
        no_mass = probs[list(no_ids)].sum().item() if no_ids else 0.0
        denom = yes_mass + no_mass
        p_yes = yes_mass / denom if denom > 0 else 0.0
        pred = 1 if p_yes > 0.5 else 0

        vsr_total += 1
        if pred == label:
            vsr_correct += 1

        if relation not in vsr_per_relation:
            vsr_per_relation[relation] = {"correct": 0, "total": 0}
        vsr_per_relation[relation]["total"] += 1
        if pred == label:
            vsr_per_relation[relation]["correct"] += 1

        if relation in CONTROL_RELATIONS:
            vsr_ctrl_total += 1
            if pred == label:
                vsr_ctrl_correct += 1

    vsr_acc = (vsr_correct / vsr_total * 100) if vsr_total > 0 else 0.0
    vsr_ctrl_acc = (vsr_ctrl_correct / vsr_ctrl_total * 100) if vsr_ctrl_total > 0 else 0.0

    # Evaluate VQA yes/no baseline
    print("[Baseline] Loading VQA yes/no...")
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")
    vqa_correct = 0
    vqa_total = 0

    for i in tqdm(range(min(VQA_MAX_SAMPLES, len(vqa))), desc="VQA baseline"):
        ex = vqa[i]
        question = ex.get("question", "")
        answers = ex.get("answers", [])
        img = ex.get("image")

        if not question or img is None:
            continue

        # Check if yes/no question
        answer_set = set()
        for a in answers:
            if isinstance(a, dict):
                answer_set.add(a.get("answer", "").lower().strip())
            else:
                answer_set.add(str(a).lower().strip())
        if not (answer_set & {"yes", "no"}):
            continue

        majority = max(answer_set, key=lambda x: sum(1 for a in answers if (a.get("answer", "") if isinstance(a, dict) else str(a)).lower().strip() == x))
        label = 1 if majority == "yes" else 0

        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            continue

        prompt = f"Answer with Yes or No: {question}"
        input_ids, attention_mask, pixel_values = process_vlm_inputs(img, prompt, processor, model)

        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            logits = out.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)[0]

        yes_mass = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
        no_mass = probs[list(no_ids)].sum().item() if no_ids else 0.0
        denom = yes_mass + no_mass
        p_yes = yes_mass / denom if denom > 0 else 0.0
        pred = 1 if p_yes > 0.5 else 0

        vqa_total += 1
        if pred == label:
            vqa_correct += 1

    vqa_acc = (vqa_correct / vqa_total * 100) if vqa_total > 0 else 0.0

    result = {
        "vsr_acc": vsr_acc,
        "vsr_correct": vsr_correct,
        "vsr_total": vsr_total,
        "vsr_ctrl_acc": vsr_ctrl_acc,
        "vsr_ctrl_correct": vsr_ctrl_correct,
        "vsr_ctrl_total": vsr_ctrl_total,
        "vsr_per_relation": {k: v for k, v in vsr_per_relation.items()},
        "vqa_acc": vqa_acc,
        "vqa_correct": vqa_correct,
        "vqa_total": vqa_total,
    }

    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()

    print(f"[Baseline] VSR: {vsr_acc:.2f}% ({vsr_correct}/{vsr_total})")
    print(f"[Baseline] VSR Ctrl: {vsr_ctrl_acc:.2f}% ({vsr_ctrl_correct}/{vsr_ctrl_total})")
    print(f"[Baseline] VQA: {vqa_acc:.2f}% ({vqa_correct}/{vqa_total})")

    return result


# --------------- Per-feature ablation worker ---------------

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/vol": volume},
    timeout=TIMEOUT,
)
def ablation_worker(worker_id: int, feature_assignments: list):
    """Ablate features and measure ∆VSR, ∆VQA, ∆Ctrl.

    feature_assignments: list of (layer_idx, feature_idx) tuples
    """
    import sys
    import hashlib
    import torch
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    from datasets import load_dataset, concatenate_datasets
    from nnsight import NNsight
    from PIL import Image
    from io import BytesIO
    import requests

    sys.path.insert(0, "/root/paligemma2")
    from utils import initialize_vlm_model, process_vlm_inputs, get_image_token_positions, initialize_jumprelu_sae, initialize_sae

    out_dir = Path(RESULTS_BASE) / "analysis" / "ablation_jumprelu"
    out_dir.mkdir(parents=True, exist_ok=True)

    sae_suffix = "_jumprelu" if SAE_TYPE == "jumprelu" else ""
    ckpt_dir = Path(RESULTS_BASE) / f"run{sae_suffix}" / "checkpoints"

    print(f"[Ablation W{worker_id}] {len(feature_assignments)} features to ablate")

    if not feature_assignments:
        return []

    # Load model
    print(f"[Ablation W{worker_id}] Loading PaliGemma2...")
    processor, model_raw = initialize_vlm_model(MODEL_NAME, device="cuda")
    model_raw.eval()
    nns_model = NNsight(model_raw)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)

    # Load VSR
    print(f"[Ablation W{worker_id}] Loading VSR...")
    vsr_splits = []
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    for split in VSR_SPLITS:
        vsr_splits.append(load_dataset(VSR_DATASET, data_files=data_files, split=split))
    vsr = concatenate_datasets(vsr_splits)

    # Load VQA yes/no
    print(f"[Ablation W{worker_id}] Loading VQA...")
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")

    # Pre-filter VQA to yes/no questions
    vqa_yesno_indices = []
    vqa_labels = {}
    for i in range(min(VQA_MAX_SAMPLES, len(vqa))):
        answers = vqa[i].get("answers", [])
        answer_set = set()
        for a in answers:
            if isinstance(a, dict):
                answer_set.add(a.get("answer", "").lower().strip())
            else:
                answer_set.add(str(a).lower().strip())
        if answer_set & {"yes", "no"}:
            majority = max(answer_set, key=lambda x: sum(1 for a in answers if (a.get("answer", "") if isinstance(a, dict) else str(a)).lower().strip() == x))
            vqa_labels[i] = 1 if majority == "yes" else 0
            vqa_yesno_indices.append(i)

    print(f"[Ablation W{worker_id}] VQA yes/no: {len(vqa_yesno_indices)} samples")

    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

    # Group features by layer to reuse SAE
    from collections import defaultdict
    layer_features = defaultdict(list)
    for (layer_idx, feature_idx) in feature_assignments:
        layer_features[layer_idx].append(feature_idx)

    results = []

    for layer_idx in tqdm(sorted(layer_features.keys()), desc=f"W{worker_id} layers"):
        features = layer_features[layer_idx]

        # Load SAE
        ckpt_path = ckpt_dir / f"pretrained_layer_{layer_idx}.pt"
        if not ckpt_path.exists():
            print(f"[Ablation W{worker_id}] SKIP L{layer_idx} — no checkpoint")
            continue

        if SAE_TYPE == "jumprelu":
            sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                          device="cuda", cache_dir="/vol/cache/huggingface")
        else:
            sae = initialize_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                 device="cuda", cache_dir="/vol/cache/huggingface")
        sae.eval()

        # Get feature vector (decoder direction, unit-normalized)
        model_dtype = next(nns_model._module.parameters()).dtype

        for feature_idx in tqdm(features, desc=f"W{worker_id} L{layer_idx}", leave=False):
            feature_vec = sae.W_dec[feature_idx].detach().to(model_dtype).to("cuda")
            feature_vec = feature_vec / feature_vec.norm().clamp(min=1e-8)

            # --- VSR ablation ---
            vsr_correct = 0
            vsr_total = 0
            vsr_ctrl_correct = 0
            vsr_ctrl_total = 0
            vsr_per_relation = {}

            for vi in range(len(vsr)):
                ex = vsr[vi]
                url = ex.get("image_link", "")
                statement = str(ex.get("caption", "")).strip()
                label = int(ex.get("label", 0))
                relation = ex.get("relation", "")

                if not statement or not url:
                    continue

                url_hash = hashlib.md5(url.encode()).hexdigest()
                img_cache = os.path.join(IMAGE_CACHE_DIR, f"{url_hash}.jpg")
                try:
                    if os.path.exists(img_cache):
                        img = Image.open(img_cache).convert("RGB")
                    else:
                        resp = requests.get(url, timeout=10)
                        resp.raise_for_status()
                        img = Image.open(BytesIO(resp.content)).convert("RGB")
                        img.save(img_cache, "JPEG")
                except Exception:
                    img = Image.new("RGB", (224, 224), (128, 128, 128))

                prompt = _build_vsr_prompt(statement)
                input_ids, attention_mask, pixel_values = process_vlm_inputs(img, prompt, processor, nns_model._module)

                try:
                    with nns_model.trace(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                    ) as tr:
                        # Ablate: project out the feature direction from layer output
                        layer_output = nns_model.model.language_model.layers[layer_idx].output[0]
                        # Project out feature: h = h - (h . d) * d
                        proj = (layer_output @ feature_vec.unsqueeze(-1)) * feature_vec.unsqueeze(0).unsqueeze(0)
                        nns_model.model.language_model.layers[layer_idx].output[0][:] = layer_output - proj

                        logits_saved = nns_model.output.logits.save()

                    logits = logits_saved[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)[0]
                    yes_mass = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
                    no_mass = probs[list(no_ids)].sum().item() if no_ids else 0.0
                    denom = yes_mass + no_mass
                    p_yes = yes_mass / denom if denom > 0 else 0.0
                    pred = 1 if p_yes > 0.5 else 0
                except Exception as e:
                    pred = 0

                vsr_total += 1
                if pred == label:
                    vsr_correct += 1

                if relation not in vsr_per_relation:
                    vsr_per_relation[relation] = {"correct": 0, "total": 0}
                vsr_per_relation[relation]["total"] += 1
                if pred == label:
                    vsr_per_relation[relation]["correct"] += 1

                if relation in CONTROL_RELATIONS:
                    vsr_ctrl_total += 1
                    if pred == label:
                        vsr_ctrl_correct += 1

            vsr_acc = (vsr_correct / vsr_total * 100) if vsr_total > 0 else 0.0
            vsr_ctrl_acc = (vsr_ctrl_correct / vsr_ctrl_total * 100) if vsr_ctrl_total > 0 else 0.0

            # --- VQA ablation ---
            vqa_correct = 0
            vqa_total = 0

            for qi in vqa_yesno_indices:
                ex = vqa[qi]
                question = ex.get("question", "")
                img = ex.get("image")
                label = vqa_labels[qi]

                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                elif not isinstance(img, Image.Image):
                    continue

                prompt = f"Answer with Yes or No: {question}"
                input_ids, attention_mask, pixel_values = process_vlm_inputs(img, prompt, processor, nns_model._module)

                try:
                    with nns_model.trace(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                    ) as tr:
                        layer_output = nns_model.model.language_model.layers[layer_idx].output[0]
                        proj = (layer_output @ feature_vec.unsqueeze(-1)) * feature_vec.unsqueeze(0).unsqueeze(0)
                        nns_model.model.language_model.layers[layer_idx].output[0][:] = layer_output - proj

                        logits_saved = nns_model.output.logits.save()

                    logits = logits_saved[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)[0]
                    yes_mass = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
                    no_mass = probs[list(no_ids)].sum().item() if no_ids else 0.0
                    denom = yes_mass + no_mass
                    p_yes = yes_mass / denom if denom > 0 else 0.0
                    pred = 1 if p_yes > 0.5 else 0
                except Exception:
                    pred = 0

                vqa_total += 1
                if pred == label:
                    vqa_correct += 1

            vqa_acc = (vqa_correct / vqa_total * 100) if vqa_total > 0 else 0.0

            result = {
                "layer": layer_idx,
                "feature": feature_idx,
                "vsr_acc": vsr_acc,
                "vsr_correct": vsr_correct,
                "vsr_total": vsr_total,
                "vsr_ctrl_acc": vsr_ctrl_acc,
                "vqa_acc": vqa_acc,
                "vqa_correct": vqa_correct,
                "vqa_total": vqa_total,
                "vsr_per_relation": vsr_per_relation,
            }
            results.append(result)
            print(f"[Ablation W{worker_id}] L{layer_idx}/F{feature_idx}: VSR={vsr_acc:.2f}%, VQA={vqa_acc:.2f}%")

        # Save per-worker results
        worker_path = out_dir / f"ablation_w{worker_id}.json"
        with open(worker_path, "w") as f:
            json.dump(results, f, indent=2)
        volume.commit()

        del sae
        torch.cuda.empty_cache()

    return results


# --------------- Load features from volume ---------------

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=120,
)
def load_features():
    """Load the final spatial-visual feature list from volume."""
    import pandas as pd
    from pathlib import Path

    sae_suffix = "_jumprelu" if SAE_TYPE == "jumprelu" else ""
    csv_path = Path(RESULTS_BASE) / "analysis" / f"final_features{sae_suffix}" / "final_spatial_visual_features.csv"
    if not csv_path.exists():
        print(f"[WARN] No final features CSV at {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    features = [(int(row["layer"]), int(row["feature"])) for _, row in df.iterrows()]
    print(f"[INFO] {len(features)} features to ablate")
    return features


# --------------- Merge and produce final table ---------------

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=600,
)
def merge_ablation_results(baseline: dict):
    """Merge per-worker ablation results into the final table."""
    import pandas as pd
    from pathlib import Path

    abl_dir = Path(RESULTS_BASE) / "analysis" / "ablation_jumprelu"
    sae_suffix = "_jumprelu" if SAE_TYPE == "jumprelu" else ""

    # Merge all worker results
    all_results = []
    for f_path in sorted(abl_dir.glob("ablation_w*.json")):
        with open(f_path) as f:
            all_results.extend(json.load(f))

    if not all_results:
        print("[WARN] No ablation results found")
        return {}

    # Load odds ratios from spatial features analysis
    spatial_dir = Path(RESULTS_BASE) / "analysis" / f"spatial{sae_suffix}"
    spatial_path = spatial_dir / "spatial_features.csv"
    feature_or = {}
    if spatial_path.exists():
        df_spatial = pd.read_csv(spatial_path)
        for _, row in df_spatial.iterrows():
            key = (int(row["layer"]), int(row["feature"]))
            feature_or[key] = float(row.get("odds_ratio", 0))

    # Build results table
    baseline_vsr = baseline["vsr_acc"]
    baseline_vqa = baseline["vqa_acc"]
    baseline_ctrl = baseline["vsr_ctrl_acc"]
    baseline_per_relation = baseline.get("vsr_per_relation", {})

    rows = []
    for r in all_results:
        layer = r["layer"]
        feature = r["feature"]
        delta_vsr = r["vsr_acc"] - baseline_vsr
        delta_vqa = r["vqa_acc"] - baseline_vqa
        delta_ctrl = r["vsr_ctrl_acc"] - baseline_ctrl

        # Derive top affected relation: relation with largest accuracy drop
        top_relation = ""
        max_drop = 0.0
        per_rel = r.get("vsr_per_relation", {})
        for rel, stats in per_rel.items():
            if rel in CONTROL_RELATIONS:
                continue
            bl = baseline_per_relation.get(rel, {})
            bl_acc = (bl["correct"] / bl["total"] * 100) if bl.get("total", 0) > 0 else 0.0
            abl_acc = (stats["correct"] / stats["total"] * 100) if stats.get("total", 0) > 0 else 0.0
            drop = bl_acc - abl_acc
            if drop > max_drop:
                max_drop = drop
                top_relation = rel

        vsr_or = feature_or.get((layer, feature), 0.0)

        rows.append({
            "Layer": layer,
            "Feature": feature,
            "∆VSR Acc": round(delta_vsr, 2),
            "∆VQA Acc": round(delta_vqa, 2),
            "∆Ctrl VSR": round(delta_ctrl, 2),
            "Relation": top_relation,
            "VSR OR": round(vsr_or, 2),
            "Ablated VSR": round(r["vsr_acc"], 2),
            "Ablated VQA": round(r["vqa_acc"], 2),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("∆VSR Acc", ascending=True)  # Most negative first

    # Save
    csv_path = abl_dir / "ablation_results_table.csv"
    df.to_csv(csv_path, index=False)

    # Print top features
    print("\n" + "=" * 80)
    print("TOP ABLATED SAE FEATURES (ranked by VSR accuracy drop)")
    print("=" * 80)
    print(f"Baseline: VSR={baseline_vsr:.2f}%, VQA={baseline_vqa:.2f}%, Ctrl={baseline_ctrl:.2f}%")
    print("-" * 80)
    print(f"{'Layer':>5} {'Feature':>8} {'∆VSR':>8} {'∆VQA':>8} {'∆Ctrl':>8} {'Relation':<25} {'OR':>6}")
    print("-" * 80)
    for _, row in df.head(20).iterrows():
        print(f"{row['Layer']:>5} {row['Feature']:>8} {row['∆VSR Acc']:>8.2f} {row['∆VQA Acc']:>8.2f} {row['∆Ctrl VSR']:>8.2f} {row['Relation']:<25} {row['VSR OR']:>6.2f}")
    print("=" * 80)

    summary = {
        "baseline_vsr": baseline_vsr,
        "baseline_vqa": baseline_vqa,
        "baseline_ctrl": baseline_ctrl,
        "n_features_ablated": len(all_results),
        "top_10": df.head(10).to_dict(orient="records"),
    }

    with open(abl_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()

    print(f"\n[SUCCESS] Ablation table saved to {csv_path}")
    return summary


# --------------- Main entrypoint ---------------

@app.local_entrypoint()
def main():
    import math

    # Step 1: Get the list of final features to ablate
    print("[Step 1] Loading final feature list...")
    features_to_ablate = load_features.remote()
    print(f"[INFO] {len(features_to_ablate)} features to ablate")

    if not features_to_ablate:
        print("[ERROR] No features found. Run modal_analysis.py first.")
        return

    # Step 2: Compute baseline (1 GPU)
    print("\n[Step 2] Computing baseline VSR + VQA accuracy...")
    baseline = compute_baseline.remote()
    print(f"Baseline: VSR={baseline['vsr_acc']:.2f}%, VQA={baseline['vqa_acc']:.2f}%")

    # Step 3: Distribute features across workers for ablation
    print(f"\n[Step 3] Ablating {len(features_to_ablate)} features across {N_GPUS} GPUs...")
    n_features = len(features_to_ablate)
    features_per_worker = math.ceil(n_features / N_GPUS)

    assignments = []
    for w in range(N_GPUS):
        start = w * features_per_worker
        end = min(start + features_per_worker, n_features)
        worker_features = features_to_ablate[start:end]
        if worker_features:
            assignments.append((w, worker_features))

    print(f"  {len(assignments)} workers, {n_features} features")
    ablation_results = list(ablation_worker.starmap(assignments))
    for r in ablation_results:
        if isinstance(r, list):
            print(f"  Worker returned {len(r)} results")

    # Step 4: Merge and produce final table
    print("\n[Step 4] Merging results into final table...")
    summary = merge_ablation_results.remote(baseline)

    print("\n[DONE] Ablation experiments complete!")
    if summary and "top_10" in summary:
        print(f"Top feature by ∆VSR: L{summary['top_10'][0]['Layer']}F{summary['top_10'][0]['Feature']} ({summary['top_10'][0]['∆VSR Acc']:.2f}%)")
