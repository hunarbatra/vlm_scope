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
    import warnings
    import numpy as np
    from pathlib import Path
    from tqdm import tqdm
    from datasets import load_dataset, concatenate_datasets
    from PIL import Image
    from io import BytesIO
    import requests

    warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

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

    # Evaluate VQA yes/no baseline — scan full dataset for yes/no questions
    print("[Baseline] Loading VQA yes/no...")
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")
    vqa_yesno_indices = []
    for i in range(len(vqa)):
        ex = vqa[i]
        at = str(ex.get("answer_type", "")).lower()
        mc = str(ex.get("multiple_choice_answer", "")).strip().lower()
        if at == "yes/no" and mc in {"yes", "no"}:
            vqa_yesno_indices.append((i, 1 if mc == "yes" else 0))
            if len(vqa_yesno_indices) >= VQA_MAX_SAMPLES:
                break
    print(f"[Baseline] VQA yes/no: {len(vqa_yesno_indices)} samples")

    vqa_correct = 0
    vqa_total = 0

    for i, (qi, label) in enumerate(tqdm(vqa_yesno_indices, desc="VQA baseline")):
        ex = vqa[qi]
        question = ex.get("question", "")
        img = ex.get("image")

        if not question or img is None:
            continue

        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            continue

        prompt = (
                    "Answer the following question with only 'Yes' or 'No':\n"
                    f"Question: {question.strip()}\n"
                    "Answer:"
                )
        try:
            input_ids, attention_mask, pixel_values = process_vlm_inputs(img, prompt, processor, model)
        except Exception:
            continue

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
    """Ablate features and measure ∆VSR, ∆VQA, ∆Ctrl VSR.

    Matches the original ablation approach:
    - Projects out feature direction from ALL transformer layers (apply_to_all_layers)
    - Only ablates on text tokens (after img_end), not image tokens
    - Tracks per-relation VSR accuracy for relation mapping + control

    feature_assignments: list of (layer_idx, feature_idx) tuples, pre-sorted by priority
    """
    import sys
    import hashlib
    import torch
    import numpy as np
    import warnings
    from pathlib import Path
    from tqdm import tqdm
    from datasets import load_dataset, concatenate_datasets
    from nnsight import NNsight
    from PIL import Image
    from io import BytesIO
    import requests

    # Suppress PaliGemma processor warnings
    warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")

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
    n_model_layers = model_raw.config.text_config.num_hidden_layers  # 26

    # Load VSR (all splits)
    print(f"[Ablation W{worker_id}] Loading VSR...")
    vsr_splits = []
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    for split in VSR_SPLITS:
        vsr_splits.append(load_dataset(VSR_DATASET, data_files=data_files, split=split))
    vsr = concatenate_datasets(vsr_splits)

    # Load VQA yes/no — scan full dataset until we find VQA_MAX_SAMPLES yes/no questions
    print(f"[Ablation W{worker_id}] Loading VQA yes/no (scanning for {VQA_MAX_SAMPLES} yes/no)...")
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")
    vqa_yesno_indices = []
    vqa_labels = {}
    for i in range(len(vqa)):
        ex = vqa[i]
        at = str(ex.get("answer_type", "")).lower()
        mc = str(ex.get("multiple_choice_answer", "")).strip().lower()
        if at == "yes/no" and mc in {"yes", "no"}:
            vqa_labels[i] = 1 if mc == "yes" else 0
            vqa_yesno_indices.append(i)
            if len(vqa_yesno_indices) >= VQA_MAX_SAMPLES:
                break

    print(f"[Ablation W{worker_id}] VQA yes/no: {len(vqa_yesno_indices)} samples")

    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

    # Pre-load SAEs for all layers that have features assigned
    # (we need them for all layers when doing apply_to_all_layers)
    from collections import defaultdict
    layer_features = defaultdict(list)
    for (layer_idx, feature_idx) in feature_assignments:
        layer_features[layer_idx].append(feature_idx)

    results = []
    model_dtype = next(nns_model._module.parameters()).dtype

    def _do_ablation_trace(input_ids, attention_mask, pixel_values, feature_vec, img_end):
        """Run NNsight trace with feature ablation on text tokens across all layers.

        Matches original ablate_sae_feature_vsr.py: --apply-to-all-layers --steering-coef -1.0

        NNsight 0.6.x: each .output access sends a VALUE request that fires once.
        Access .output ONCE per module, modify the view in-place with -=.
        Do NOT re-access .output on the same module (causes deadlock in 0.6.x).
        """
        fv = feature_vec.unsqueeze(0)  # (1, d_in)

        with nns_model.trace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        ) as tr:
            for l in range(n_model_layers):
                # 1. Self-attention output — access ONCE, modify view in-place
                attn_out = nns_model.model.language_model.layers[l].self_attn.output[0][0, img_end:]
                attn_proj = (attn_out @ fv.T) * fv
                attn_out -= attn_proj

                # 2. MLP output — access ONCE, modify view in-place
                mlp_out = nns_model.model.language_model.layers[l].mlp.output[0, img_end:]
                mlp_proj = (mlp_out @ fv.T) * fv
                mlp_out -= mlp_proj

                # 3. Layer output (residual stream) — access ONCE, modify view in-place
                # Note: layer.output[0] has shape (seq, d_in) — no batch dim
                layer_out = nns_model.model.language_model.layers[l].output[0][img_end:]
                layer_proj = (layer_out @ fv.T) * fv
                layer_out -= layer_proj

            logits_saved = nns_model.output.logits.save()
        return logits_saved

    def _predict_yesno(logits_saved):
        """Extract yes/no prediction from logits."""
        logits = logits_saved[:, -1, :]
        probs = torch.softmax(logits, dim=-1)[0]
        yes_mass = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
        no_mass = probs[list(no_ids)].sum().item() if no_ids else 0.0
        denom = yes_mass + no_mass
        p_yes = yes_mass / denom if denom > 0 else 0.0
        return 1 if p_yes > 0.5 else 0

    # Process features in priority order (already sorted by load_features)
    for feat_i, (layer_idx, feature_idx) in enumerate(feature_assignments):
        # Load SAE for this feature's layer (to get W_dec direction)
        ckpt_path = ckpt_dir / f"pretrained_layer_{layer_idx}.pt"
        if not ckpt_path.exists():
            print(f"[Ablation W{worker_id}] SKIP L{layer_idx}/F{feature_idx} — no checkpoint")
            continue

        if SAE_TYPE == "jumprelu":
            sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                          device="cuda", cache_dir="/vol/cache/huggingface")
        else:
            sae = initialize_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                 device="cuda", cache_dir="/vol/cache/huggingface")
        sae.eval()

        feature_vec = sae.W_dec[feature_idx].detach().to(model_dtype).to("cuda")
        raw_norm = feature_vec.norm().item()
        feature_vec = feature_vec / feature_vec.norm().clamp(min=1e-8)
        print(f"[Ablation W{worker_id}] L{layer_idx}/F{feature_idx}: vec norm={raw_norm:.4f}, first3={feature_vec[:3].tolist()}")

        del sae
        torch.cuda.empty_cache()

        # --- VSR ablation ---
        vsr_correct = 0
        vsr_total = 0
        vsr_ctrl_correct = 0
        vsr_ctrl_total = 0
        vsr_per_relation = {}

        for vi in tqdm(range(len(vsr)), desc=f"W{worker_id} [{feat_i+1}/{len(feature_assignments)}] L{layer_idx}F{feature_idx} VSR", leave=False):
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
            _, img_end = get_image_token_positions(input_ids)

            try:
                logits_saved = _do_ablation_trace(input_ids, attention_mask, pixel_values, feature_vec, img_end)
                pred = _predict_yesno(logits_saved)
            except Exception as e:
                if vi < 3:
                    print(f"  [ERROR] W{worker_id} L{layer_idx}F{feature_idx} VSR sample {vi}: {e}")
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

            if (vi + 1) % 100 == 0 and vsr_total > 0:
                run_acc = vsr_correct / vsr_total * 100
                print(f"  W{worker_id} L{layer_idx}F{feature_idx} VSR [{vi+1}/{len(vsr)}] acc={run_acc:.2f}%")

        vsr_acc = (vsr_correct / vsr_total * 100) if vsr_total > 0 else 0.0
        vsr_ctrl_acc = (vsr_ctrl_correct / vsr_ctrl_total * 100) if vsr_ctrl_total > 0 else 0.0

        # --- VQA ablation ---
        vqa_correct = 0
        vqa_total = 0

        for qi in tqdm(vqa_yesno_indices, desc=f"W{worker_id} L{layer_idx}F{feature_idx} VQA", leave=False):
            ex = vqa[qi]
            question = ex.get("question", "")
            img = ex.get("image")
            label = vqa_labels[qi]

            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                img = img.convert("RGB")
            else:
                continue

            prompt = (
                    "Answer the following question with only 'Yes' or 'No':\n"
                    f"Question: {question.strip()}\n"
                    "Answer:"
                )
            input_ids, attention_mask, pixel_values = process_vlm_inputs(img, prompt, processor, nns_model._module)
            _, img_end = get_image_token_positions(input_ids)

            try:
                logits_saved = _do_ablation_trace(input_ids, attention_mask, pixel_values, feature_vec, img_end)
                pred = _predict_yesno(logits_saved)
            except Exception as e:
                if vqa_total < 3:
                    print(f"  [ERROR] W{worker_id} L{layer_idx}F{feature_idx} VQA sample {vqa_total}: {e}")
                pred = 0

            vqa_total += 1
            if pred == label:
                vqa_correct += 1

            if vqa_total % 100 == 0:
                run_acc = vqa_correct / vqa_total * 100
                print(f"  W{worker_id} L{layer_idx}F{feature_idx} VQA [{vqa_total}/{len(vqa_yesno_indices)}] acc={run_acc:.2f}%")

        vqa_acc = (vqa_correct / vqa_total * 100) if vqa_total > 0 else 0.0

        result = {
            "layer": layer_idx,
            "feature": feature_idx,
            "vsr_acc": vsr_acc,
            "vsr_correct": vsr_correct,
            "vsr_total": vsr_total,
            "vsr_ctrl_acc": vsr_ctrl_acc,
            "vsr_ctrl_correct": vsr_ctrl_correct,
            "vsr_ctrl_total": vsr_ctrl_total,
            "vqa_acc": vqa_acc,
            "vqa_correct": vqa_correct,
            "vqa_total": vqa_total,
            "vsr_per_relation": vsr_per_relation,
        }
        results.append(result)
        print(f"[Ablation W{worker_id}] L{layer_idx}/F{feature_idx}: VSR={vsr_acc:.2f}% Ctrl={vsr_ctrl_acc:.2f}% VQA={vqa_acc:.2f}%")

        # Save after every feature (resume-friendly)
        worker_path = out_dir / f"ablation_w{worker_id}.json"
        with open(worker_path, "w") as f:
            json.dump(results, f, indent=2)
        volume.commit()

        torch.cuda.empty_cache()

    return results


# --------------- Load features from volume ---------------

@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=120,
)
def load_features():
    """Load final features ranked by adaptation strength + spatial relevance.

    Priority score = (1 - cosine_sim) * odds_ratio * Ev
    Features with low cosine (more adapted), high odds ratio (more spatial),
    and high Ev (more visual) are ranked first.
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path

    sae_suffix = "_jumprelu" if SAE_TYPE == "jumprelu" else ""
    analysis_dir = Path(RESULTS_BASE) / "analysis"
    csv_path = analysis_dir / f"final_features{sae_suffix}" / "final_spatial_visual_features.csv"
    if not csv_path.exists():
        print(f"[WARN] No final features CSV at {csv_path}")
        return []
    df_final = pd.read_csv(csv_path)

    # Load spatial features for odds_ratio
    spatial_path = analysis_dir / f"spatial{sae_suffix}" / "spatial_features.csv"
    or_map = {}
    if spatial_path.exists():
        df_sp = pd.read_csv(spatial_path)
        for _, row in df_sp.iterrows():
            or_map[(int(row["layer"]), int(row["feature"]))] = float(row.get("odds_ratio", 1.0))

    # Load cosine similarities per layer
    cosine_dir = analysis_dir / f"cosines{sae_suffix}"
    cosine_map = {}
    if cosine_dir.exists():
        for layer_idx in range(N_LAYERS):
            cos_path = cosine_dir / f"cosines_layer_{layer_idx}.npy"
            if cos_path.exists():
                cosines = np.load(cos_path)
                for fi in range(len(cosines)):
                    cosine_map[(layer_idx, fi)] = float(cosines[fi])

    # Load Ev per layer
    energy_dir = analysis_dir / f"energy{sae_suffix}"
    ev_map = {}
    if energy_dir.exists():
        for layer_idx in range(N_LAYERS):
            ev_path = energy_dir / f"Ev_layer_{layer_idx}.npy"
            if ev_path.exists():
                evs = np.load(ev_path)
                for fi in range(len(evs)):
                    ev_map[(layer_idx, fi)] = float(evs[fi])

    # Compute priority score and sort
    scored = []
    for _, row in df_final.iterrows():
        layer, feature = int(row["layer"]), int(row["feature"])
        key = (layer, feature)
        cos = cosine_map.get(key, 0.9)
        odds = or_map.get(key, 1.0)
        ev = ev_map.get(key, 0.01)
        # Higher score = more promising for ablation
        score = (1.0 - cos) * odds * ev
        scored.append((layer, feature, score, cos, odds, ev))

    scored.sort(key=lambda x: -x[2])  # Highest score first

    print(f"[INFO] {len(scored)} features ranked by priority (adaptation * spatial * visual)")
    print(f"  Top 10:")
    for i, (l, f, s, c, o, e) in enumerate(scored[:10]):
        print(f"    {i+1}. L{l}/F{f}: score={s:.4f} (cos={c:.3f}, OR={o:.1f}, Ev={e:.3f})")
    print(f"  Bottom 5:")
    for l, f, s, c, o, e in scored[-5:]:
        print(f"    L{l}/F{f}: score={s:.6f} (cos={c:.3f}, OR={o:.1f}, Ev={e:.3f})")

    features = [(l, f) for l, f, *_ in scored]
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
