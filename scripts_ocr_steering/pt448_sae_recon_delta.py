#!/usr/bin/env python3
"""
SAE Reconstruction Delta Injection: mix-448 → pt-448.

Core idea:
  The SAE reconstruction r = decode(encode(h)) is the SAE-decodable component of h.
  mix-448 has stronger spatial SAE activations than pt-448.
  Injecting alpha*(r_mix[l] - r_pt[l]) at each spatial layer transfers exactly the
  "spatial SAE knowledge gap" — all features weighted by their natural firing rate,
  only the difference.

Three phases (set via PHASE env var):
  PHASE=1  Extract mix-448 SAE reconstructions → mix_reconstructions/
  PHASE=2  Extract pt-448 SAE reconstructions + base predictions → pt_reconstructions/
  PHASE=3  Pre-compute deltas, run injection alpha sweep, save oracle_results.json

Usage:
  CUDA_VISIBLE_DEVICES=4 PHASE=1 python3 pt448_sae_recon_delta.py
  CUDA_VISIBLE_DEVICES=5 PHASE=2 python3 pt448_sae_recon_delta.py
  CUDA_VISIBLE_DEVICES=4 PHASE=3 python3 pt448_sae_recon_delta.py
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

# ─────────────────────────── Config ───────────────────────────

MIX_MODEL      = "google/paligemma2-3b-mix-448"
PT_MODEL       = "google/paligemma2-3b-pt-448"
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR        = Path(os.environ.get("OUT_DIR", "/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_sae_recon_delta"))
RECON_CACHE    = OUT_DIR / "mix_reconstructions"
PT_RECON_CACHE = OUT_DIR / "pt_reconstructions"
DELTA_CACHE    = OUT_DIR / "deltas"
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

SAE_LAYERS = list(range(26))  # all 26 layers — full SAE-structured gap, not just top-feature layers
ALPHAS     = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5]

PHASE = os.environ.get("PHASE", "1")


# ─────────────────────────── Helpers ───────────────────────────

def _build_vsr_prompt(statement):
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\nAnswer:"
    )


def _get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks:
            no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap
    no_ids  -= overlap
    return yes_ids, no_ids


def _predict_and_margin(logits, yes_ids, no_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item()  if no_ids  else 1e-9
    d = y + n
    p_yes = max(y / d if d > 0 else 0.5, 1e-7)
    p_no  = max(1.0 - p_yes, 1e-7)
    return (1 if p_yes > 0.5 else 0), math.log(p_yes / p_no)


def _load_image(ex):
    url = ex.get("image_link", "")
    if not url.startswith("http"):
        return None
    h  = hashlib.md5(url.encode()).hexdigest()
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


def _sae_recon_cpu(h_float_cpu, W_enc, b_enc, threshold, W_dec):
    """Compute JumpReLU SAE reconstruction on CPU.

    Args:
        h_float_cpu : [T_text, 2304] float32 CPU tensor
        W_enc       : [2304, 16384] float32 CPU tensor
        b_enc       : [16384]       float32 CPU tensor
        threshold   : [16384]       float32 CPU tensor
        W_dec       : [16384, 2304] float32 CPU tensor

    Returns:
        recon_mean  : [2304] bfloat16 CPU tensor — mean over text tokens
    """
    with torch.no_grad():
        pre_act = h_float_cpu @ W_enc + b_enc        # [T_text, 16384]
        acts    = torch.relu(pre_act - threshold)    # [T_text, 16384]
        recon   = acts @ W_dec                       # [T_text, 2304]
        return recon.mean(0).to(torch.bfloat16)      # [2304]


def _load_sae_weights_cpu(layer):
    """Load a text-only SAE checkpoint and return raw weight tensors on CPU (float32)."""
    ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer}.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    W_enc     = state["W_enc"].float()      # [2304, 16384]
    W_dec     = state["W_dec"].float()      # [16384, 2304]
    b_enc     = state["b_enc"].float()      # [16384]
    threshold = state["threshold"].float()  # [16384]
    return W_enc, W_dec, b_enc, threshold


# ─────────────────────────── Phase 1: mix-448 reconstructions ───────────────────────────

def phase1_extract_mix_reconstructions():
    """Run mix-448 forward passes and save per-sample SAE reconstructions."""
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    print("=" * 80)
    print("PHASE 1: Extracting mix-448 SAE reconstructions")
    print("=" * 80, flush=True)

    device = "cuda:0"
    RECON_CACHE.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    # Load SAE weights to CPU (all 7 layers at once; ~2GB RAM)
    print("[INFO] Loading all 7 SAE checkpoints to CPU...", flush=True)
    sae_weights = {}
    for layer in SAE_LAYERS:
        sae_weights[layer] = _load_sae_weights_cpu(layer)
        print(f"  Loaded layer {layer}", flush=True)

    print(f"[INFO] Loading {MIX_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MIX_MODEL)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_model = NNsight(model_raw)

    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    N = len(vsr_all)
    print(f"[INFO] VSR total samples: {N}", flush=True)

    metadata = {}
    # Load existing metadata to support resuming
    meta_path = RECON_CACHE / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    n_skipped = n_done = n_failed = 0

    for vi in range(N):
        vi_key = str(vi)
        out_path = RECON_CACHE / f"vi_{vi:05d}.pt"

        if out_path.exists():
            n_skipped += 1
            if vi_key not in metadata:
                metadata[vi_key] = {"layers_computed": SAE_LAYERS, "failed": False}
            continue

        ex  = vsr_all[vi]
        img = _load_image(ex)
        if img is None:
            metadata[vi_key] = {"layers_computed": [], "failed": True}
            n_failed += 1
            continue

        prompt = _build_vsr_prompt(str(ex.get("caption", "")))

        try:
            iids, attn, pv = process_vlm_inputs(
                img, prompt, processor, model_raw, device=device
            )
            _, img_end = get_image_token_positions(iids)

            # Capture hidden states at all SAE layers in one forward pass
            saved_list = []
            with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                for l in SAE_LAYERS:
                    saved_list.append(nns_model.model.language_model.layers[l].output[0].save())

            # Compute SAE reconstructions on CPU
            recon_dict = {}
            for idx, l in enumerate(SAE_LAYERS):
                h_gpu   = saved_list[idx]              # [1, T, 2304]
                h_cpu   = h_gpu[0, img_end:, :].float().cpu()  # [T_text, 2304]
                W_enc, W_dec, b_enc, threshold = sae_weights[l]
                recon_dict[l] = _sae_recon_cpu(h_cpu, W_enc, b_enc, threshold, W_dec)
                del h_gpu

            torch.save(recon_dict, out_path)
            metadata[vi_key] = {"layers_computed": SAE_LAYERS, "failed": False}
            n_done += 1

        except Exception as e:
            metadata[vi_key] = {"layers_computed": [], "failed": True}
            n_failed += 1
            if n_failed <= 5:
                print(f"  [WARN] vi={vi} failed: {e}", flush=True)

        if (vi + 1) % 500 == 0 or vi == N - 1:
            print(
                f"  [{vi+1}/{N}] done={n_done} skipped={n_skipped} failed={n_failed}",
                flush=True,
            )
            # Save metadata checkpoint
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    total_ready = sum(
        1 for v in metadata.values() if not v["failed"] and v["layers_computed"]
    )
    print(f"\n[DONE] Phase 1 complete. {total_ready}/{N} samples saved to {RECON_CACHE}", flush=True)


# ─────────────────────────── Phase 2: pt-448 reconstructions + base predictions ───────────────────────────

def phase2_extract_pt_reconstructions():
    """Run pt-448 forward passes, save per-sample SAE reconstructions and base predictions."""
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    print("=" * 80)
    print("PHASE 2: Extracting pt-448 SAE reconstructions + base predictions")
    print("=" * 80, flush=True)

    device = "cuda:0"
    PT_RECON_CACHE.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading all 7 SAE checkpoints to CPU...", flush=True)
    sae_weights = {}
    for layer in SAE_LAYERS:
        sae_weights[layer] = _load_sae_weights_cpu(layer)
        print(f"  Loaded layer {layer}", flush=True)

    print(f"[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_model = NNsight(model_raw)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)

    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    N = len(vsr_all)
    print(f"[INFO] VSR total samples: {N}", flush=True)

    metadata = {}
    meta_path = PT_RECON_CACHE / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    # Base predictions are stored in a single JSON: {vi: {pred, margin, label}}
    base_preds_path = PT_RECON_CACHE / "base_predictions.json"
    base_preds = {}
    if base_preds_path.exists():
        with open(base_preds_path) as f:
            base_preds = json.load(f)

    n_skipped = n_done = n_failed = 0

    for vi in range(N):
        vi_key   = str(vi)
        out_path = PT_RECON_CACHE / f"vi_{vi:05d}.pt"

        if out_path.exists() and vi_key in base_preds:
            n_skipped += 1
            if vi_key not in metadata:
                metadata[vi_key] = {"layers_computed": SAE_LAYERS, "failed": False}
            continue

        ex  = vsr_all[vi]
        img = _load_image(ex)
        if img is None:
            metadata[vi_key] = {"layers_computed": [], "failed": True}
            n_failed += 1
            continue

        label  = int(ex.get("label", 0))
        prompt = _build_vsr_prompt(str(ex.get("caption", "")))

        try:
            iids, attn, pv = process_vlm_inputs(
                img, prompt, processor, model_raw, device=device
            )
            _, img_end = get_image_token_positions(iids)

            # Capture hidden states + logits in one forward pass
            saved_list2 = []
            with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                for l in SAE_LAYERS:
                    saved_list2.append(nns_model.model.language_model.layers[l].output[0].save())
                logits_s = nns_model.output.logits.save()

            # Base prediction
            pred, margin = _predict_and_margin(logits_s[0, -1, :], yes_ids, no_ids)
            base_preds[vi_key] = {
                "pred": pred, "margin": margin, "label": label,
                "correct": int(pred == label),
            }

            # Compute SAE reconstructions on CPU
            recon_dict = {}
            for idx, l in enumerate(SAE_LAYERS):
                h_gpu   = saved_list2[idx]
                h_cpu   = h_gpu[0, img_end:, :].float().cpu()
                W_enc, W_dec, b_enc, threshold = sae_weights[l]
                recon_dict[l] = _sae_recon_cpu(h_cpu, W_enc, b_enc, threshold, W_dec)
                del h_gpu

            if not out_path.exists():
                torch.save(recon_dict, out_path)
            metadata[vi_key] = {"layers_computed": SAE_LAYERS, "failed": False}
            n_done += 1

        except Exception as e:
            metadata[vi_key] = {"layers_computed": [], "failed": True}
            n_failed += 1
            if n_failed <= 5:
                print(f"  [WARN] vi={vi} failed: {e}", flush=True)

        if (vi + 1) % 500 == 0 or vi == N - 1:
            print(
                f"  [{vi+1}/{N}] done={n_done} skipped={n_skipped} failed={n_failed}",
                flush=True,
            )
            with open(meta_path, "w") as f:
                json.dump(metadata, f)
            with open(base_preds_path, "w") as f:
                json.dump(base_preds, f)

    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    with open(base_preds_path, "w") as f:
        json.dump(base_preds, f)

    total_ready = sum(
        1 for v in metadata.values() if not v["failed"] and v["layers_computed"]
    )
    base_acc = (
        sum(v["correct"] for v in base_preds.values()) / max(len(base_preds), 1) * 100
    )
    print(
        f"\n[DONE] Phase 2 complete. {total_ready}/{N} pt recons saved. "
        f"Base acc = {base_acc:.2f}% over {len(base_preds)} samples.",
        flush=True,
    )


# ─────────────────────────── Phase 3: delta injection sweep ───────────────────────────

def phase3_inject_delta():
    """Pre-compute deltas, run injection sweep over ALPHAS, save oracle_results.json."""
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    print("=" * 80)
    print("PHASE 3: SAE delta injection sweep")
    print("=" * 80, flush=True)

    device    = "cuda:0"
    DELTA_CACHE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Check availability ──
    mix_meta_path = RECON_CACHE / "metadata.json"
    pt_meta_path  = PT_RECON_CACHE / "metadata.json"
    base_preds_path = PT_RECON_CACHE / "base_predictions.json"

    if not mix_meta_path.exists():
        print("[ERROR] mix_reconstructions/metadata.json not found. Run PHASE=1 first.", flush=True)
        sys.exit(1)
    if not pt_meta_path.exists():
        print("[ERROR] pt_reconstructions/metadata.json not found. Run PHASE=2 first.", flush=True)
        sys.exit(1)
    if not base_preds_path.exists():
        print("[ERROR] pt_reconstructions/base_predictions.json not found. Run PHASE=2 first.", flush=True)
        sys.exit(1)

    with open(mix_meta_path) as f:
        mix_meta = json.load(f)
    with open(pt_meta_path) as f:
        pt_meta = json.load(f)
    with open(base_preds_path) as f:
        base_preds = json.load(f)

    # Identify samples where both mix and pt reconstructions are ready
    ready_vis = []
    for vi_key, mm in mix_meta.items():
        pm = pt_meta.get(vi_key, {})
        if not mm.get("failed") and mm.get("layers_computed") and \
           not pm.get("failed") and pm.get("layers_computed") and \
           vi_key in base_preds:
            ready_vis.append(int(vi_key))
    ready_vis.sort()

    n_mix_only   = sum(1 for k, v in mix_meta.items() if not v.get("failed") and v.get("layers_computed"))
    n_pt_only    = sum(1 for k, v in pt_meta.items()  if not v.get("failed") and v.get("layers_computed"))
    print(
        f"[INFO] mix recons ready: {n_mix_only}  pt recons ready: {n_pt_only}  "
        f"both ready: {len(ready_vis)}",
        flush=True,
    )

    # ── Pre-compute deltas ──
    print(f"\n[INFO] Pre-computing deltas for {len(ready_vis)} samples...", flush=True)
    n_delta_done = 0
    for vi in ready_vis:
        delta_path = DELTA_CACHE / f"vi_{vi:05d}.pt"
        if delta_path.exists():
            n_delta_done += 1
            continue
        mix_path = RECON_CACHE    / f"vi_{vi:05d}.pt"
        pt_path  = PT_RECON_CACHE / f"vi_{vi:05d}.pt"
        if not mix_path.exists() or not pt_path.exists():
            continue
        try:
            mix_recon = torch.load(mix_path, map_location="cpu", weights_only=True)
            pt_recon  = torch.load(pt_path,  map_location="cpu", weights_only=True)
            delta = {}
            for l in SAE_LAYERS:
                if l in mix_recon and l in pt_recon:
                    delta[l] = (mix_recon[l].float() - pt_recon[l].float()).to(torch.bfloat16)
            torch.save(delta, delta_path)
            n_delta_done += 1
        except Exception as e:
            print(f"  [WARN] delta vi={vi} failed: {e}", flush=True)
        if n_delta_done % 1000 == 0 and n_delta_done > 0:
            print(f"  deltas computed: {n_delta_done}/{len(ready_vis)}", flush=True)

    print(f"[INFO] Deltas ready: {n_delta_done}/{len(ready_vis)}", flush=True)

    # Build final list of vis with delta files
    inject_vis = [vi for vi in ready_vis if (DELTA_CACHE / f"vi_{vi:05d}.pt").exists()]
    print(f"[INFO] Samples to inject: {len(inject_vis)}", flush=True)

    # ── Load pt-448 ──
    print(f"\n[INFO] Loading {PT_MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(PT_MODEL)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_model  = NNsight(model_raw)
    tokenizer  = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_raw.parameters()).dtype

    print("[INFO] Loading VSR dataset...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])

    # ── Baseline stats over inject_vis ──
    base_correct = sum(
        base_preds[str(vi)]["correct"] for vi in inject_vis if str(vi) in base_preds
    )
    base_n = sum(1 for vi in inject_vis if str(vi) in base_preds)
    base_acc = base_correct / max(base_n, 1) * 100
    print(f"[INFO] Baseline acc on {base_n} inject samples: {base_acc:.2f}%", flush=True)

    # Per-sample results file for detailed logging
    per_sample_path = OUT_DIR / "per_sample_results.json"
    per_sample = {}
    if per_sample_path.exists():
        with open(per_sample_path) as f:
            per_sample = json.load(f)

    # ── Alpha sweep ──
    alpha_results = {}
    results_path  = OUT_DIR / "oracle_results.json"

    # Load existing results to allow partial resume
    if results_path.exists():
        with open(results_path) as f:
            alpha_results = json.load(f)

    for alpha in ALPHAS:
        alpha_key = str(alpha)
        if alpha_key in alpha_results and alpha_results[alpha_key].get("n_inject", 0) > 0:
            r = alpha_results[alpha_key]
            print(
                f"[SKIP] alpha={alpha}: acc={r['acc']:.2f}% Δ={r['delta_acc']:+.2f}% "
                f"n={r['n_inject']}",
                flush=True,
            )
            continue

        print(f"\n[INJECT] alpha={alpha} over {len(inject_vis)} samples...", flush=True)
        correct = total = 0
        alpha_sample_log = {}

        for step_i, vi in enumerate(inject_vis):
            ex    = vsr_all[vi]
            img   = _load_image(ex)
            if img is None:
                continue
            label  = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))

            delta_path = DELTA_CACHE / f"vi_{vi:05d}.pt"
            try:
                delta = torch.load(delta_path, map_location="cpu", weights_only=True)
            except Exception:
                continue

            # Move delta vectors to GPU for injection
            delta_gpu = {
                l: delta[l].to(model_dtype).to(device)
                for l in SAE_LAYERS if l in delta
            }

            try:
                iids, attn, pv = process_vlm_inputs(
                    img, prompt, processor, model_raw, device=device
                )
                _, img_end = get_image_token_positions(iids)

                with nns_model.trace(input_ids=iids, attention_mask=attn, pixel_values=pv):
                    for l in SAE_LAYERS:
                        if l not in delta_gpu:
                            continue
                        lo     = nns_model.model.language_model.layers[l].output[0][0, img_end:]
                        d_vec  = delta_gpu[l]          # [2304]
                        # Broadcast delta over all text positions (constant add)
                        # Proxy trick: derive a ones-tensor through the graph to keep nnsight happy
                        d_col  = d_vec.unsqueeze(0)    # [1, 2304]
                        proxy  = (lo * 0.0)[:1, :] + d_col   # [1, 2304]  constant
                        # Expand proxy to [T_text, 2304] by tiling
                        ones   = torch.ones(
                            lo.shape[0], 1, dtype=lo.dtype, device=lo.device
                        )
                        lo    += alpha * (ones @ proxy)  # broadcasts: ones [T,1] @ proxy [1,2304]
                    logits_s = nns_model.output.logits.save()

                pred, margin = _predict_and_margin(logits_s[0, -1, :], yes_ids, no_ids)

                base_info = base_preds.get(str(vi), {})
                alpha_sample_log[str(vi)] = {
                    "pred": pred, "label": label, "correct": int(pred == label),
                    "margin": margin,
                    "base_pred": base_info.get("pred"),
                    "base_margin": base_info.get("margin"),
                }
                total   += 1
                correct += int(pred == label)

            except Exception as e:
                if total < 5:
                    print(f"  [WARN] vi={vi} alpha={alpha}: {e}", flush=True)

            # Clean up GPU tensors
            del delta_gpu

            if (step_i + 1) % 500 == 0:
                cur_acc = correct / max(total, 1) * 100
                print(
                    f"  [{step_i+1}/{len(inject_vis)}] acc={cur_acc:.2f}% "
                    f"({correct}/{total})",
                    flush=True,
                )

        inj_acc   = correct / max(total, 1) * 100
        delta_acc = inj_acc - base_acc
        alpha_results[alpha_key] = {
            "alpha":     alpha,
            "acc":       inj_acc,
            "delta_acc": delta_acc,
            "n":         base_n,
            "n_inject":  total,
        }
        per_sample[alpha_key] = alpha_sample_log

        print(
            f"[RESULT] alpha={alpha}: acc={inj_acc:.2f}% Δ={delta_acc:+.2f}% "
            f"({correct}/{total})",
            flush=True,
        )

        # Save after each alpha
        with open(results_path, "w") as f:
            json.dump(alpha_results, f, indent=2)
        with open(per_sample_path, "w") as f:
            json.dump(per_sample, f)

        torch.cuda.empty_cache()
        gc.collect()

    # ── Final summary ──
    print(f"\n{'='*80}")
    print("SAE Reconstruction Delta Injection Results")
    print(f"{'='*80}")
    print(f"{'Alpha':>8}  {'Acc':>7}  {'Δ Acc':>8}  {'N':>6}")
    print("-" * 40)
    print(f"{'base':>8}  {base_acc:>6.2f}%  {'--':>8}  {base_n:>6}")
    for alpha in ALPHAS:
        r = alpha_results.get(str(alpha), {})
        if r:
            print(
                f"{alpha:>8.2f}  {r['acc']:>6.2f}%  {r['delta_acc']:>+7.2f}%  "
                f"{r['n_inject']:>6}"
            )
    print(f"\nResults saved to: {results_path}", flush=True)


# ─────────────────────────── Entry point ───────────────────────────

def main():
    if PHASE == "1":
        phase1_extract_mix_reconstructions()
    elif PHASE == "2":
        phase2_extract_pt_reconstructions()
    elif PHASE == "3":
        phase3_inject_delta()
    else:
        print(f"[ERROR] Unknown PHASE={PHASE!r}. Use PHASE=1, 2, or 3.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
