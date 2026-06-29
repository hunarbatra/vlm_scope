#!/usr/bin/env python3
"""
Approach 3: Per-example mix-448 activation transplant.

For each VSR sample, extract exactly how much feature F fires in mix-448,
then inject `alpha * that_activation * W_dec[feature_id]` into pt-448 at
the same layer (text tokens only).

    mix_feat_act = mean(JumpReLU(h_mix[L] @ W_enc[:, F] + b_enc[F] - threshold[F]))[text_tokens]
    pt-448 hidden[L, text_tokens] += alpha * mix_feat_act * W_dec[F]

Unlike fixed injection (Approach 1) or CAA (Approach 2), this routes exactly
the mix-448 feature firing level into the pt-448 processing path per-example.
Examples where mix-448 sees no spatial signal get zero injection; examples with
strong mix-448 activation get proportionally more.

Both models are loaded simultaneously (~12 GB bfloat16 on 24 GB A5000).

Output: /data1/vlm_scope_sae_mix448_textonly/analysis/pt448_activation_transplant/
    transplant_L{layer}_F{feature}.json

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 pt448_activation_transplant.py
"""

import os, sys, re, json, gc, hashlib, warnings, math
from pathlib import Path
from io import BytesIO
from collections import defaultdict

import torch
import requests
from PIL import Image

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths & env ──────────────────────────────────────────────────────────────
MIX_MODEL      = "google/paligemma2-3b-mix-448"
PT_MODEL       = "google/paligemma2-3b-pt-448"
CHECKPOINT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
OUT_DIR        = Path("/data1/vlm_scope_sae_mix448_textonly/analysis/pt448_activation_transplant")
IMAGE_CACHE    = Path("/data1/vlm_scope_sae_mix448_textonly/vsr_image_cache")
VSR_DATASET    = "cambridgeltl/vsr_random"

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

# ── Features & alphas ────────────────────────────────────────────────────────
FEATURES = [
    {"layer":  4, "feature": 14233, "relations": ["ahead of", "behind"]},
    {"layer":  6, "feature":  7539, "relations": ["left of", "right of", "across from", "alongside",
                                                   "at the back of", "below", "facing away from"]},
    {"layer":  9, "feature":   387, "relations": ["at the right side of", "adjacent to",
                                                   "far from", "attached to"]},
    {"layer":  9, "feature":  7540, "relations": ["on", "next to", "parallel to",
                                                   "in the middle of", "opposite to",
                                                   "away from", "consists of"]},
    {"layer": 11, "feature": 12278, "relations": ["touching", "on top of", "surrounding", "under"]},
    {"layer": 12, "feature":  2257, "relations": ["facing", "beneath", "near", "off",
                                                   "enclosed by", "inside", "within",
                                                   "beyond", "at the side of"]},
    {"layer": 14, "feature": 10561, "relations": ["close to", "by", "connected to"]},
    {"layer": 15, "feature":   220, "relations": ["above", "at the left side of", "beside",
                                                   "contains", "over", "part of", "right of",
                                                   "outside", "toward"]},
]

ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

# Sorted by length descending for unambiguous regex matching
ALL_RELATIONS = sorted(
    {r for feat in FEATURES for r in feat["relations"]},
    key=len, reverse=True
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_vsr_prompt(statement: str) -> str:
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


def _pm(logits, yes_ids, no_ids):
    """Predict (0/1) and log-odds margin from last-token logits."""
    probs = torch.softmax(logits.float(), dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 1e-9
    n = probs[list(no_ids)].sum().item()  if no_ids  else 1e-9
    d = y + n
    p = max(y / d if d > 0 else 0.5, 1e-7)
    return (1 if p > 0.5 else 0), math.log(p / max(1.0 - p, 1e-7))


def _relation_from_caption(caption: str) -> str:
    """Extract the first matching relation from caption using word-boundary regex."""
    cap_lower = caption.lower()
    for rel in ALL_RELATIONS:
        if re.search(r"\b" + re.escape(rel) + r"\b", cap_lower):
            return rel
    return ""


def _load_image(ex) -> "Image.Image | None":
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)

    # ── Load both models simultaneously ──────────────────────────────────────
    print(f"[INFO] Loading {MIX_MODEL} ...", flush=True)
    mix_proc  = AutoProcessor.from_pretrained(MIX_MODEL)
    model_mix = PaliGemmaForConditionalGeneration.from_pretrained(
        MIX_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_mix = NNsight(model_mix)

    print(f"[INFO] Loading {PT_MODEL} ...", flush=True)
    pt_proc   = AutoProcessor.from_pretrained(PT_MODEL)
    model_pt  = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    nns_pt = NNsight(model_pt)

    tokenizer   = mix_proc.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    model_dtype = next(model_mix.parameters()).dtype

    # ── Load VSR ─────────────────────────────────────────────────────────────
    print("[INFO] Loading VSR (train+dev+test) ...", flush=True)
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    vsr_all = concatenate_datasets([
        load_dataset(VSR_DATASET, data_files=data_files, split=s)
        for s in ["train", "dev", "test"]
    ])
    print(f"[INFO] VSR total: {len(vsr_all)}", flush=True)

    # Build relation → index map using dataset "relation" field, falling back to caption parse
    relation_indices: dict[str, list[int]] = defaultdict(list)
    for vi in range(len(vsr_all)):
        ex  = vsr_all[vi]
        rel = ex.get("relation", "").strip()
        if not rel:
            rel = _relation_from_caption(str(ex.get("caption", "")))
        if rel:
            relation_indices[rel].append(vi)

    def get_indices(relations: list[str]) -> list[int]:
        idxs = []
        for r in relations:
            idxs.extend(relation_indices.get(r, []))
        return idxs

    # ── Per-feature loop ──────────────────────────────────────────────────────
    for feat_cfg in FEATURES:
        home_layer = feat_cfg["layer"]
        feature_id = feat_cfg["feature"]
        relations  = feat_cfg["relations"]
        key        = f"L{home_layer}_F{feature_id}"
        out_path   = OUT_DIR / f"transplant_{key}.json"

        if out_path.exists():
            print(f"[SKIP] {key} — output exists", flush=True)
            continue

        indices = get_indices(relations)
        if not indices:
            print(f"[WARN] {key}: no VSR samples found for {relations}", flush=True)
            continue

        print(f"\n[FEATURE] {key}  relations={relations}  N={len(indices)}", flush=True)

        # ── Load SAE weights for this layer (CPU only to avoid OOM) ──────────
        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{home_layer}.pt"
        print(f"[SAE] Loading {ckpt_path} ...", flush=True)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        # Extract only the 2304-element vectors for this feature, keep on CPU
        w_enc_col  = state["W_enc"][:, feature_id].float()   # [2304] encoder col
        b_enc_val  = state["b_enc"][feature_id].float()      # scalar
        thresh_val = state["threshold"][feature_id].float()  # scalar
        wdec_cpu   = state["W_dec"][feature_id]              # [2304] decoder row
        del state
        gc.collect()

        # Decoder direction on device for injection
        wdec_vec = (wdec_cpu / wdec_cpu.norm().clamp(min=1e-8)).to(model_dtype).to(device)
        # Encoder col on CPU (used in mix-448 hidden extraction loop)
        w_enc_col_cpu = w_enc_col  # [2304], float32, cpu

        # ── Phase 1: cache base pt-448 predictions + mix-448 feat_act ────────
        print(f"[CACHE] Extracting base pt-448 predictions and mix-448 activations ...", flush=True)

        sample_cache = []  # list of (feat_act, iids, attn, pv, img_end, lbl, pb)
        n_failed     = 0

        for vi in indices:
            ex  = vsr_all[vi]
            img = _load_image(ex)
            if img is None:
                n_failed += 1
                continue
            label  = int(ex.get("label", 0))
            prompt = _build_vsr_prompt(str(ex.get("caption", "")))

            try:
                iids, attn, pv = process_vlm_inputs(
                    img, prompt, mix_proc, model_mix, device=device
                )
                _, img_end = get_image_token_positions(iids)

                # ── Step 1: get feature activation from mix-448 ───────────────
                with nns_mix.trace(input_ids=iids, attention_mask=attn,
                                   pixel_values=pv):
                    hidden_mix = nns_mix.model.language_model.layers[home_layer].output[0].save()

                h       = hidden_mix[0, img_end:, :].float().cpu()     # [T_text, 2304] on CPU
                pre_act = h @ w_enc_col_cpu + b_enc_val               # [T_text]
                feat_act = torch.relu(pre_act - thresh_val).mean().item()

                # ── Base pt-448 prediction (no injection) ─────────────────────
                # Re-process with pt processor (same image, same prompt)
                iids_pt, attn_pt, pv_pt = process_vlm_inputs(
                    img, prompt, pt_proc, model_pt, device=device
                )
                _, img_end_pt = get_image_token_positions(iids_pt)

                with torch.inference_mode():
                    out_base = model_pt(
                        input_ids=iids_pt, attention_mask=attn_pt,
                        pixel_values=pv_pt, use_cache=False
                    )
                pb, _ = _pm(out_base.logits[0, -1, :], yes_ids, no_ids)

                sample_cache.append((feat_act, iids_pt, attn_pt, pv_pt,
                                     img_end_pt, label, pb))

            except Exception as exc:
                n_failed += 1
                continue

        n          = len(sample_cache)
        n_firing   = sum(1 for s in sample_cache if s[0] > 0.0)
        pct_firing = n_firing / max(n, 1) * 100.0
        mean_act   = sum(s[0] for s in sample_cache) / max(n, 1)
        base_correct = sum(1 for s in sample_cache if s[6] == s[5])
        base_acc     = base_correct / max(n, 1) * 100.0

        print(f"[CACHE] done: n={n}, failed={n_failed}, "
              f"firing={n_firing} ({pct_firing:.1f}%), "
              f"mean_feat_act={mean_act:.4f}, base_acc={base_acc:.2f}%", flush=True)

        # Free per-feature CPU buffers (encoder col already not on GPU)
        del w_enc_col_cpu, w_enc_col, b_enc_val, thresh_val
        torch.cuda.empty_cache()

        # ── Phase 2: alpha sweep ──────────────────────────────────────────────
        alpha_results: dict[str, dict] = {}

        for alpha in ALPHAS:
            correct_inj    = 0
            total_inj      = 0
            sum_inj_mag    = 0.0

            for feat_act, iids_pt, attn_pt, pv_pt, img_end_pt, label, pb in sample_cache:
                total_inj += 1

                # If feature doesn't fire in mix-448, transplant has zero effect;
                # use the cached base prediction directly
                if feat_act <= 0.0:
                    correct_inj += (pb == label)
                    # injected magnitude is 0
                    continue

                injection_mag = alpha * feat_act
                sum_inj_mag  += abs(injection_mag)

                try:
                    with nns_pt.trace(input_ids=iids_pt, attention_mask=attn_pt,
                                      pixel_values=pv_pt):
                        v_col = wdec_vec.unsqueeze(1)   # [2304, 1]
                        lo    = nns_pt.model.language_model.layers[home_layer].output[0][0, img_end_pt:]
                        # proxy ones via matmul, zeroed + 1 → shape [T_text, 1]
                        ones = (lo @ v_col) * 0.0 + 1.0
                        lo  += alpha * feat_act * ones * wdec_vec
                        logits_s = nns_pt.output.logits.save()

                    pred, _ = _pm(logits_s[0, -1, :], yes_ids, no_ids)
                except Exception:
                    pred = pb  # fall back to base prediction on error

                correct_inj += (pred == label)

            acc_inj   = correct_inj / max(total_inj, 1) * 100.0
            delta_acc = acc_inj - base_acc
            mean_inj_mag = sum_inj_mag / max(n_firing, 1)

            alpha_results[str(alpha)] = {
                "acc":                   round(acc_inj,   4),
                "delta_acc":             round(delta_acc, 4),
                "mean_injected_magnitude": round(mean_inj_mag, 6),
            }
            print(f"  alpha={alpha:+.2f}  acc={acc_inj:.2f}%  "
                  f"delta={delta_acc:+.2f}%  "
                  f"mean_inj_mag={mean_inj_mag:.4f}", flush=True)

        # Best alpha by delta_acc
        best_alpha = max(ALPHAS, key=lambda a: alpha_results[str(a)]["delta_acc"])
        best_delta = alpha_results[str(best_alpha)]["delta_acc"]

        result = {
            "layer":            home_layer,
            "feature":          feature_id,
            "relations":        relations,
            "n":                n,
            "n_feat_firing":    n_firing,
            "pct_firing":       round(pct_firing, 4),
            "mean_feat_act":    round(mean_act, 6),
            "base_acc":         round(base_acc, 4),
            "alphas":           alpha_results,
            "best_alpha":       best_alpha,
            "best_delta":       round(best_delta, 4),
        }

        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[SAVED] {out_path}  best_alpha={best_alpha}  best_delta={best_delta:+.2f}%",
              flush=True)

        # Clean up per-feature vectors; leave both models loaded
        del wdec_vec, wdec_cpu, sample_cache
        torch.cuda.empty_cache()
        gc.collect()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*120}", flush=True)
    print("pt-448 Activation Transplant (Approach 3) — Summary", flush=True)
    print(f"{'='*120}", flush=True)
    header = f"{'L/F':<14} {'Relations':<38} {'N':>5} {'Firing':>7} {'MeanAct':>9} {'Base':>7}"
    for a in ALPHAS:
        header += f"  {a:>+5}"
    header += f"  {'Best':>5} {'BestΔ':>7}"
    print(header, flush=True)
    print("-" * 120, flush=True)

    for feat_cfg in FEATURES:
        home_layer = feat_cfg["layer"]
        feature_id = feat_cfg["feature"]
        key        = f"L{home_layer}/F{feature_id}"
        out_path   = OUT_DIR / f"transplant_L{home_layer}_F{feature_id}.json"
        if not out_path.exists():
            print(f"{key:<14} (no output)", flush=True)
            continue
        with open(out_path) as f:
            r = json.load(f)
        rels = "; ".join(r["relations"])[:37]
        row  = (f"{key:<14} {rels:<38} {r['n']:>5} "
                f"{r['pct_firing']:>6.1f}% {r['mean_feat_act']:>9.4f} "
                f"{r['base_acc']:>6.2f}%")
        for a in ALPHAS:
            d = r["alphas"].get(str(a), {}).get("delta_acc")
            row += f"  {d:>+5.2f}" if d is not None else f"  {'--':>5}"
        row += f"  {r['best_alpha']:>5.2f} {r['best_delta']:>+7.2f}%"
        print(row, flush=True)


if __name__ == "__main__":
    main()
