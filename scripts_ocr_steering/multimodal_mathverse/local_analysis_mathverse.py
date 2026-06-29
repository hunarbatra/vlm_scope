"""
MathVerse 8-GPU analysis pipeline — mirrors local_analysis_textonly_ocr.py exactly,
with MathVerse (hunarbatra/MathVerse_Vision_MCQ, testmini, 430 samples) substituted
for OCR-Bench in the domain pass.

Full pipeline:
  Step 1: FVU table              (reuse analysis_ocr/fvu_table.csv — same checkpoints)
  Step 2: Cosine similarity      (reuse analysis_ocr/cosines/ — same)
  Step 3: Visual energy Ev       (reuse analysis_ocr/energy/ — same)
  Step 4: Select adapted features (reuse analysis_ocr/adapted/ — same)
  Step 5: Firing — VQA 50K (reuse analysis_ocr/firing_vqa/) + MathVerse 430 (new)
  Step 6: Fisher test VQA vs MathVerse → math-specific features
  Step 7: Lexical filtering (generic prompts, math-domain generic)
  Step 8: Intersection adapted ∩ math ∩ lexical

Then: ablation → top-5 → CAA steering.

Usage:
    cd /data1/vlm_scope_sae_mix448_textonly/scripts/multimodal_mathverse
    python3 local_analysis_mathverse.py --step 5 --gpus 8   # MathVerse firing only
    python3 local_analysis_mathverse.py --step 5 6 7 8      # full pipeline
    python3 local_analysis_mathverse.py                     # all steps 1-8
"""

import os, sys, gc, re, ast, csv, json, math, argparse, time
from pathlib import Path
from collections import defaultdict

import torch
import torch.multiprocessing as mp
import numpy as np

# ======================== Configuration ========================

MODEL_NAME   = "google/paligemma2-3b-mix-448"
TEXT_MODEL   = "google/gemma-2-2b"
N_LAYERS     = 26
D_SAE        = 16384
N_GPUS       = 8

ROOT         = Path("/data1/vlm_scope_sae_mix448_textonly")
CKPT_DIR     = ROOT / "checkpoints"
ANALYSIS_DIR = ROOT / "analysis_mathverse"   # mathverse-specific outputs
OCR_DIR      = ROOT / "analysis_ocr"         # reuse steps 1-4 + VQA firing from here
HF_CACHE     = "/data1/hf_cache/hub"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

N_FIRING_SAMPLES = 50_000  # VQA baseline

ODDS_THR      = 3.0
MIN_FREQ_DIFF = 0.05
EPSILON       = 0.01
COSINE_PCT    = 25.0

# Math-domain keywords for lexical filtering (generic math prompts avoid domain cues)
GENERIC_PROMPTS = [
    "Describe this image.",
    "What do you see in this picture?",
    "Summarize the contents of the image.",
    "Describe the objects and scene in this image.",
    "What is shown in the figure?",
]

# Math-domain keyword list for step 6 (contrast: VQA vs MathVerse)
MATH_KEYWORDS = [
    # Geometry
    "angle", "triangle", "circle", "rectangle", "square", "polygon", "parallel",
    "perpendicular", "diameter", "radius", "circumference", "area", "perimeter",
    "volume", "surface area", "congruent", "similar", "tangent", "chord", "arc",
    "vertex", "vertices", "edge", "face", "diagonal", "hypotenuse", "bisect",
    "intersect", "midpoint", "segment", "line", "ray", "plane", "coordinate",
    # Algebra / numbers
    "equation", "variable", "function", "graph", "slope", "intercept", "axis",
    "linear", "quadratic", "polynomial", "exponent", "logarithm", "inequality",
    "solution", "root", "factor", "integer", "fraction", "decimal", "ratio",
    "proportion", "percent", "probability", "statistics", "mean", "median", "mode",
    # Calculus / higher
    "derivative", "integral", "limit", "matrix", "vector", "theorem", "proof",
    # Figures and diagrams
    "figure", "diagram", "chart", "plot", "curve", "pattern", "sequence",
    "grid", "number line", "table", "formula", "expression", "calculate", "compute",
    "solve", "find", "given", "prove", "show that", "degree", "radian",
]


def _compile_math_regex():
    parts = [rf"\b{re.escape(kw)}\b" for kw in MATH_KEYWORDS]
    return re.compile("|".join(parts), flags=re.IGNORECASE)


# ======================== Step 1-4: reuse from OCR pipeline ========================

def step1_fvu_table():
    src = OCR_DIR / "fvu_table.csv"
    dst = ANALYSIS_DIR / "fvu_table.csv"
    if src.exists():
        import shutil; shutil.copy(src, dst)
        print(f"[Step 1] Copied from {src}")
    else:
        print("[Step 1] fvu_table.csv not found in OCR analysis dir — skipping (optional)")


def step2_cosine():
    src_dir = OCR_DIR / "cosines"
    if src_dir.exists() and len(list(src_dir.glob("cosines_layer_*.npy"))) == N_LAYERS:
        print(f"[Step 2] Cosines already in {src_dir} — reusing (same checkpoints)")
    else:
        print("[Step 2] Cosines missing — run local_analysis_textonly_ocr.py --step 2 first")


def step3_energy():
    src_dir = OCR_DIR / "energy"
    if src_dir.exists() and len(list(src_dir.glob("Ev_layer_*.npy"))) == N_LAYERS:
        print(f"[Step 3] Visual energy already in {src_dir} — reusing")
    else:
        print("[Step 3] Energy missing — run local_analysis_textonly_ocr.py --step 3 first")


def step4_adapted():
    src = OCR_DIR / "adapted" / "adapted_features_results.csv"
    if src.exists():
        print(f"[Step 4] Adapted features already in {src} — reusing")
    else:
        print("[Step 4] Adapted features missing — run local_analysis_textonly_ocr.py --step 4 first")


# ======================== Step 5: MathVerse firing ========================

def _firing_worker_mathverse(gpu_id, layer_indices):
    """Sample-level firing on MathVerse testmini (430 samples).
    VQA firing is reused from analysis_ocr/firing_vqa/."""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    from datasets import load_dataset

    math_dir = ANALYSIS_DIR / "firing_math"
    math_dir.mkdir(parents=True, exist_ok=True)

    remaining = [l for l in layer_indices
                 if not (math_dir / f"firing_math_layer_{l}.json").exists()]
    if not remaining:
        print(f"[Firing GPU{gpu_id}] All layers done, skipping"); return

    print(f"[Firing GPU{gpu_id}] Loading mix-448...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()
    nns_model = NNsight(model_raw)

    print(f"[Firing GPU{gpu_id}] Loading MathVerse testmini...")
    ds = load_dataset("hunarbatra/MathVerse_Vision_MCQ", split="testmini")
    N = len(ds)
    print(f"[Firing GPU{gpu_id}] {N} samples, layers: {remaining}", flush=True)

    for layer_idx in remaining:
        ckpt = CKPT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae  = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                        device=device, cache_dir=HF_CACHE)
        sae.eval()

        sample_fire_count = np.zeros(D_SAE, dtype=np.int64)
        n_samples = 0; n_failed = 0

        for si in range(N):
            ex = ds[si]
            img = ex.get("image")
            if img is None: n_failed += 1; continue
            try:
                img = img.convert("RGB")
                prompt = f"answer en {ex['prompt']}"
                iids, attn, pv = process_vlm_inputs(img, prompt, processor, model_raw, device=device)
                img_s, img_e = get_image_token_positions(iids)

                with torch.no_grad():
                    with nns_model.trace(input_ids=iids, attention_mask=attn,
                                         pixel_values=pv, use_cache=False) as tr:
                        layer_out = nns_model.model.language_model.layers[layer_idx].output[0].save()

                act   = layer_out.detach().squeeze(0).float()
                codes = sae.encode(act).detach()

                seq_len  = codes.shape[0]
                txt_mask = torch.ones(seq_len, dtype=torch.bool, device=codes.device)
                if img_e > img_s: txt_mask[img_s:img_e] = False

                txt_codes = codes[txt_mask]
                if txt_codes.shape[0] > 0:
                    fired = (txt_codes > 0).any(dim=0).cpu().numpy()
                    sample_fire_count += fired.astype(np.int64)
                n_samples += 1

            except Exception as e:
                n_failed += 1
                if n_failed <= 5:
                    print(f"[Firing GPU{gpu_id}] err L{layer_idx} si={si}: {e}", flush=True)
                continue

            if (si + 1) % 100 == 0 and gpu_id == 0:
                print(f"  L{layer_idx} Math: {si+1}/{N} ({n_failed} failed)", flush=True)

        out = {"layer": int(layer_idx), "n_samples": int(n_samples), "n_failed": int(n_failed),
               "dataset": "mathverse", "fire_count_all": sample_fire_count.tolist()}
        with open(math_dir / f"firing_math_layer_{layer_idx}.json", "w") as f:
            json.dump(out, f)

        print(f"[Firing GPU{gpu_id}] Math L{layer_idx}: {n_samples} samples "
              f"({n_failed} failed), features >50%: "
              f"{(sample_fire_count > n_samples * 0.5).sum()}", flush=True)

        del sae; torch.cuda.empty_cache(); gc.collect()

    del nns_model, model_raw, processor
    torch.cuda.empty_cache(); gc.collect()


def step5_firing(n_gpus=N_GPUS):
    print("\n" + "=" * 60)
    print(f"[Step 5] MathVerse firing (sample-level, {N_GPUS} GPUs)")
    print("=" * 60)

    # VQA baseline is reused from OCR pipeline
    vqa_dir = OCR_DIR / "firing_vqa"
    n_vqa = len(list(vqa_dir.glob("firing_vqa_layer_*.json"))) if vqa_dir.exists() else 0
    print(f"  VQA baseline: {n_vqa}/26 layers (reusing from {vqa_dir})")
    if n_vqa < N_LAYERS:
        print("  [WARN] VQA baseline incomplete — run OCR pipeline step 5 first")
        return

    layers_per_worker = math.ceil(N_LAYERS / n_gpus)
    assignments = []
    for w in range(n_gpus):
        start = w * layers_per_worker
        end   = min(start + layers_per_worker, N_LAYERS)
        if end > start: assignments.append((w, list(range(start, end))))

    for g, layers in assignments:
        print(f"  GPU {g}: layers {layers}")

    try: mp.set_start_method("spawn", force=True)
    except RuntimeError: pass

    procs = []
    for gpu_id, layers in assignments:
        p = mp.Process(target=_firing_worker_mathverse, args=(gpu_id, layers))
        p.start(); procs.append(p)
    for p in procs: p.join()

    failed = [i for i, p in enumerate(procs) if p.exitcode != 0]
    if failed: print(f"[ERROR] Firing workers {failed} failed!")


# ======================== Step 6: Fisher — VQA vs MathVerse ========================

def step6_math_features():
    print("\n" + "=" * 60)
    print("[Step 6] Math Features (Fisher Test — VQA vs MathVerse, sample-level)")
    print("=" * 60)

    from scipy.stats import fisher_exact
    from statsmodels.stats.multitest import multipletests
    import pandas as pd

    vqa_dir  = OCR_DIR / "firing_vqa"
    math_dir = ANALYSIS_DIR / "firing_math"
    out_dir  = ANALYSIS_DIR / "math_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer_idx in range(N_LAYERS):
        vqa_path  = vqa_dir  / f"firing_vqa_layer_{layer_idx}.json"
        math_path = math_dir / f"firing_math_layer_{layer_idx}.json"
        if not vqa_path.exists() or not math_path.exists():
            print(f"  L{layer_idx}: SKIP (missing data)"); continue

        vqa_data  = json.load(open(vqa_path))
        math_data = json.load(open(math_path))
        n_vqa  = vqa_data["n_samples"]
        n_math = math_data["n_samples"]
        if n_vqa == 0 or n_math == 0: continue

        fire_vqa  = np.array(vqa_data["fire_count_all"])
        fire_math = np.array(math_data["fire_count_all"])
        print(f"  L{layer_idx}: n_vqa={n_vqa}, n_math={n_math}", flush=True)

        for fi in range(D_SAE):
            c_vqa  = int(fire_vqa[fi])
            c_math = int(fire_math[fi])
            if c_vqa == 0 and c_math == 0: continue
            rows.append({"layer": layer_idx, "feature": fi,
                         "c_vqa": c_vqa, "n_vqa": n_vqa,
                         "c_math": c_math, "n_math": n_math})

    if not rows:
        print("  No features with nonzero firing"); return {"total_math": 0}

    df = pd.DataFrame(rows)
    print(f"  {len(df)} features with nonzero firing across {df['layer'].nunique()} layers")

    pvals, odds = [], []
    for _, r in df.iterrows():
        c_math = min(int(r.c_math), int(r.n_math))
        c_vqa  = min(int(r.c_vqa),  int(r.n_vqa))
        table  = [[c_math, max(0, int(r.n_math) - c_math)],
                  [c_vqa,  max(0, int(r.n_vqa)  - c_vqa)]]
        try:
            o, p = fisher_exact(table, alternative="greater")
            odds.append(o if not math.isinf(o) else 1e9)
            pvals.append(p)
        except ValueError:
            odds.append(1.0); pvals.append(1.0)

    df["odds_ratio"] = odds
    df["p_raw"]      = pvals
    df["freq_math"]  = df.c_math / df.n_math
    df["freq_vqa"]   = df.c_vqa  / df.n_vqa
    df["freq_diff"]  = df.freq_math - df.freq_vqa
    df["p_adj"]      = multipletests(df.p_raw, method="fdr_bh")[1]

    keep  = (df.odds_ratio >= ODDS_THR) & (df.freq_diff >= MIN_FREQ_DIFF)
    math_feats = df.loc[keep].sort_values("odds_ratio", ascending=False).copy()

    print(f"  Math features: {len(math_feats)} (OR>={ODDS_THR}, freq_diff>={MIN_FREQ_DIFF})")
    for layer_idx in sorted(math_feats["layer"].unique()):
        n = (math_feats["layer"] == layer_idx).sum()
        print(f"    L{layer_idx}: {n} features")

    math_feats.to_csv(out_dir / "math_features.csv", index=False)
    df.to_csv(out_dir / "all_features_stats.csv", index=False)
    summary = {"total_math": len(math_feats), "odds_threshold": ODDS_THR,
               "min_freq_diff": MIN_FREQ_DIFF, "total_tested": len(df)}
    with open(out_dir / "math_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ======================== Step 7: Lexical Filtering ========================

def _lexical_worker_mathverse(gpu_id, feature_assignments):
    """Lexical filter: test math candidates with generic prompts on MathVerse images."""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    from datasets import load_dataset

    out_dir = ANALYSIS_DIR / "lexical"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not feature_assignments: return

    print(f"[Lexical GPU{gpu_id}] {len(feature_assignments)} features to test", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16).to(device).eval()
    nns_model = NNsight(model_raw)

    ds = load_dataset("hunarbatra/MathVerse_Vision_MCQ", split="testmini")
    print(f"[Lexical GPU{gpu_id}] {len(ds)} MathVerse samples", flush=True)

    layer_features = defaultdict(list)
    for (layer_idx, feat_idx) in feature_assignments:
        layer_features[layer_idx].append(feat_idx)

    results = []
    top_k   = 5
    scan_n  = min(300, len(ds))
    act_thr = 0.01

    for layer_idx in sorted(layer_features.keys()):
        features = layer_features[layer_idx]
        ckpt = CKPT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae  = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                        device=device, cache_dir=HF_CACHE)
        sae.eval()

        # Phase 1: scan MathVerse to find top-activating samples per feature
        feature_cols = {f: [] for f in features}
        print(f"[Lexical GPU{gpu_id}] L{layer_idx}: scanning {scan_n} samples "
              f"for {len(features)} features...", flush=True)

        for si in range(scan_n):
            try:
                ex = ds[si]; img = ex.get("image")
                if img is None: continue
                img = img.convert("RGB")
                iids, attn, pv = process_vlm_inputs(
                    img, f"answer en {ex['prompt']}", processor, model_raw, device=device)
                with torch.no_grad():
                    with nns_model.trace(input_ids=iids, attention_mask=attn,
                                          pixel_values=pv, use_cache=False) as tr:
                        layer_out = nns_model.model.language_model.layers[layer_idx].output[0].save()
                act   = layer_out.detach().squeeze(0).float()
                codes = sae.encode(act.to(device)).detach().cpu()
                for f in features:
                    max_act = float(codes[:, f].max().item())
                    if max_act > 0: feature_cols[f].append((max_act, si))
            except Exception: continue

        # Phase 2: test top-k with generic prompts
        for feat_idx in features:
            candidates = sorted(feature_cols[feat_idx], key=lambda x: -x[0])[:top_k]
            if not candidates:
                results.append({"layer": layer_idx, "feature": feat_idx,
                                 "passed": False, "n_tested": 0}); continue

            passed = True; n_tested = 0
            for (mag, math_idx) in candidates:
                try:
                    ex = ds[math_idx]; img = ex.get("image")
                    if img is None: continue
                    img = img.convert("RGB")
                    best = 0.0
                    for raw_prompt in GENERIC_PROMPTS:
                        iids, attn, pv = process_vlm_inputs(
                            img, f"answer en {raw_prompt}", processor, model_raw, device=device)
                        with torch.no_grad():
                            with nns_model.trace(input_ids=iids, attention_mask=attn,
                                                  pixel_values=pv, use_cache=False) as tr:
                                lo = nns_model.model.language_model.layers[layer_idx].output[0].save()
                        act   = lo.detach().squeeze(0).float()
                        codes = sae.encode(act.to(device)).detach().cpu()
                        best  = max(best, float(codes[:, feat_idx].max().item()))
                    n_tested += 1
                    if best <= act_thr: passed = False; break
                except Exception: continue

            results.append({"layer": layer_idx, "feature": feat_idx,
                             "passed": passed, "n_tested": n_tested})
            status = "PASS" if passed else "FAIL"
            print(f"[Lexical GPU{gpu_id}] L{layer_idx} F{feat_idx}: {status} "
                  f"({n_tested}/{len(candidates)} tested)", flush=True)

        del sae; torch.cuda.empty_cache()

    out_path = out_dir / f"lexical_results_w{gpu_id}.json"
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    passed_n = sum(1 for r in results if r["passed"])
    print(f"[Lexical GPU{gpu_id}] {passed_n}/{len(results)} passed", flush=True)


def step7_lexical(n_gpus=N_GPUS):
    print("\n" + "=" * 60)
    print("[Step 7] Lexical Artifact Filtering")
    print("=" * 60)

    import pandas as pd

    # Pre-filter to adapted ∩ math to cut lexical work by ~5x
    adapted_path = OCR_DIR / "adapted" / "adapted_features_results.csv"
    adapted_by_layer = {}
    if adapted_path.exists():
        with open(adapted_path) as f:
            for row in csv.DictReader(f):
                layer = int(row["layer"])
                adapted_by_layer[layer] = set(ast.literal_eval(row["adapted_indices"]))

    csv_path = ANALYSIS_DIR / "math_features" / "math_features.csv"
    if not csv_path.exists():
        print("  No math features found. Run step 6 first."); return

    df = pd.read_csv(csv_path)
    if adapted_by_layer:
        # Only test features that are also in adapted set (will be required for intersection anyway)
        mask = df.apply(lambda r: int(r["feature"]) in adapted_by_layer.get(int(r["layer"]), set()), axis=1)
        df_filtered = df[mask]
        print(f"  Math candidates: {len(df)} → pre-filtered to adapted∩math: {len(df_filtered)}")
        df = df_filtered
    candidates = [(int(r["layer"]), int(r["feature"])) for _, r in df.iterrows()]
    print(f"  {len(candidates)} candidates to lexical-test")
    if not candidates: return

    (ANALYSIS_DIR / "lexical").mkdir(parents=True, exist_ok=True)

    feats_per_worker = math.ceil(len(candidates) / n_gpus)
    assignments = []
    for w in range(n_gpus):
        start = w * feats_per_worker
        end   = min(start + feats_per_worker, len(candidates))
        if end > start: assignments.append((w, candidates[start:end]))

    try: mp.set_start_method("spawn", force=True)
    except RuntimeError: pass

    procs = []
    for gpu_id, feats in assignments:
        p = mp.Process(target=_lexical_worker_mathverse, args=(gpu_id, feats))
        p.start(); procs.append(p)
    for p in procs: p.join()


# ======================== Step 8: Intersection ========================

def step8_intersection():
    print("\n" + "=" * 60)
    print("[Step 8] Feature Intersection (adapted ∩ math ∩ lexical)")
    print("=" * 60)

    import pandas as pd

    out_dir = ANALYSIS_DIR / "final_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Adapted (from OCR pipeline — same checkpoints)
    adapted_path = OCR_DIR / "adapted" / "adapted_features_results.csv"
    adapted_by_layer = {}
    if adapted_path.exists():
        with open(adapted_path) as f:
            for row in csv.DictReader(f):
                layer = int(row["layer"])
                adapted_by_layer[layer] = set(ast.literal_eval(row["adapted_indices"]))
    print(f"  Adapted: {sum(len(v) for v in adapted_by_layer.values())} features")

    # Math-specific (step 6)
    math_path = ANALYSIS_DIR / "math_features" / "math_features.csv"
    math_by_layer = {}
    if math_path.exists():
        df = pd.read_csv(math_path)
        for layer in df["layer"].unique():
            math_by_layer[int(layer)] = set(df[df["layer"] == layer]["feature"].tolist())
    print(f"  Math-specific: {sum(len(v) for v in math_by_layer.values())} features")

    # Lexical-passed (step 7)
    lex_dir = ANALYSIS_DIR / "lexical"
    lexical_passed = {}
    if lex_dir.exists():
        for fpath in sorted(lex_dir.glob("lexical_results_w*.json")):
            for r in json.load(open(fpath)):
                if r["passed"]:
                    layer = r["layer"]
                    lexical_passed.setdefault(layer, set()).add(r["feature"])
    print(f"  Lexical passed: {sum(len(v) for v in lexical_passed.values())} features")

    # Intersection
    all_layers   = sorted(set(adapted_by_layer) | set(math_by_layer))
    final_feats  = []
    for layer in all_layers:
        adapted = adapted_by_layer.get(layer, set())
        math    = math_by_layer.get(layer, set())
        common  = adapted & math
        if lexical_passed:
            common &= lexical_passed.get(layer, set())
        for fi in sorted(common):
            final_feats.append({"layer": layer, "feature": fi})
        if common:
            print(f"  L{layer}: adapted={len(adapted)}, math={len(math)}, "
                  f"lex={len(lexical_passed.get(layer,set()))}, final={len(common)}")

    if final_feats:
        pd.DataFrame(final_feats).to_csv(out_dir / "final_math_features.csv", index=False)
        print(f"  Saved {len(final_feats)} final features → {out_dir}/final_math_features.csv")
    else:
        print("  [WARN] Zero features in intersection — check thresholds / intermediate outputs")

    summary = {"total_final": len(final_feats),
               "total_adapted": sum(len(v) for v in adapted_by_layer.values()),
               "total_math":    sum(len(v) for v in math_by_layer.values()),
               "total_lex":     sum(len(v) for v in lexical_passed.values())}
    with open(out_dir / "intersection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ======================== Monitor ========================

def monitor():
    """Print current pipeline status."""
    print("\n" + "=" * 60 + "\nMATHVERSE PIPELINE STATUS\n" + "=" * 60)

    def _count(d, pat): return len(list(d.glob(pat))) if d.exists() else 0

    vqa_done  = _count(OCR_DIR / "firing_vqa",  "firing_vqa_layer_*.json")
    math_done = _count(ANALYSIS_DIR / "firing_math", "firing_math_layer_*.json")
    lex_done  = _count(ANALYSIS_DIR / "lexical", "lexical_results_w*.json")
    math_csv  = ANALYSIS_DIR / "math_features" / "math_features.csv"
    final_csv = ANALYSIS_DIR / "final_features" / "final_math_features.csv"
    adapted   = OCR_DIR / "adapted" / "adapted_features_results.csv"

    print(f"  Steps 1-4 (adapted/energy/cosine): {'✓' if adapted.exists() else '…'}")
    print(f"  Step 5a VQA baseline firing:       {vqa_done}/26 layers")
    print(f"  Step 5b MathVerse firing:           {math_done}/26 layers")
    if math_csv.exists():
        import pandas as pd
        df = pd.read_csv(math_csv)
        print(f"  Step 6 Math features:              {len(df)} features across {df['layer'].nunique()} layers")
    else:
        print(f"  Step 6 Math features:              …")
    print(f"  Step 7 Lexical results:             {lex_done} worker files")
    if final_csv.exists():
        import pandas as pd
        df = pd.read_csv(final_csv)
        print(f"  Step 8 Final intersection:         {len(df)} features ✓")
    else:
        print(f"  Step 8 Final intersection:         …")
    print("=" * 60)


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, nargs="+", default=None)
    parser.add_argument("--gpus", type=int, default=N_GPUS)
    parser.add_argument("--monitor", action="store_true")
    args = parser.parse_args()

    if args.monitor:
        monitor(); return

    steps = args.step if args.step else [1, 2, 3, 4, 5, 6, 7, 8]
    t0 = time.time()

    if 1 in steps: step1_fvu_table()
    if 2 in steps: step2_cosine()
    if 3 in steps: step3_energy()
    if 4 in steps: step4_adapted()
    if 5 in steps: step5_firing(n_gpus=args.gpus)
    if 6 in steps: step6_math_features()
    if 7 in steps: step7_lexical(n_gpus=args.gpus)
    if 8 in steps: step8_intersection()

    print(f"\n{'='*60}\n[DONE] {(time.time()-t0)/3600:.1f}h  Results: {ANALYSIS_DIR}\n{'='*60}")


if __name__ == "__main__":
    main()
