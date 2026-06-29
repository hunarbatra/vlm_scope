#!/usr/bin/env python3
"""
OCR steering — global CAA (mix→pt), mirroring VSR Recipe D exactly.

Mirrors pt448_true_caa_v4_spatial_layer_wdec_mix_to_pt.py:
  1. Collect mix-448 hiddens on ALL 600 train samples (all categories pooled).
     pos = mix-448 correct, neg = mix-448 incorrect.
  2. Build ONE global CAA vector per layer: v[L] = mean(pos[L]) - mean(neg[L]).
  3. Unload mix-448. Load pt-448.
  4. For each of 5 targets (layer, feature, category):
       a. Baseline eval on that category's test split (pt-448, no injection).
       b. Steered eval: inject α·unit(v[L]) at backbone layers {17,19,20,21}
          + γ·unit(W_dec[lF,F]) at feature layer lF. Sweep α×γ.
       c. VQA-ctrl eval for specificity.
  5. Write results.json per target + global summary.

Usage:
    python3 -u caa_global_ocr.py --device cuda:7
    python3 -u caa_global_ocr.py --device cuda:7 --middle-only   # L13 baseline
"""
import os, sys, json, gc, warnings, argparse
from pathlib import Path

import torch
from PIL import Image as PILImage

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MIX_MODEL  = "google/paligemma2-3b-mix-448"
PT_MODEL   = "google/paligemma2-3b-pt-448"
SAE_CKPT_DIR  = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
VQA_CTRL_PATH = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr/vqa_clean_yesno/indices.json")
SPLIT_PATH    = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr/ocrbench_split_600_400.json")
OUT_BASE      = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr/caa_global_mix2pt")

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

OCR_BACKBONE_LAYERS = [17, 19, 20, 21]
MAX_NEW_TOKENS = 64
ALPHAS_DEFAULT = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
GAMMAS_DEFAULT = [0.0, 3.0, 5.0, 7.0]

# 5 steering targets — (layer, feature, category)
TARGETS = [
    {"layer": 19, "feature": 10089, "category": "Scene Text-centric VQA"},
    {"layer": 17, "feature": 13602, "category": "Non-Semantic Text Recognition"},
    {"layer": 21, "feature": 9577,  "category": "Digit String Recognition"},
    {"layer": 19, "feature": 14093, "category": "Irregular Text Recognition"},
    {"layer": 20, "feature": 10687, "category": "Key Information Extraction"},
]


def _ocr_correct(response, gt_list):
    if response is None: return False
    if isinstance(gt_list, str): gt_list = [gt_list]
    resp = response.strip().lower()
    if not resp: return False
    for gt in gt_list:
        gt_l = str(gt).strip().lower()
        if not gt_l: continue
        if gt_l in resp or resp in gt_l: return True
    return False


class MultiLayerInjector:
    def __init__(self, model, steer_vecs):
        self._model = model
        self._vecs = {L: v.view(1, -1) for L, v in steer_vecs.items()}
        self.img_end = 0
        self._handles = []

    def set_img_end(self, img_end):
        self.img_end = int(img_end)

    def _make_hook(self, sv):
        def _hook(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            if x.shape[1] > 1:
                start = min(self.img_end, x.shape[1])
                x[:, start:, :] = x[:, start:, :] + sv
            else:
                x.add_(sv)
            return (x,) + out[1:] if isinstance(out, tuple) else x
        return _hook

    def install(self):
        for L, sv in self._vecs.items():
            h = self._model.model.language_model.layers[L].register_forward_hook(
                self._make_hook(sv))
            self._handles.append(h)
        return self

    def remove(self):
        for h in self._handles:
            try: h.remove()
            except: pass
        self._handles.clear()


def collect_all_hiddens(model, processor, ocr, tokenizer, device,
                        all_train_indices, collect_layers, verbose_every=50):
    """Collect mix-448 hiddens on all train indices (all categories pooled).
    Returns (hidden_cache={si: {layer: tensor}}, results=[{si, correct, ...}]).
    """
    from utils import process_vlm_inputs, get_image_token_positions

    collected = {}
    c_handles = []
    for L in collect_layers:
        def _make_collect(li):
            def _hook(module, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                collected[li] = x.detach()
                return out
            return _hook
        h = model.model.language_model.layers[L].register_forward_hook(_make_collect(L))
        c_handles.append(h)

    hidden_cache = {}
    results = []
    try:
        for idx, si in enumerate(all_train_indices):
            sample = ocr[si]
            question = str(sample.get("question", "")).strip()
            img = sample.get("image")
            gt_list = sample.get("answer", [])
            if isinstance(gt_list, str): gt_list = [gt_list]
            if img is None or not question or not gt_list: continue
            try:
                img = img.convert("RGB") if isinstance(img, PILImage.Image) else None
                if img is None: continue
                prompt = f"answer en {question}"
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, model, device=device)
                _, img_end = get_image_token_positions(input_ids)

                with torch.inference_mode():
                    _ = model(input_ids=input_ids, attention_mask=attn_mask,
                              pixel_values=pixel_values, use_cache=False)
                hidden_cache[si] = {L: collected[L][0, -1, :].float().cpu().clone()
                                    for L in collect_layers if L in collected}

                with torch.inference_mode():
                    out = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                                         pixel_values=pixel_values,
                                         max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                         use_cache=True)
                resp = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
                ok = _ocr_correct(resp, gt_list)
            except Exception:
                resp, ok = "", False

            results.append({"si": si, "correct": bool(ok), "response": resp,
                            "question": question,
                            "question_type": str(sample.get("question_type", ""))})
            if verbose_every and (idx + 1) % verbose_every == 0:
                c = sum(1 for r in results if r["correct"])
                print(f"  {idx+1}/{len(all_train_indices)}  mix_acc={100*c/len(results):.1f}%",
                      flush=True)
    finally:
        for h in c_handles:
            try: h.remove()
            except: pass

    return hidden_cache, results


def build_caa_vectors(hidden_cache, train_results, collect_layers):
    """Build global CAA per layer from all train results."""
    pos_hs = {L: [] for L in collect_layers}
    neg_hs = {L: [] for L in collect_layers}
    for r in train_results:
        si = r["si"]
        if si not in hidden_cache: continue
        for L in collect_layers:
            if L in hidden_cache[si]:
                (pos_hs if r["correct"] else neg_hs)[L].append(hidden_cache[si][L])

    caa_vecs = {}
    for L in collect_layers:
        np_, nn = len(pos_hs[L]), len(neg_hs[L])
        if not pos_hs[L] or not neg_hs[L]:
            print(f"  [WARN] L{L}: pos={np_} neg={nn} — skip"); continue
        v = torch.stack(pos_hs[L]).mean(0) - torch.stack(neg_hs[L]).mean(0)
        caa_vecs[L] = v / v.norm().clamp(min=1e-8)
        print(f"  CAA L{L}: n_pos={np_} n_neg={nn} ||v||={v.norm():.3f}", flush=True)
    return caa_vecs


def run_pt_eval(model, processor, ocr, tokenizer, device,
                indices, injector=None, verbose_every=20):
    from utils import process_vlm_inputs, get_image_token_positions
    results = []
    for idx, si in enumerate(indices):
        sample = ocr[si]
        question = str(sample.get("question", "")).strip()
        img = sample.get("image")
        gt_list = sample.get("answer", [])
        if isinstance(gt_list, str): gt_list = [gt_list]
        if img is None or not question or not gt_list: continue
        try:
            img = img.convert("RGB") if isinstance(img, PILImage.Image) else None
            if img is None: continue
            prompt = f"answer en {question}"
            input_ids, attn_mask, pixel_values = process_vlm_inputs(
                img, prompt, processor, model, device=device)
            _, img_end = get_image_token_positions(input_ids)
            if injector: injector.set_img_end(img_end)
            with torch.inference_mode():
                out = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                                     pixel_values=pixel_values,
                                     max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     use_cache=True)
            resp = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
            ok = _ocr_correct(resp, gt_list)
        except Exception:
            resp, ok = "", False
        results.append({"si": si, "correct": bool(ok), "response": resp,
                        "question": question,
                        "question_type": str(sample.get("question_type", ""))})
        if verbose_every and (idx + 1) % verbose_every == 0:
            c = sum(1 for r in results if r["correct"])
            print(f"    {idx+1}/{len(indices)}  pt_acc={100*c/len(results):.1f}%", flush=True)
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    return {"acc": 100 * correct / max(total, 1), "correct": correct,
            "total": total, "results": results}


def run_vqa_ctrl(model, processor, tokenizer, device, vqa, ctrl_indices, injector=None):
    from utils import process_vlm_inputs, get_image_token_positions
    correct = total = 0
    for vqa_idx, label in ctrl_indices:
        sample = vqa[vqa_idx]
        question = str(sample.get("question", "")).strip()
        img = sample.get("image")
        if img is None or not question: continue
        try:
            img = img.convert("RGB") if isinstance(img, PILImage.Image) else None
            if img is None: continue
            prompt = f"answer en {question}"
            input_ids, attn_mask, pixel_values = process_vlm_inputs(
                img, prompt, processor, model, device=device)
            _, img_end = get_image_token_positions(input_ids)
            if injector: injector.set_img_end(img_end)
            with torch.inference_mode():
                out = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                                     pixel_values=pixel_values,
                                     max_new_tokens=8, do_sample=False, use_cache=True)
            resp = tokenizer.decode(out[0, input_ids.shape[1]:],
                                    skip_special_tokens=True).strip().lower()
            gt = "yes" if label == 1 else "no"
            ok = resp.startswith(gt) or gt in resp
        except Exception:
            ok = False
        correct += int(ok); total += 1
    return {"acc": 100 * correct / max(total, 1), "correct": correct, "total": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda:7")
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS_DEFAULT)
    ap.add_argument("--gammas", type=float, nargs="+", default=GAMMAS_DEFAULT)
    ap.add_argument("--middle-only", action="store_true",
                    help="L13 only — Rimsky middle baseline")
    args = ap.parse_args()

    backbone_layers = [13] if args.middle_only else OCR_BACKBONE_LAYERS
    condition = "mid" if args.middle_only else "ocr"
    layers_tag = "_".join(str(l) for l in backbone_layers)

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    device = args.device

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    # Load split
    split = json.load(open(SPLIT_PATH))

    # All train indices pooled across all categories
    all_train_indices = []
    for cat_indices in split["train"].values():
        all_train_indices.extend(cat_indices)
    all_train_indices = sorted(set(all_train_indices))
    print(f"[INFO] Global train pool: {len(all_train_indices)} samples across all categories",
          flush=True)

    print("[INFO] Loading OCR-Bench...", flush=True)
    ocr = load_dataset("echo840/OCRBench", split="test")

    # VQA ctrl
    vqa_ctrl, vqa = [], None
    if VQA_CTRL_PATH.exists():
        meta = json.load(open(VQA_CTRL_PATH))
        vqa_ctrl = [(d["vqa_index"], int(d["label"])) for d in meta]
        vqa = load_dataset("lmms-lab/VQAv2", split="validation")
        print(f"[INFO] VQA-ctrl: {len(vqa_ctrl)} samples", flush=True)

    # Load W_dec for each target feature
    w_units = {}
    for t in TARGETS:
        wdec_path = SAE_CKPT_DIR / f"text-only_layer_{t['layer']}.pt"
        if wdec_path.exists():
            ckpt = torch.load(wdec_path, map_location="cpu", weights_only=True)
            w = ckpt["W_dec"][t["feature"]].float()
            w_units[t["layer"]] = w / w.norm().clamp(min=1e-8)
            print(f"[INFO] W_dec[L{t['layer']}/F{t['feature']}] loaded", flush=True)
        else:
            print(f"[WARN] W_dec missing: {wdec_path}", flush=True)

    # All unique layers needed for CAA collection
    collect_layers = sorted(set(backbone_layers + [t["layer"] for t in TARGETS]))

    # ---------------------------------------------------------------
    # STEP 1: mix-448 — collect hiddens on ALL train samples
    # ---------------------------------------------------------------
    global_hidden_path = OUT_BASE / f"hidden_global_{condition}_layers_{layers_tag}.pt"
    global_train_results_path = OUT_BASE / f"train_results_global_{condition}.json"
    global_caa_path = OUT_BASE / f"caa_global_{condition}_layers_{layers_tag}.pt"

    if global_hidden_path.exists() and global_caa_path.exists():
        print(f"[CACHED] Loading global hidden + CAA...", flush=True)
        hidden_cache = torch.load(global_hidden_path, map_location="cpu", weights_only=False)
        caa_data = torch.load(global_caa_path, map_location="cpu", weights_only=False)
        caa_vecs = caa_data["vecs"]
        train_results = (json.load(open(global_train_results_path))
                         if global_train_results_path.exists() else [])
        print(f"[CACHED] {len(hidden_cache)} samples, layers={sorted(caa_vecs.keys())}",
              flush=True)
    else:
        print(f"\n[STEP 1] Loading mix-448...", flush=True)
        mix_proc = AutoProcessor.from_pretrained(MIX_MODEL)
        mix_model = PaliGemmaForConditionalGeneration.from_pretrained(
            MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()

        print(f"[STEP 1] Collecting hiddens on {len(all_train_indices)} samples "
              f"at layers {collect_layers}...", flush=True)
        hidden_cache, train_results = collect_all_hiddens(
            mix_model, mix_proc, ocr, mix_proc.tokenizer, device,
            all_train_indices, collect_layers, verbose_every=50)

        n_correct = sum(1 for r in train_results if r["correct"])
        print(f"[STEP 1] mix-448 global train acc: "
              f"{100*n_correct/max(len(train_results),1):.1f}%  "
              f"({n_correct}/{len(train_results)} correct)", flush=True)

        torch.save(hidden_cache, global_hidden_path)
        with open(global_train_results_path, "w") as f:
            json.dump(train_results, f)

        print(f"[STEP 1] Building global CAA vectors...", flush=True)
        caa_vecs = build_caa_vectors(hidden_cache, train_results, collect_layers)

        torch.save({"vecs": caa_vecs, "layers": collect_layers,
                    "n_train": len(train_results),
                    "n_pos": sum(1 for r in train_results if r["correct"]),
                    "n_neg": sum(1 for r in train_results if not r["correct"])},
                   global_caa_path)

        del mix_model, mix_proc
        gc.collect(); torch.cuda.empty_cache()
        print("[STEP 1] mix-448 unloaded.", flush=True)

    # ---------------------------------------------------------------
    # STEP 2: pt-448 — eval per target on test split
    # ---------------------------------------------------------------
    print(f"\n[STEP 2] Loading pt-448...", flush=True)
    pt_proc = AutoProcessor.from_pretrained(PT_MODEL)
    pt_model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = pt_proc.tokenizer
    dtype = next(pt_model.parameters()).dtype

    # VQA-ctrl baseline (once, shared across all targets)
    base_ctrl_acc = None
    vqa_ctrl_results_path = OUT_BASE / "base_ctrl_pt.json"
    if vqa_ctrl_results_path.exists():
        base_ctrl_acc = json.load(open(vqa_ctrl_results_path))["acc"]
        print(f"[BASE ctrl cached] {base_ctrl_acc:.2f}%", flush=True)
    elif vqa is not None and vqa_ctrl:
        print(f"[STEP 2] VQA-ctrl baseline...", flush=True)
        ctrl_res = run_vqa_ctrl(pt_model, pt_proc, tokenizer, device, vqa, vqa_ctrl)
        base_ctrl_acc = ctrl_res["acc"]
        with open(vqa_ctrl_results_path, "w") as f: json.dump(ctrl_res, f)
        print(f"[BASE ctrl] {base_ctrl_acc:.2f}%", flush=True)

    # All test indices pooled across all categories (400 total)
    all_test_indices = []
    for cat_indices in split["test"].values():
        all_test_indices.extend(cat_indices)
    all_test_indices = sorted(set(all_test_indices))
    # Map si → category for per-category breakdown
    si_to_cat = {}
    for cat, indices in split["test"].items():
        for si in indices:
            si_to_cat[si] = cat
    print(f"[INFO] Global test pool: {len(all_test_indices)} samples", flush=True)

    def _acc_from_results(results, cat_filter=None):
        """Compute accuracy from list of {si, correct} dicts, optionally filtered by category."""
        subset = [r for r in results if cat_filter is None or si_to_cat.get(r["si"]) == cat_filter]
        if not subset: return 0.0, 0, 0
        correct = sum(1 for r in subset if r["correct"])
        return 100 * correct / len(subset), correct, len(subset)

    # One shared results file for all targets under this condition
    results_path = OUT_BASE / f"results_{condition}.json"
    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # Baseline: run once on all 400 test samples
    base_key = f"{condition}_base_pt"
    if base_key in all_results:
        base_results = all_results[f"{condition}_base_pt_results"]
        print(f"[BASE cached] loaded {len(base_results)} results", flush=True)
    else:
        print(f"[STEP 2] Baseline pt-448 on all {len(all_test_indices)} test samples...",
              flush=True)
        base_res = run_pt_eval(pt_model, pt_proc, ocr, tokenizer, device,
                               all_test_indices, verbose_every=50)
        base_results = base_res["results"]
        all_results[base_key] = {k: v for k, v in base_res.items() if k != "results"}
        all_results[f"{condition}_base_pt_results"] = base_results
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
        print(f"[BASE] overall={base_res['acc']:.2f}%  ({base_res['correct']}/{base_res['total']})",
              flush=True)
        # Per-category baseline
        for cat in split["test"]:
            a, c, n = _acc_from_results(base_results, cat)
            print(f"  {cat}: {a:.1f}% ({c}/{n})", flush=True)

    # Sweep α × γ — inject same global vector, evaluate on all 400
    # Per-target: the W_dec boost differs, so sweep separately per target
    for t in TARGETS:
        layer, feature, category = t["layer"], t["feature"], t["category"]
        cat_short = category.replace(" ", "_").replace("-", "_")
        base_acc_cat, _, _ = _acc_from_results(base_results, category)
        print(f"\n{'='*70}", flush=True)
        print(f"TARGET: L{layer}/F{feature}  '{category}'  "
              f"base_cat={base_acc_cat:.2f}%  backbone={backbone_layers}", flush=True)

        w_norm = w_units.get(layer)

        print(f"[STEP 3] Sweeping α={args.alphas} × γ={args.gammas}", flush=True)
        for alpha in args.alphas:
            for gamma in args.gammas:
                key = f"{condition}_L{layer}F{feature}_a{alpha:g}_g{gamma:g}"
                if key in all_results and all_results[key].get("total", 0) > 0:
                    r = all_results[key]
                    print(f"  [SKIP {key}] Δcat={r.get('delta_cat',0):+.2f}%", flush=True)
                    continue

                steer_vecs = {}
                for L in backbone_layers:
                    if L in caa_vecs:
                        steer_vecs[L] = (caa_vecs[L] * alpha).to(dtype).to(device)
                if gamma > 0 and w_norm is not None:
                    boost = (gamma * w_norm).to(dtype).to(device)
                    if layer in steer_vecs:
                        steer_vecs[layer] = steer_vecs[layer] + boost
                    else:
                        steer_vecs[layer] = boost

                if not steer_vecs:
                    print(f"  [SKIP] no steer vectors"); continue

                injector = MultiLayerInjector(pt_model, steer_vecs).install()
                try:
                    res = run_pt_eval(pt_model, pt_proc, ocr, tokenizer, device,
                                      all_test_indices, injector=injector, verbose_every=100)
                    ctrl_res = None
                    if vqa is not None and vqa_ctrl:
                        ctrl_res = run_vqa_ctrl(pt_model, pt_proc, tokenizer, device,
                                                vqa, vqa_ctrl, injector=injector)
                finally:
                    injector.remove()

                # Overall and per-category accuracy
                acc_cat, c_cat, n_cat = _acc_from_results(res["results"], category)
                delta_cat = acc_cat - base_acc_cat
                delta_overall = res["acc"] - all_results[base_key]["acc"]

                entry = {
                    "acc_overall": res["acc"], "delta_overall": delta_overall,
                    "acc_cat": acc_cat, "delta_cat": delta_cat,
                    "correct_cat": c_cat, "total_cat": n_cat,
                    "correct": res["correct"], "total": res["total"],
                    "alpha": alpha, "gamma": gamma,
                    "backbone_layers": backbone_layers,
                    "results": res["results"],
                }
                if ctrl_res:
                    entry["acc_ctrl"] = ctrl_res["acc"]
                    entry["delta_ctrl"] = ctrl_res["acc"] - (base_ctrl_acc or ctrl_res["acc"])
                all_results[key] = entry
                with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

                ctrl_str = f"  ΔCtrl={entry.get('delta_ctrl',0):+.2f}%" if ctrl_res else ""
                print(f"  [{key}] cat={acc_cat:.2f}% Δcat={delta_cat:+.2f}%  "
                      f"overall={res['acc']:.2f}%{ctrl_str}", flush=True)

        # Per-target summary
        print(f"\n  {'Key':40}  {'Δcat':>7}  {'cat%':>6}  {'ΔCtrl':>7}", flush=True)
        rows = [(v.get("delta_cat",0), k, v) for k, v in all_results.items()
                if k.startswith(f"{condition}_L{layer}F{feature}_a") and "delta_cat" in v]
        rows.sort(key=lambda x: -x[0])
        for _, k, r in rows[:5]:
            ctrl = f"{r.get('delta_ctrl',0):+.2f}%" if "delta_ctrl" in r else "  n/a"
            print(f"  {k:40}  {r['delta_cat']:>+6.2f}%  {r['acc_cat']:>5.2f}%  {ctrl}",
                  flush=True)

    # Global summary
    print(f"\n{'='*70}", flush=True)
    print(f"GLOBAL SUMMARY — mix→pt, {condition.upper()}, backbone={backbone_layers}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Category':<40} {'Base':>6}  {'Best Δcat':>10}  {'Best key'}", flush=True)
    print("  " + "-"*80, flush=True)
    for t in TARGETS:
        layer, feature, category = t["layer"], t["feature"], t["category"]
        base_acc_cat, _, _ = _acc_from_results(
            all_results.get(f"{condition}_base_pt_results", []), category)
        best_d, best_k = -999, None
        for k, v in all_results.items():
            if k.startswith(f"{condition}_L{layer}F{feature}_a") and "delta_cat" in v:
                if v["delta_cat"] > best_d: best_d = v["delta_cat"]; best_k = k
        best_s = f"{best_d:+.2f}% ({best_k})" if best_k else "pending"
        print(f"  {category:<40} {base_acc_cat:>5.1f}%  {best_s}", flush=True)

    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
