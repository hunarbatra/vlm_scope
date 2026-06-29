#!/usr/bin/env python3
"""
Per-category OCR steering — Recipe D (mix→pt, multi-layer OCR backbone + W_dec boost).

Mirrors VSR Recipe D exactly:
  1. Load mix-448. Run on TRAIN split (60%) of category — collect hiddens at OCR
     backbone layers {17,19,20,21} + feature layer. Also record mix-448 correctness.
  2. Build per-layer CAA: v[L] = mean(correct hiddens[L]) - mean(incorrect hiddens[L]).
  3. Unload mix-448. Load pt-448.
  4. Baseline eval on TEST split (40%) of category using pt-448.
  5. Steered eval: inject α·unit(v[L]) at all backbone layers + γ·unit(W_dec[L_feat,F])
     at feature layer, sweep α×γ.
  6. Eval on VQA-ctrl for specificity check.

Usage:
    python3 -u caa_per_category_ocr.py \
        --layer 19 --feature 10089 --category "Scene Text-centric VQA" \
        --backbone-layers 17 19 20 21 \
        --device cuda:0 --alphas 1 2 3 5 7 --gammas 0 3 5 7
"""
import os, sys, json, gc, warnings, argparse
from pathlib import Path

import torch
from PIL import Image as PILImage

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MIX_MODEL = "google/paligemma2-3b-mix-448"
PT_MODEL  = "google/paligemma2-3b-pt-448"
SAE_CKPT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/checkpoints")
VQA_CTRL_PATH = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr/vqa_clean_yesno/indices.json")
SPLIT_PATH = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr/ocrbench_split_600_400.json")
OUT_BASE = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr/caa_per_category_mix2pt")

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

OCR_BACKBONE_LAYERS = [17, 19, 20, 21]
MAX_NEW_TOKENS = 64
ALPHAS_DEFAULT = [1.0, 2.0, 3.0, 5.0, 7.0]
GAMMAS_DEFAULT = [0.0, 3.0, 5.0, 7.0]


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


def collect_hiddens(model, processor, ocr, tokenizer, device,
                    indices, collect_layers, verbose_every=30):
    """Run mix-448 forward passes on `indices`, collect per-layer last-text-token hiddens.
    Returns (hidden_cache={si:{layer:tensor}}, results=[{si,correct,response}]).
    """
    from utils import process_vlm_inputs, get_image_token_positions

    hidden_cache = {}
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

    results = []
    try:
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

                # Collect hiddens (forward only, no generate)
                with torch.inference_mode():
                    _ = model(input_ids=input_ids, attention_mask=attn_mask,
                              pixel_values=pixel_values, use_cache=False)
                hidden_cache[si] = {L: collected[L][0, -1, :].float().cpu().clone()
                                    for L in collect_layers if L in collected}

                # Also generate to get correctness label
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
                print(f"    {idx+1}/{len(indices)}  mix_acc={100*c/len(results):.1f}%", flush=True)
    finally:
        for h in c_handles:
            try: h.remove()
            except: pass

    return hidden_cache, results


def run_pt_eval(model, processor, ocr, tokenizer, device,
                indices, injector=None, verbose_every=30):
    """Run pt-448 eval on `indices`, optionally with injector installed."""
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
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--feature", type=int, required=True)
    ap.add_argument("--category", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS_DEFAULT)
    ap.add_argument("--gammas", type=float, nargs="+", default=GAMMAS_DEFAULT)
    ap.add_argument("--backbone-layers", type=int, nargs="+", default=OCR_BACKBONE_LAYERS)
    ap.add_argument("--middle-only", action="store_true",
                    help="Use single middle layer (L13) only — baseline Rimsky condition")
    args = ap.parse_args()

    if args.middle_only:
        args.backbone_layers = [13]

    collect_layers = sorted(set(args.backbone_layers + [args.layer]))
    layers_tag = "_".join(str(l) for l in sorted(args.backbone_layers))
    condition = "mid" if args.middle_only else "ocr"

    cat_short = args.category.replace(" ", "_").replace("-", "_")
    out_dir = OUT_BASE / f"L{args.layer}_F{args.feature}_{cat_short}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    all_results = json.load(open(results_path)) if results_path.exists() else {}

    device = args.device
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    # Load 600/400 split
    if SPLIT_PATH.exists():
        split = json.load(open(SPLIT_PATH))
        train_indices = split["train"].get(args.category, [])
        test_indices  = split["test"].get(args.category, [])
    else:
        print("[WARN] No split file — using all samples (no train/test separation)")
        raise FileNotFoundError(f"Split file not found: {SPLIT_PATH}")

    print(f"[INFO] Category '{args.category}': {len(train_indices)} train / {len(test_indices)} test", flush=True)

    print("[INFO] Loading OCR-Bench...", flush=True)
    ocr = load_dataset("echo840/OCRBench", split="test")

    # VQA ctrl
    vqa_ctrl, vqa = [], None
    if VQA_CTRL_PATH.exists():
        meta = json.load(open(VQA_CTRL_PATH))
        vqa_ctrl = [(d["vqa_index"], int(d["label"])) for d in meta]
        vqa = load_dataset("lmms-lab/VQAv2", split="validation")
        print(f"[INFO] VQA-ctrl: {len(vqa_ctrl)} samples", flush=True)

    # W_dec for feature boost
    wdec_path = SAE_CKPT_DIR / f"text-only_layer_{args.layer}.pt"
    w_unit = None
    if wdec_path.exists():
        ckpt = torch.load(wdec_path, map_location="cpu", weights_only=True)
        w = ckpt["W_dec"][args.feature].float()
        w_norm = w / w.norm().clamp(min=1e-8)
        print(f"[INFO] W_dec[L{args.layer}/F{args.feature}] ||w||={w.norm():.3f}", flush=True)
    else:
        w_norm = None
        print(f"[WARN] W_dec checkpoint not found: {wdec_path}", flush=True)

    # ---------------------------------------------------------------
    # STEP 1: mix-448 — collect train hiddens + correctness labels
    # ---------------------------------------------------------------
    hidden_path = out_dir / f"hidden_{condition}_layers_{layers_tag}.pt"
    train_results_key = f"{condition}_train_results"
    caa_path = out_dir / f"caa_{condition}_layers_{layers_tag}.pt"

    if hidden_path.exists() and caa_path.exists():
        hidden = torch.load(hidden_path, map_location="cpu", weights_only=False)
        caa_data = torch.load(caa_path, map_location="cpu", weights_only=False)
        caa_vecs = caa_data["vecs"]
        print(f"[CACHED] hidden+CAA loaded, layers={sorted(caa_vecs.keys())}", flush=True)
    else:
        print(f"\n[STEP 1] Loading mix-448 for hidden collection...", flush=True)
        mix_proc = AutoProcessor.from_pretrained(MIX_MODEL)
        mix_model = PaliGemmaForConditionalGeneration.from_pretrained(
            MIX_MODEL, torch_dtype=torch.bfloat16).to(device).eval()

        print(f"[STEP 1] Collecting hiddens on {len(train_indices)} train samples "
              f"at layers {collect_layers}...", flush=True)
        hidden, train_res = collect_hiddens(mix_model, mix_proc, ocr,
                                            mix_proc.tokenizer, device,
                                            train_indices, collect_layers,
                                            verbose_every=30)
        all_results[train_results_key] = train_res
        torch.save(hidden, hidden_path)
        print(f"[STEP 1] mix-448 train acc: "
              f"{100*sum(r['correct'] for r in train_res)/max(len(train_res),1):.1f}%", flush=True)

        # Build CAA
        pos_hs = {L: [] for L in collect_layers}
        neg_hs = {L: [] for L in collect_layers}
        for r in train_res:
            si = r["si"]
            if si not in hidden: continue
            for L in collect_layers:
                if L in hidden[si]:
                    (pos_hs if r["correct"] else neg_hs)[L].append(hidden[si][L])

        caa_vecs = {}
        for L in collect_layers:
            if not pos_hs[L] or not neg_hs[L]:
                print(f"  [WARN] L{L}: pos={len(pos_hs[L])} neg={len(neg_hs[L])} — skip"); continue
            v = torch.stack(pos_hs[L]).mean(0) - torch.stack(neg_hs[L]).mean(0)
            caa_vecs[L] = v / v.norm().clamp(min=1e-8)
            print(f"  CAA L{L}: n_pos={len(pos_hs[L])} n_neg={len(neg_hs[L])} ||v||={v.norm():.3f}", flush=True)

        torch.save({"vecs": caa_vecs, "layers": collect_layers}, caa_path)

        # Unload mix-448
        del mix_model, mix_proc
        gc.collect()
        torch.cuda.empty_cache()
        print("[STEP 1] mix-448 unloaded.", flush=True)

    # ---------------------------------------------------------------
    # STEP 2: pt-448 — baseline + steering eval on TEST split
    # ---------------------------------------------------------------
    base_key = f"{condition}_base_pt"

    print(f"\n[STEP 2] Loading pt-448 for eval...", flush=True)
    pt_proc = AutoProcessor.from_pretrained(PT_MODEL)
    pt_model = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = pt_proc.tokenizer
    dtype = next(pt_model.parameters()).dtype

    if base_key in all_results:
        base_acc = all_results[base_key]["acc"]
        print(f"[BASE cached] pt-448 test acc={base_acc:.2f}%", flush=True)
    else:
        print(f"[STEP 2] Baseline pt-448 eval on {len(test_indices)} test samples...", flush=True)
        base_res = run_pt_eval(pt_model, pt_proc, ocr, tokenizer, device,
                               test_indices, verbose_every=30)
        base_acc = base_res["acc"]
        all_results[base_key] = {k: v for k, v in base_res.items() if k != "results"}
        all_results[f"{condition}_base_pt_results"] = base_res["results"]

        if vqa is not None and vqa_ctrl:
            base_ctrl = run_vqa_ctrl(pt_model, pt_proc, tokenizer, device, vqa, vqa_ctrl)
            all_results["base_ctrl_pt"] = base_ctrl
            print(f"[BASE ctrl] {base_ctrl['acc']:.2f}%", flush=True)

        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
        print(f"[BASE pt-448] {base_acc:.2f}%  ({base_res['correct']}/{base_res['total']})", flush=True)

    base_ctrl_acc = all_results.get("base_ctrl_pt", {}).get("acc", None)

    # ---------------------------------------------------------------
    # STEP 3: Sweep α × γ
    # ---------------------------------------------------------------
    print(f"\n[STEP 3] Sweeping α={args.alphas} × γ={args.gammas}", flush=True)
    print(f"  backbone={args.backbone_layers}  W_dec @ L{args.layer}/F{args.feature}", flush=True)

    for alpha in args.alphas:
        for gamma in args.gammas:
            key = f"{condition}_a{alpha:g}_g{gamma:g}"
            if key in all_results and all_results[key].get("total", 0) > 0:
                r = all_results[key]
                print(f"  [SKIP {key}] Δ={r['delta']:+.2f}%", flush=True)
                continue

            steer_vecs = {}
            for L in args.backbone_layers:
                if L in caa_vecs:
                    steer_vecs[L] = (caa_vecs[L] * alpha).to(dtype).to(device)
            if gamma > 0 and w_norm is not None:
                boost = (gamma * w_norm).to(dtype).to(device)
                if args.layer in steer_vecs:
                    steer_vecs[args.layer] = steer_vecs[args.layer] + boost
                else:
                    steer_vecs[args.layer] = boost

            if not steer_vecs:
                print(f"  [SKIP] no steer vectors"); continue

            injector = MultiLayerInjector(pt_model, steer_vecs).install()
            try:
                res = run_pt_eval(pt_model, pt_proc, ocr, tokenizer, device,
                                  test_indices, injector=injector)
                ctrl_res = None
                if vqa is not None and vqa_ctrl:
                    ctrl_res = run_vqa_ctrl(pt_model, pt_proc, tokenizer, device,
                                            vqa, vqa_ctrl, injector=injector)
            finally:
                injector.remove()

            delta = res["acc"] - base_acc
            entry = {"acc": res["acc"], "delta": delta, "correct": res["correct"],
                     "total": res["total"], "alpha": alpha, "gamma": gamma,
                     "backbone_layers": args.backbone_layers,
                     "results": res["results"]}
            if ctrl_res:
                entry["acc_ctrl"] = ctrl_res["acc"]
                entry["delta_ctrl"] = ctrl_res["acc"] - (base_ctrl_acc or ctrl_res["acc"])
            all_results[key] = entry
            with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

            ctrl_str = f"  ΔCtrl={entry.get('delta_ctrl',0):+.2f}%" if ctrl_res else ""
            print(f"  [{key}] pt={res['acc']:.2f}%  Δ={delta:+.2f}%{ctrl_str}", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"Recipe D (mix→pt)  L{args.layer}/F{args.feature}  '{args.category}'", flush=True)
    print(f"Backbone: {args.backbone_layers}  {'(middle only)' if args.middle_only else '(OCR layers)'}", flush=True)
    print(f"Baseline pt-448: {base_acc:.2f}%  (test n={len(test_indices)})", flush=True)
    print(f"{'='*70}", flush=True)
    rows = [(v["delta"], k, v) for k, v in all_results.items()
            if k.startswith(f"{condition}_a") and "delta" in v]
    rows.sort(key=lambda x: -x[0])
    print(f"  {'Key':25}  {'Δ':>7}  {'pt%':>6}  {'ΔCtrl':>7}", flush=True)
    for _, k, r in rows:
        ctrl = f"{r.get('delta_ctrl',0):+.2f}%" if "delta_ctrl" in r else "  n/a"
        print(f"  {k:25}  {r['delta']:>+6.2f}%  {r['acc']:>5.2f}%  {ctrl}", flush=True)
    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
