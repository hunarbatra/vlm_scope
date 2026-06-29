#!/usr/bin/env python3
"""
Baseline CAA for OCR-Bench (mix-448).

Rimsky-style mean-difference steering:
  1. Run baseline OCR-Bench eval to split samples into correct / incorrect.
  2. Build CAA vector at middle layer (layer 12) as:
       v = mean(last-text-token residual | correct) - mean(... | incorrect)
  3. Inject α·unit(v) at the middle layer on every text-token position
     (prompt and generated) and re-evaluate on OCR-Bench.

Sweeps α ∈ {0.5, 1, 2, 3, 5}. Writes per-α acc / Δ to results.json.

Usage:
    CUDA_VISIBLE_DEVICES=X python3 -B caa_baseline_ocr.py
"""
import os, sys, json, gc, warnings, argparse
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image as PILImage

warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

MODEL = "google/paligemma2-3b-mix-448"
OUT_DIR = Path("/data1/vlm_scope_sae_mix448_textonly/analysis_ocr/caa_baseline_ocr")

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"] = "/data1/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MIDDLE_LAYER = 13  # matches original VSR caa_recipe_compare_mix_to_pt.py (Rimsky middle for 26-layer PG2)
ALPHAS = [0.5, 1.0, 2.0, 3.0, 5.0]
MAX_NEW_TOKENS = 64


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


class LayerInjector:
    """Adds `steer` to every text-token position at a given layer's output.

    Uses the same forward-hook strategy as ablation: on the prompt pass
    (seq_len > 1) we add only from img_end onward; on decode passes
    (seq_len == 1) we always add (it's a generated text token).
    """

    def __init__(self, model, layer_idx, steer_vec):
        self.model = model
        self.layer = layer_idx
        self.sv = steer_vec.view(1, -1)  # (1, d)
        self.img_end = 0
        self.handle = None

    def set_img_end(self, img_end):
        self.img_end = int(img_end)

    def _hook(self, module, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if x.shape[1] > 1:
            start = min(self.img_end, x.shape[1])
            x[:, start:, :] = x[:, start:, :] + self.sv
        else:
            x.add_(self.sv)
        return (x,) + out[1:] if isinstance(out, tuple) else x

    def install(self):
        layer = self.model.model.language_model.layers[self.layer]
        self.handle = layer.register_forward_hook(self._hook)
        return self

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def run_ocr_eval(model, processor, ocr, tokenizer, device, injector=None,
                 collect_hidden_layer=None, verbose_every=200):
    """Run OCR-Bench eval. If `collect_hidden_layer` is not None, also return
    a dict {si: last-text-token residual at that layer} captured during forward.
    `injector` is an optional LayerInjector already installed.
    """
    from utils import process_vlm_inputs, get_image_token_positions

    hidden_cache = {}
    if collect_hidden_layer is not None:
        assert injector is None, "collect + inject simultaneously not supported"
        collected = {"x": None}
        def _collect_hook(module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            collected["x"] = x.detach()
            return out
        layer = model.model.language_model.layers[collect_hidden_layer]
        collect_handle = layer.register_forward_hook(_collect_hook)
    else:
        collect_handle = None

    results = []
    try:
        for si in range(len(ocr)):
            sample = ocr[si]
            question = str(sample.get("question", "")).strip()
            img = sample.get("image")
            gt_list = sample.get("answer", [])
            if isinstance(gt_list, str): gt_list = [gt_list]
            if img is None or not question or not gt_list:
                continue
            try:
                img = img.convert("RGB") if isinstance(img, PILImage.Image) else None
                if img is None: continue
                prompt = f"answer en {question}"
                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    img, prompt, processor, model, device=device)
                _, img_end = get_image_token_positions(input_ids)
                if injector is not None:
                    injector.set_img_end(img_end)

                if collect_hidden_layer is not None:
                    with torch.inference_mode():
                        _ = model(input_ids=input_ids, attention_mask=attn_mask,
                                  pixel_values=pixel_values, use_cache=False)
                    hidden_cache[si] = collected["x"][0, -1, :].float().cpu().clone()

                with torch.inference_mode():
                    out = model.generate(
                        input_ids=input_ids, attention_mask=attn_mask,
                        pixel_values=pixel_values,
                        max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                        use_cache=True,
                    )
                gen_ids = out[0, input_ids.shape[1]:]
                resp = tokenizer.decode(gen_ids, skip_special_tokens=True)
                ok = _ocr_correct(resp, gt_list)
            except Exception as e:
                resp = ""
                ok = False
            cat = str(sample.get("question_type", "unknown"))
            results.append({"si": si, "cat": cat, "correct": bool(ok), "response": resp})

            if verbose_every and (si + 1) % verbose_every == 0:
                c = sum(1 for r in results if r["correct"])
                print(f"    {si+1}/{len(ocr)}  acc={100*c/len(results):.2f}%", flush=True)
    finally:
        if collect_handle is not None:
            collect_handle.remove()

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    by_cat = defaultdict(lambda: {"c": 0, "t": 0})
    for r in results:
        by_cat[r["cat"]]["t"] += 1
        if r["correct"]: by_cat[r["cat"]]["c"] += 1
    return {
        "acc": 100 * correct / max(total, 1),
        "correct": correct, "total": total,
        "per_cat": {k: {"acc": 100*v["c"]/max(v["t"],1), "c": v["c"], "t": v["t"]}
                    for k, v in by_cat.items()},
        "results": results,
    }, hidden_cache


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=MIDDLE_LAYER)
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    args = ap.parse_args()

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    print(f"[INFO] Loading {MODEL}...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=False)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = processor.tokenizer
    dtype = next(model.parameters()).dtype

    print("[INFO] Loading OCR-Bench...", flush=True)
    ocr = load_dataset("echo840/OCRBench", split="test")
    print(f"[INFO] OCR-Bench: {len(ocr)} samples", flush=True)

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    # --- 1) Baseline eval + hidden collection ---
    hidden_path = OUT_DIR / f"hidden_L{args.layer}.pt"
    if "base" in all_results and hidden_path.exists():
        base = all_results["base"]
        hidden = torch.load(hidden_path, map_location="cpu")
        print(f"[BASE cached] {base['acc']:.2f}% ({base['correct']}/{base['total']})", flush=True)
    else:
        print(f"[INFO] Running baseline eval + collecting layer {args.layer} hidden...", flush=True)
        base, hidden = run_ocr_eval(model, processor, ocr, tokenizer, device,
                                    injector=None, collect_hidden_layer=args.layer)
        all_results["base"] = {"acc": base["acc"], "correct": base["correct"],
                               "total": base["total"], "per_cat": base["per_cat"]}
        with open(OUT_DIR / "baseline_per_sample.json", "w") as f:
            json.dump(base["results"], f, indent=2)
        torch.save(hidden, hidden_path)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
        print(f"[BASE] {base['acc']:.2f}% ({base['correct']}/{base['total']})", flush=True)

    # --- 2) Build CAA vector (correct - incorrect) ---
    caa_path = OUT_DIR / f"caa_L{args.layer}.pt"
    if caa_path.exists():
        caa = torch.load(caa_path, map_location="cpu")
        v = caa["v"]
        print(f"[CAA cached] L{args.layer} n_pos={caa['n_pos']} n_neg={caa['n_neg']} ||v||={v.norm():.3f}", flush=True)
    else:
        base_results = json.load(open(OUT_DIR / "baseline_per_sample.json"))
        pos_hs, neg_hs = [], []
        for r in base_results:
            si = r["si"]
            if si not in hidden: continue
            (pos_hs if r["correct"] else neg_hs).append(hidden[si])
        if not pos_hs or not neg_hs:
            print(f"[ERROR] No positive or negative samples (pos={len(pos_hs)}, neg={len(neg_hs)})")
            return
        pos_mean = torch.stack(pos_hs).mean(0)
        neg_mean = torch.stack(neg_hs).mean(0)
        v = pos_mean - neg_mean
        torch.save({"v": v, "n_pos": len(pos_hs), "n_neg": len(neg_hs),
                    "layer": args.layer}, caa_path)
        print(f"[CAA] L{args.layer} n_pos={len(pos_hs)} n_neg={len(neg_hs)} ||v||={v.norm():.3f}", flush=True)

    v_unit = (v / v.norm().clamp(min=1e-8)).to(dtype).to(device)

    # --- 3) Alpha sweep ---
    base_acc = all_results["base"]["acc"]
    for alpha in args.alphas:
        key = f"alpha_{alpha:g}_L{args.layer}"
        if key in all_results and all_results[key].get("total", 0) > 0:
            r = all_results[key]
            print(f"  [SKIP α={alpha:g}] {r['acc']:.2f}% Δ={r['delta']:+.2f}%", flush=True)
            continue

        sv = (v_unit * alpha)
        injector = LayerInjector(model, args.layer, sv).install()
        try:
            res, _ = run_ocr_eval(model, processor, ocr, tokenizer, device,
                                   injector=injector, collect_hidden_layer=None)
        finally:
            injector.remove()

        delta = res["acc"] - base_acc
        all_results[key] = {"acc": res["acc"], "delta": delta, "alpha": alpha,
                            "layer": args.layer, "correct": res["correct"],
                            "total": res["total"], "per_cat": res["per_cat"]}
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
        print(f"  [α={alpha:g} L{args.layer}] {res['acc']:.2f}%  Δ={delta:+.2f}%  "
              f"({res['correct']}/{res['total']})", flush=True)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Baseline CAA  —  layer {args.layer}  —  base {base_acc:.2f}%")
    print(f"{'='*60}")
    for alpha in args.alphas:
        key = f"alpha_{alpha:g}_L{args.layer}"
        if key in all_results:
            r = all_results[key]
            print(f"  α={alpha:g}  {r['acc']:.2f}%  Δ={r['delta']:+.2f}%")
    print(f"\nResults: {results_path}", flush=True)


if __name__ == "__main__":
    main()
