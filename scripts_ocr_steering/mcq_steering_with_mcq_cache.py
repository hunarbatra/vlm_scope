#!/usr/bin/env python3
"""MCQ steering using the MCQ-specific paired cache.

The cache contrasts pos = forward(prompt + correct_letter) vs neg = forward(prompt + wrong_letter).
This direction should DIRECTLY shift the model toward picking the correct letter.

Eval: same MCQ format as before, lenient logit-comparison on A/B/C/D.
"""
import os, sys, json, gc, random, string, argparse, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

PT_MODEL     = "google/paligemma2-3b-pt-448"
SAE_ROOT     = Path("/data1/vlm_scope_sae_mix448_textonly")
SAE_CKPT_DIR = SAE_ROOT / "checkpoints"
MCQ_CACHE    = SAE_ROOT / "analysis_ocr/paired_cache_mcq"
SAE_ACTS_DIR = SAE_ROOT / "analysis_ocr/sae_acts_ocrprompt"
OUT_DIR      = SAE_ROOT / "analysis_ocr/mcq_with_mcq_cache"

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MIDDLE_LAYER = 13
BACKBONE_LAYERS = [17, 19, 20, 21]
ALPHAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
GAMMAS = [1.0, 3.0, 10.0, 30.0]


def _parse_gt(raw):
    if isinstance(raw, list):
        for x in raw:
            if x is not None and str(x).strip(): return str(x).strip()
        return ""
    return str(raw).strip() if raw is not None else ""


def make_mcq_prompt(question, choices):
    letters = ["A", "B", "C", "D"]
    body = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
    return f"ocr {question}\n{body}\nAnswer: ("


def compute_paired_caa(layer, indices=None):
    """v[L] = mean(pos[L]) - mean(neg[L]) over MCQ cache."""
    if indices is None: indices = range(1000)
    pos = neg = None; n = 0
    for si in indices:
        p = MCQ_CACHE / f"vi_{si:05d}.pt"
        if not p.exists(): continue
        try: d = torch.load(p, map_location="cpu", weights_only=False)
        except: continue
        if "pos" not in d or "neg" not in d: continue
        if layer not in d["pos"] or layer not in d["neg"]: continue
        hp = d["pos"][layer].float(); hn = d["neg"][layer].float()
        pos = hp.clone() if pos is None else pos + hp
        neg = hn.clone() if neg is None else neg + hn
        n += 1
    if n == 0 or pos is None: return None, 0
    return (pos - neg) / n, n


def _load_wdec(layer, fi):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["W_dec"][fi].float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-key", type=str, default=None,
                   help="If set: also include γ·W_dec[F] at lF. Format: L17_F13602")
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    # Tag this run by feature key (if any)
    suffix = args.feature_key if args.feature_key else "global"
    results_path = OUT_DIR / f"results_{suffix}.json"

    print(f"=== MCQ-cache steering [{suffix}] ===", flush=True)
    ds = load_dataset("echo840/OCRBench", split="test")

    # Load MCQ data (positions, choices) from the cache itself — already built
    mcq_data = {}
    for si in range(1000):
        p = MCQ_CACHE / f"vi_{si:05d}.pt"
        if not p.exists(): continue
        try: d = torch.load(p, map_location="cpu", weights_only=False)
        except: continue
        mcq_data[si] = {
            "q": d.get("q", ""), "choices": d.get("choices", []),
            "gt_pos": d.get("gt_pos", 0),
        }
    print(f"MCQ samples in cache: {len(mcq_data)}", flush=True)

    # Compute CAA from MCQ cache (using all samples)
    needed_layers = sorted(set(BACKBONE_LAYERS) | {MIDDLE_LAYER})
    if args.feature_key:
        feat_layer = int(args.feature_key.split("_F")[0][1:])
        feat_id = int(args.feature_key.split("_F")[1])
        needed_layers = sorted(set(needed_layers) | {feat_layer})
    caa_unit = {}
    for L in needed_layers:
        v, n = compute_paired_caa(L)
        if v is not None:
            caa_unit[L] = v / v.norm().clamp(min=1e-8)
            print(f"  L{L}: paired n={n} raw_norm={v.norm():.3f}", flush=True)

    proc = AutoProcessor.from_pretrained(PT_MODEL)
    mdl = PaliGemmaForConditionalGeneration.from_pretrained(
        PT_MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    tok = proc.tokenizer
    dtype = next(mdl.parameters()).dtype

    letter_tok_ids = {}
    for L in ["A", "B", "C", "D"]:
        ids = tok.encode(L, add_special_tokens=False)
        if ids: letter_tok_ids[L] = ids[0]

    img_end_r = [0]
    def make_hook(sv_):
        def f(m, inp, out):
            ie = img_end_r[0]
            h = out[0] if isinstance(out, tuple) else out
            h[0, ie:] = h[0, ie:] + sv_.unsqueeze(0)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return f

    def eval_mcq(inject_pairs):
        c = t = 0
        for si in mcq_data:
            m = mcq_data[si]
            ex = ds[si]; img = ex.get("image")
            if img is None or not m.get("choices"): continue
            try:
                img = img.convert("RGB")
                prompt = make_mcq_prompt(m["q"], m["choices"])
                iids, attn, pv = process_vlm_inputs(img, prompt, proc, mdl, device=device)
                _, img_end_r[0] = get_image_token_positions(iids)
                hooks = []
                for (l, sv) in inject_pairs:
                    hooks.append(mdl.model.language_model.layers[l].register_forward_hook(make_hook(sv)))
                with torch.no_grad():
                    out = mdl(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                for h in hooks:
                    try: h.remove()
                    except: pass
                logits = out.logits[0, -1, :].float()
                letter_logits = {L: logits[lid].item() for L, lid in letter_tok_ids.items()}
                pred_letter = max(letter_logits, key=letter_logits.get)
                pred_idx = ["A", "B", "C", "D"].index(pred_letter)
                t += 1; c += int(pred_idx == m["gt_pos"])
            except: continue
        return c, t

    all_results = json.load(open(results_path)) if results_path.exists() else {}

    bk = "mcq_cache_base"
    if bk not in all_results:
        c, t = eval_mcq([])
        all_results[bk] = {"acc": c/max(t,1)*100, "n": t}
        print(f"\n  Baseline: {all_results[bk]['acc']:.2f}% (n={t})", flush=True)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)
    base = all_results[bk]["acc"]

    # A_MIDDLE
    for a in ALPHAS:
        rk = f"A_a{a}"
        if rk in all_results: continue
        sv = (caa_unit[MIDDLE_LAYER] * a).to(dtype).to(device)
        c, t = eval_mcq([(MIDDLE_LAYER, sv)])
        if t == 0: continue
        acc = c/t*100; delta = acc - base
        all_results[rk] = {"acc": acc, "delta": delta, "n": t}
        print(f"  [A α={a}] {acc:.2f}% Δ={delta:+.2f}% ({c}/{t})", flush=True)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    # C_BACKBONE (pure backbone CAA)
    for a in ALPHAS:
        rk = f"C_a{a}"
        if rk in all_results: continue
        inject = [(L, (caa_unit[L]*a).to(dtype).to(device)) for L in BACKBONE_LAYERS if L in caa_unit]
        c, t = eval_mcq(inject)
        if t == 0: continue
        acc = c/t*100; delta = acc - base
        all_results[rk] = {"acc": acc, "delta": delta, "n": t}
        print(f"  [C_BACKBONE α={a}] {acc:.2f}% Δ={delta:+.2f}% ({c}/{t})", flush=True)
        with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    # D_BB+WDEC if feature key provided
    if args.feature_key:
        feat_layer = int(args.feature_key.split("_F")[0][1:])
        feat_id = int(args.feature_key.split("_F")[1])
        w_dec = _load_wdec(feat_layer, feat_id)
        for a in ALPHAS:
            for g in GAMMAS:
                rk = f"D_a{a}_g{g}"
                if rk in all_results: continue
                inject = []
                for L in BACKBONE_LAYERS:
                    if L not in caa_unit: continue
                    if L == feat_layer:
                        sv = (caa_unit[L]*a + w_dec*g).to(dtype).to(device)
                    else:
                        sv = (caa_unit[L]*a).to(dtype).to(device)
                    inject.append((L, sv))
                c, t = eval_mcq(inject)
                if t == 0: continue
                acc = c/t*100; delta = acc - base
                all_results[rk] = {"acc": acc, "delta": delta, "n": t}
                print(f"  [D α={a} γ={g}] {acc:.2f}% Δ={delta:+.2f}% ({c}/{t})", flush=True)
                with open(results_path, "w") as f: json.dump(all_results, f, indent=2)

    print(f"\n[DONE] base={base:.2f}%", flush=True)


if __name__ == "__main__":
    main()
