#!/usr/bin/env python3
"""Find R(F) samples where the winning steering config flipped wrong→right.
Saves images + details for a qualitative doc.
"""
import os, sys, json, random, string, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")

PT_MODEL     = "google/paligemma2-3b-pt-448"
SAE_ROOT     = Path("/data1/vlm_scope_sae_mix448_textonly")
SAE_CKPT_DIR = SAE_ROOT / "checkpoints"
PAIR_CACHE   = SAE_ROOT / "analysis_ocr/paired_cache_ocrprompt"
SAE_ACTS_DIR = SAE_ROOT / "analysis_ocr/sae_acts"
DOC_DIR      = Path("/home/hbatra/vision-language-scope/docs/ocr_steering_examples")

os.environ["HF_DATASETS_CACHE"] = "/data1/hf_cache/datasets"
os.environ["HF_HOME"]           = "/data1/hf_cache"
os.environ["HF_TOKEN"]          = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")

MIDDLE_LAYER = 13
BACKBONE_LAYERS = [17, 19, 20, 21]

# Best config per feature (from results)
CONFIGS = [
    {"layer": 21, "feature": 9577,  "key": "L21_F9577",  "alpha": 50.0, "gamma": 100.0,
     "category": "Digit String", "delta": 5.81},
    {"layer": 17, "feature": 13602, "key": "L17_F13602", "alpha": 100.0, "gamma": 0.5,
     "category": "Scene Text-centric VQA", "delta": 3.85},
    {"layer": 19, "feature": 14093, "key": "L19_F14093", "alpha": 20.0, "gamma": 10.0,
     "category": "Irregular Text", "delta": 1.68},
]


def _parse_gt(raw):
    if isinstance(raw, list):
        for x in raw:
            if x is not None and str(x).strip(): return str(x).strip()
        return ""
    return str(raw).strip() if raw is not None else ""


def distort_answer(gt, rng, n_subs=None):
    s = list(str(gt).strip())
    if len(s) == 0: return "X"
    if n_subs is None: n_subs = max(1, min(3, len(s) // 3))
    n_subs = min(n_subs, len(s))
    idxs = rng.sample(range(len(s)), n_subs)
    for idx in idxs:
        c = s[idx]
        if c.isdigit():
            pool = [d for d in "0123456789" if d != c]
        elif c.isalpha():
            pool_lower = [d for d in string.ascii_lowercase if d != c.lower()]
            new = rng.choice(pool_lower)
            s[idx] = new.upper() if c.isupper() else new
            continue
        else: continue
        s[idx] = rng.choice(pool)
    out = "".join(s)
    if out == str(gt).strip(): out = out + "x"
    return out


def make_mcq_prompt(question, choices):
    letters = ["A", "B", "C", "D"]
    body = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
    return f"ocr {question}\n{body}\nAnswer: ("


def compute_paired_caa(layer, indices):
    pos = neg = None; n = 0
    for si in indices:
        p = PAIR_CACHE / f"vi_{si:05d}.pt"
        if not p.exists(): continue
        try: d = torch.load(p, map_location="cpu", weights_only=False)
        except: continue
        if "pos" not in d or "neg" not in d: continue
        if layer not in d["pos"] or layer not in d["neg"]: continue
        hp = d["pos"][layer].float(); hn = d["neg"][layer].float()
        pos = hp.clone() if pos is None else pos + hp
        neg = hn.clone() if neg is None else neg + hn
        n += 1
    if n == 0 or pos is None: return None
    return (pos - neg) / n


def _load_wdec(layer, fi):
    p = SAE_CKPT_DIR / f"text-only_layer_{layer}.pt"
    d = torch.load(p, map_location="cpu", weights_only=True)
    return d["W_dec"][fi].float()


def main():
    from datasets import load_dataset
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    sys.path.insert(0, str(Path(__file__).parent))
    from utils import process_vlm_inputs, get_image_token_positions

    device = "cuda:0"
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "images").mkdir(exist_ok=True)
    rng = random.Random(0)

    print("[INFO] Loading OCR-Bench + pt-448...", flush=True)
    ds = load_dataset("echo840/OCRBench", split="test")
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

    all_examples = []  # for the doc

    for cfg in CONFIGS:
        key = cfg["key"]; layer = cfg["layer"]; feature = cfg["feature"]
        print(f"\n=== {key} ({cfg['category']}, +{cfg['delta']}pp) ===", flush=True)

        ap = SAE_ACTS_DIR / f"acts_{key}.json"
        ad = json.load(open(ap))
        rF = sorted({int(x) for x, v in ad.get("acts", {}).items() if v > 0})

        # Build CAA
        needed_layers = sorted(set(BACKBONE_LAYERS) | {MIDDLE_LAYER, layer})
        caa_unit = {}
        for L in needed_layers:
            v = compute_paired_caa(L, rF)
            if v is not None:
                caa_unit[L] = v / v.norm().clamp(min=1e-8)
        w_dec = _load_wdec(layer, feature)

        # Build steering injection (D recipe)
        inject_pairs = []
        for L in BACKBONE_LAYERS:
            if L not in caa_unit: continue
            if L == layer:
                sv = (caa_unit[L]*cfg["alpha"] + w_dec*cfg["gamma"]).to(dtype).to(device)
            else:
                sv = (caa_unit[L]*cfg["alpha"]).to(dtype).to(device)
            inject_pairs.append((L, sv))

        # Generate MCQ data (same seed as eval)
        feat_examples = []
        for si in rF:
            ex = ds[si]
            img = ex.get("image"); gt = _parse_gt(ex.get("answer"))
            q = str(ex.get("question", "")).strip() or "What is the text?"
            if img is None or not gt: continue
            d1 = distort_answer(gt, rng, n_subs=2)
            d2 = distort_answer(gt, rng, n_subs=3)
            d3 = distort_answer(gt, rng, n_subs=2)
            gt_pos = rng.randrange(4)
            non_gt = [d1, d2, d3]
            choices = list(non_gt); choices.insert(gt_pos, gt); choices = choices[:4]

            try:
                img = img.convert("RGB")
                prompt = make_mcq_prompt(q, choices)
                iids, attn, pv = process_vlm_inputs(img, prompt, proc, mdl, device=device)
                _, img_end_r[0] = get_image_token_positions(iids)

                # Baseline (no steering)
                with torch.no_grad():
                    out = mdl(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                logits = out.logits[0, -1, :].float()
                ll = {L: logits[lid].item() for L, lid in letter_tok_ids.items()}
                base_letter = max(ll, key=ll.get)
                base_idx = ["A","B","C","D"].index(base_letter)
                base_correct = (base_idx == gt_pos)

                # With steering
                hooks = []
                for (l, sv) in inject_pairs:
                    hooks.append(mdl.model.language_model.layers[l].register_forward_hook(make_hook(sv)))
                with torch.no_grad():
                    out_s = mdl(input_ids=iids, attention_mask=attn, pixel_values=pv, use_cache=False)
                for h in hooks:
                    try: h.remove()
                    except: pass
                logits_s = out_s.logits[0, -1, :].float()
                ll_s = {L: logits_s[lid].item() for L, lid in letter_tok_ids.items()}
                steer_letter = max(ll_s, key=ll_s.get)
                steer_idx = ["A","B","C","D"].index(steer_letter)
                steer_correct = (steer_idx == gt_pos)

                feat_examples.append({
                    "si": si, "q": q, "gt": gt, "choices": choices, "gt_pos": gt_pos,
                    "base_letter": base_letter, "steer_letter": steer_letter,
                    "base_correct": base_correct, "steer_correct": steer_correct,
                    "base_logits": ll, "steer_logits": ll_s,
                })
            except Exception:
                continue

        # Find wins (wrong baseline → correct steered)
        wins = [e for e in feat_examples if not e["base_correct"] and e["steer_correct"]]
        already = [e for e in feat_examples if e["base_correct"] and e["steer_correct"]]
        regressions = [e for e in feat_examples if e["base_correct"] and not e["steer_correct"]]
        print(f"  wins(wrong→right)={len(wins)}  unchanged-correct={len(already)}  "
              f"regressions(right→wrong)={len(regressions)}  total_eval={len(feat_examples)}", flush=True)

        # Save up to 4 wins per feature with images
        all_examples.append({"cfg": cfg, "wins": wins[:5], "regressions": regressions[:2],
                             "stats": {"n": len(feat_examples), "wins": len(wins),
                                       "regressions": len(regressions)}})

        # Save images for wins
        for ex in wins[:5]:
            si = ex["si"]
            img = ds[si].get("image").convert("RGB")
            img_path = DOC_DIR / "images" / f"{key}_si{si:04d}.jpg"
            # Resize for doc
            img.thumbnail((400, 400))
            img.save(img_path, "JPEG", quality=85)
            ex["image_path"] = f"images/{key}_si{si:04d}.jpg"
        for ex in regressions[:2]:
            si = ex["si"]
            img = ds[si].get("image").convert("RGB")
            img_path = DOC_DIR / "images" / f"{key}_si{si:04d}.jpg"
            img.thumbnail((400, 400))
            img.save(img_path, "JPEG", quality=85)
            ex["image_path"] = f"images/{key}_si{si:04d}.jpg"

    # Generate the markdown doc
    doc = ["# OCR Steering — Hand-picked Examples\n",
           "Side-by-side comparison: pt-448 with no steering (baseline) vs pt-448 with",
           "D_BB+WDEC steering. Format: 4-choice MCQ. Steering flips wrong → right.\n"]
    for fe in all_examples:
        cfg = fe["cfg"]; wins = fe["wins"]; regs = fe["regressions"]; stats = fe["stats"]
        doc.append(f"\n## {cfg['key']} — {cfg['category']}\n")
        doc.append(f"**Best config**: D recipe at α={cfg['alpha']}, γ={cfg['gamma']}, "
                   f"δ=**+{cfg['delta']}pp** (n={stats['n']})")
        doc.append(f"- Wins (wrong → right): **{stats['wins']}**")
        doc.append(f"- Regressions (right → wrong): {stats['regressions']}\n")

        if wins:
            doc.append("### Wins (steering flipped wrong → right)\n")
            for i, e in enumerate(wins):
                doc.append(f"#### Example {i+1} — sample {e['si']}\n")
                doc.append(f"![sample {e['si']}]({e['image_path']})\n")
                doc.append(f"**Question**: {e['q']}")
                doc.append(f"**Ground truth**: `{e['gt']}` (option {chr(65+e['gt_pos'])})")
                doc.append(f"**Choices**:")
                for j, c in enumerate(e["choices"]):
                    mark = "✓" if j == e["gt_pos"] else " "
                    doc.append(f"- ({chr(65+j)}) {c} {mark}")
                doc.append(f"**Baseline pt-448**: picked `({e['base_letter']})` ✗")
                doc.append(f"**With steering**: picked `({e['steer_letter']})` ✓")
                bl = e["base_logits"]; sl = e["steer_logits"]
                doc.append(f"\nLogit shift on correct letter `({chr(65+e['gt_pos'])})`: "
                          f"{bl[chr(65+e['gt_pos'])]:.2f} → {sl[chr(65+e['gt_pos'])]:.2f}")
                doc.append("")

        if regs:
            doc.append("### Regressions (steering flipped right → wrong) — for honesty\n")
            for i, e in enumerate(regs):
                doc.append(f"#### Regression {i+1} — sample {e['si']}\n")
                doc.append(f"![sample {e['si']}]({e['image_path']})\n")
                doc.append(f"GT `{e['gt']}` — baseline ✓ ({e['base_letter']}), "
                          f"steered ✗ ({e['steer_letter']})\n")

    doc_text = "\n".join(doc)
    out_path = DOC_DIR / "examples.md"
    out_path.write_text(doc_text)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
