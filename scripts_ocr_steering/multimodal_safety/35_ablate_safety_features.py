#!/usr/bin/env python3
"""
Step 10 (safety) — per-feature ablation on VLSBench + VQA control.

For each feature F in features_to_ablate.csv:
  1. Regenerate VLSBench responses with F ablated (3-point projection:
     self_attn.output, mlp.output, layer.output at every layer).
     Implementation uses torch forward hooks → native model.generate()
     to leverage KV cache (~10–20× faster than per-token nnsight trace).
  2. Measure VQA yes/no accuracy with the same ablation applied.

Outputs:
  analysis_safety/ablation_results/responses_L{L}_F{F}.jsonl
  analysis_safety/ablation_results/vqa_L{L}_F{F}.json
  analysis_safety/ablation_results/vqa_baseline_gpu{G}.json

Step 36 judges the ablated responses with Qwen3-VL-8B to compute ΔASR.

Usage: python3 -B 35_ablate_safety_features.py
"""
import os, sys, json, math, gc, warnings, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")

MODEL_NAME = "google/paligemma2-3b-mix-448"
N_LAYERS = 26
D_SAE = 16384
N_GPUS = 8

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
CHECKPOINT_DIR = ROOT / "checkpoints"
HF_CACHE       = "/data1/vlm_scope_sae_docci/hf_cache/hub"
ANALYSIS_DIR   = ROOT / "analysis_safety"
FEATURES_CSV   = ANALYSIS_DIR / "ablation_input" / "features_to_ablate.csv"
JUDGE_FILE     = ANALYSIS_DIR / "judgments" / "mix448_vlsbench_qwen_judgments.jsonl"
OUT_DIR        = ANALYSIS_DIR / "ablation_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_VLSBENCH_EVAL = 100   # default; CLI override possible
N_VQA_EVAL      = 1000  # matches original spatial ablation
MAX_NEW_TOKENS  = 80  # most unsafe/refusal content shows up by this point

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"]  = "/data1/hbatra/mmdiff/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")


def load_eval_set():
    """Default behavior: stratified subsample of N_VLSBENCH_EVAL.
    If N_VLSBENCH_EVAL is None or >=835, use ALL baseline-UNSAFE samples."""
    per_cat = defaultdict(list)
    with open(JUDGE_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r.get("judge_label") == "UNSAFE":
                per_cat[r["category"]].append(r["instruction_id"])
    if N_VLSBENCH_EVAL is None or N_VLSBENCH_EVAL >= 835:
        out = []
        for cat, ids in per_cat.items():
            for iid in ids:
                out.append((iid, cat))
        return out
    total = sum(len(v) for v in per_cat.values())
    stratified = []
    for cat, ids in per_cat.items():
        n = max(1, round(N_VLSBENCH_EVAL * len(ids) / total))
        stride = max(1, len(ids) // n)
        picks = ids[::stride][:n]
        for iid in picks:
            stratified.append((iid, cat))
    return stratified[:N_VLSBENCH_EVAL]


def _get_yes_no_ids(tokenizer):
    yes_ids, no_ids = set(), set()
    for t in [" Yes", "Yes", " yes", "YES"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: yes_ids.add(toks[0])
    for t in [" No", "No", " no", "NO"]:
        toks = tokenizer.encode(t, add_special_tokens=False)
        if toks: no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap; no_ids -= overlap
    return yes_ids, no_ids


def _predict_yesno(logits, yes_ids, no_ids):
    probs = torch.softmax(logits, dim=-1)
    y = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
    n = probs[list(no_ids)].sum().item() if no_ids else 0.0
    d = y + n
    return 1 if (y / d if d > 0 else 0) > 0.5 else 0


class AblationHooks:
    """Registers forward hooks that subtract a feature-direction projection from
    self_attn.output, mlp.output, and layer output at every decoder layer.

    During the prompt-phase forward pass (T > 1), ablation is applied only to
    text tokens (position >= img_end). During KV-cache generation (T == 1),
    every new token is text, so ablation is applied unconditionally.
    """

    def __init__(self, model, feature_vec, img_end_ref):
        self.model = model
        self.fv = feature_vec  # (D,)
        self.img_end_ref = img_end_ref  # mutable dict with 'v' key
        self.handles = []

    def _project_out(self, hidden):
        # hidden: (B, T, D); fv: (D,)
        T = hidden.shape[1]
        if T > 1:
            start = self.img_end_ref["v"]
            if start >= T:
                return hidden
            dots = torch.einsum("btd,d->bt", hidden[:, start:], self.fv)
            proj = dots.unsqueeze(-1) * self.fv
            hidden = hidden.clone()
            hidden[:, start:] = hidden[:, start:] - proj
        else:
            dots = torch.einsum("btd,d->bt", hidden, self.fv)
            proj = dots.unsqueeze(-1) * self.fv
            hidden = hidden - proj
        return hidden

    def _wrap_tuple_output(self, hook):
        def wrapped(module, inputs, output):
            if isinstance(output, tuple):
                new_h = hook(output[0])
                return (new_h,) + output[1:]
            return hook(output)
        return wrapped

    def register(self):
        for l in range(N_LAYERS):
            layer = self.model.model.language_model.layers[l]
            self.handles.append(
                layer.self_attn.register_forward_hook(self._wrap_tuple_output(self._project_out))
            )
            self.handles.append(
                layer.mlp.register_forward_hook(self._wrap_tuple_output(self._project_out))
            )
            self.handles.append(
                layer.register_forward_hook(self._wrap_tuple_output(self._project_out))
            )

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


IMAGE_TOKEN_ID = 257152


def _get_image_end(input_ids):
    ids = input_ids[0].tolist()
    end = 0
    started = False
    for i, t in enumerate(ids):
        if t == IMAGE_TOKEN_ID:
            started = True
            end = i + 1
        elif started:
            break
    return end


def _ablation_worker(gpu_id, feature_rows, n_vlsbench=None, out_dir=None):
    global N_VLSBENCH_EVAL, OUT_DIR
    if n_vlsbench is not None:
        N_VLSBENCH_EVAL = n_vlsbench
    if out_dir is not None:
        OUT_DIR = Path(out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, "/home/hbatra/vlm_scope_backup/vlm_scope/finetune/paligemma2")
    from utils import initialize_jumprelu_sae
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from datasets import load_dataset

    print(f"[GPU{gpu_id}] loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE, local_files_only=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=HF_CACHE, local_files_only=True
    ).to(device).eval()
    tokenizer = processor.tokenizer
    model_dtype = next(model.parameters()).dtype
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)

    print(f"[GPU{gpu_id}] loading VLSBench...", flush=True)
    ds_vls = load_dataset("Foreshhh/vlsbench", split="train")
    eval_ids = load_eval_set()
    eval_set = {iid for iid, _ in eval_ids}
    vls_dsidxs = [i for i in range(len(ds_vls)) if str(ds_vls[i]["instruction_id"]) in eval_set]
    print(f"[GPU{gpu_id}] VLSBench eval set: {len(vls_dsidxs)} samples", flush=True)

    print(f"[GPU{gpu_id}] loading VQA yes/no...", flush=True)
    vqa = load_dataset("lmms-lab/VQAv2", split="validation")
    vqa_yesno = []
    for i in range(len(vqa)):
        ex = vqa[i]
        at = str(ex.get("answer_type", "")).lower()
        mc = str(ex.get("multiple_choice_answer", "")).strip().lower()
        if at == "yes/no" and mc in {"yes", "no"}:
            vqa_yesno.append((i, 1 if mc == "yes" else 0))
            if len(vqa_yesno) >= N_VQA_EVAL:
                break
    print(f"[GPU{gpu_id}] VQA yes/no: {len(vqa_yesno)}", flush=True)

    img_end_ref = {"v": 0}

    def _generate_with_ablation(image, prompt, hooks):
        inputs = processor(text=prompt, images=image.convert("RGB"),
                           return_tensors="pt").to(device, torch.bfloat16)
        img_end_ref["v"] = _get_image_end(inputs["input_ids"])
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                use_cache=True, pad_token_id=tokenizer.pad_token_id,
            )
        return processor.decode(out_ids[0, input_len:], skip_special_tokens=True).strip()

    def _generate_baseline(image, prompt):
        inputs = processor(text=prompt, images=image.convert("RGB"),
                           return_tensors="pt").to(device, torch.bfloat16)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                use_cache=True, pad_token_id=tokenizer.pad_token_id,
            )
        return processor.decode(out_ids[0, input_len:], skip_special_tokens=True).strip()

    def _vqa_one(ex, ablated=False):
        img = ex.get("image")
        if img is None: return -1
        question = ex.get("question", "")
        prompt = ("Answer the following question with only 'Yes' or 'No':\n"
                  f"Question: {question.strip()}\nAnswer:")
        inputs = processor(text=prompt, images=img.convert("RGB"),
                           return_tensors="pt").to(device, torch.bfloat16)
        if ablated:
            img_end_ref["v"] = _get_image_end(inputs["input_ids"])
        with torch.inference_mode():
            out = model(**inputs, use_cache=False)
        return _predict_yesno(out.logits[0, -1, :], yes_ids, no_ids)

    def _vqa_baseline():
        c = t = 0
        for qi, label in vqa_yesno:
            pred = _vqa_one(vqa[qi], ablated=False)
            if pred < 0: continue
            t += 1
            if pred == label: c += 1
        return {"vqa_acc": c / max(t, 1) * 100, "vqa_correct": c, "vqa_total": t}

    def _vqa_ablated():
        c = t = 0
        for qi, label in vqa_yesno:
            try:
                pred = _vqa_one(vqa[qi], ablated=True)
            except Exception:
                pred = -1
            if pred < 0: continue
            t += 1
            if pred == label: c += 1
        return {"vqa_acc": c / max(t, 1) * 100, "vqa_correct": c, "vqa_total": t}

    # --- Per-GPU VQA baseline ---
    vqa_base_path = OUT_DIR / f"vqa_baseline_gpu{gpu_id}.json"
    if vqa_base_path.exists():
        vqa_baseline = json.load(open(vqa_base_path))
        print(f"[GPU{gpu_id}] loaded VQA baseline: {vqa_baseline['vqa_acc']:.1f}%", flush=True)
    else:
        print(f"[GPU{gpu_id}] computing VQA baseline...", flush=True)
        vqa_baseline = _vqa_baseline()
        with open(vqa_base_path, "w") as f:
            json.dump(vqa_baseline, f, indent=2)
        print(f"[GPU{gpu_id}] VQA baseline: {vqa_baseline['vqa_acc']:.1f}%", flush=True)

    # --- Per-feature loop ---
    for feat_i, row in enumerate(feature_rows):
        layer_idx = int(row["layer"])
        feature_idx = int(row["feature"])
        cat = row["selected_for_category"]
        is_ctrl = int(row["is_control"])

        resp_path = OUT_DIR / f"responses_L{layer_idx}_F{feature_idx}.jsonl"
        vqa_path  = OUT_DIR / f"vqa_L{layer_idx}_F{feature_idx}.json"
        if resp_path.exists() and vqa_path.exists():
            print(f"[GPU{gpu_id}] L{layer_idx}/F{feature_idx}: done, skip", flush=True)
            continue

        ckpt = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()
        feature_vec = sae.W_dec[feature_idx].detach().to(model_dtype).to(device)
        feature_vec = feature_vec / feature_vec.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()

        print(f"[GPU{gpu_id}] [{feat_i+1}/{len(feature_rows)}] L{layer_idx}/F{feature_idx} "
              f"cat={cat} ctrl={is_ctrl}", flush=True)

        hooks = AblationHooks(model, feature_vec, img_end_ref)
        hooks.register()
        try:
            # --- VLSBench ablated generations ---
            if not resp_path.exists():
                with open(resp_path, "w") as outf:
                    for i, dsidx in enumerate(vls_dsidxs):
                        sample = ds_vls[dsidx]
                        iid = str(sample["instruction_id"])
                        try:
                            prompt = f"answer en {sample['instruction']}"
                            text = _generate_with_ablation(sample["image"], prompt, hooks)
                            rec = {
                                "instruction_id": iid, "category": sample["category"],
                                "sub_category": sample.get("sub_category"),
                                "instruction": sample["instruction"],
                                "image_description": sample.get("image_description"),
                                "safety_reason": sample.get("safety_reason"),
                                "response": text, "status": "ok",
                            }
                        except Exception as e:
                            rec = {"instruction_id": iid, "response": None,
                                   "status": f"error: {e}"}
                        outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        outf.flush()

            # --- VQA ablated ---
            if not vqa_path.exists():
                vqa_res = _vqa_ablated()
                vqa_res["baseline_vqa_acc"] = vqa_baseline["vqa_acc"]
                vqa_res["delta_vqa"] = vqa_res["vqa_acc"] - vqa_baseline["vqa_acc"]
                vqa_res["layer"] = layer_idx
                vqa_res["feature"] = feature_idx
                vqa_res["selected_for_category"] = cat
                vqa_res["is_control"] = is_ctrl
                with open(vqa_path, "w") as f:
                    json.dump(vqa_res, f, indent=2)
                print(f"[GPU{gpu_id}]   VQA: {vqa_res['vqa_acc']:.1f}% "
                      f"(base {vqa_baseline['vqa_acc']:.1f}%, ∆{vqa_res['delta_vqa']:+.1f})", flush=True)
        finally:
            hooks.remove()

        torch.cuda.empty_cache(); gc.collect()

    print(f"[GPU{gpu_id}] all features done", flush=True)


def main():
    global N_VLSBENCH_EVAL, OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(N_GPUS)))
    parser.add_argument("--features", type=str, default=str(FEATURES_CSV))
    parser.add_argument("--n-vlsbench", type=int, default=N_VLSBENCH_EVAL,
                        help="VLSBench eval size; >=835 uses all baseline-UNSAFE")
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    N_VLSBENCH_EVAL = args.n_vlsbench
    OUT_DIR = Path(args.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats_path = Path(args.features)
    if not feats_path.exists():
        print(f"[FATAL] {feats_path} missing"); return
    df = pd.read_csv(feats_path)
    print(f"[MAIN] features={feats_path.name}  N_VLSBENCH_EVAL={N_VLSBENCH_EVAL}  out={OUT_DIR}")
    print(f"[MAIN] {len(df)} features to ablate across {len(args.gpus)} GPUs")

    shards = [[] for _ in args.gpus]
    for i, row in df.iterrows():
        shards[i % len(args.gpus)].append(row.to_dict())
    for g, s in zip(args.gpus, shards):
        print(f"  GPU{g}: {len(s)} features")

    mp.set_start_method("spawn", force=True)
    procs = []
    for gpu_id, shard in zip(args.gpus, shards):
        if not shard: continue
        p = mp.Process(target=_ablation_worker,
                       args=(gpu_id, shard, N_VLSBENCH_EVAL, str(OUT_DIR)))
        p.start(); procs.append(p)
    for p in procs: p.join()
    print("[MAIN] all workers done")


if __name__ == "__main__":
    main()
