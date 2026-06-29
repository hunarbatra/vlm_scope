#!/usr/bin/env python3
"""
Step 12b — ablation on 100 MSSBench-safe samples (truly-benign control).

Mirrors 35_ablate_safety_features.py but loads from a JSONL eval set with
absolute image paths instead of HuggingFace VLSBench.

Output dir: analysis_safety/ablation_results_mssbench_safe/
"""
import os, sys, json, gc, warnings, argparse
from pathlib import Path
import pandas as pd
import torch
import torch.multiprocessing as mp
from PIL import Image

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ab35", Path(__file__).parent / "35_ablate_safety_features.py")
ab35 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab35)

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
ANALYSIS_DIR = ROOT / "analysis_safety"
FEATURES_CSV = ANALYSIS_DIR / "ablation_input" / "features_to_ablate.csv"
SAFE_EVAL    = ANALYSIS_DIR / "mssbench_safe" / "safe_eval.jsonl"
OUT_DIR      = ANALYSIS_DIR / "ablation_results_mssbench_safe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_GPUS = 8
MAX_NEW_TOKENS = 80


def _safe_worker(gpu_id, feature_rows, out_dir=None):
    global OUT_DIR
    if out_dir is not None:
        OUT_DIR = Path(out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, "/home/hbatra/vlm_scope_backup/vlm_scope/finetune/paligemma2")
    from utils import initialize_jumprelu_sae
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    print(f"[GPU{gpu_id}] loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(ab35.MODEL_NAME, cache_dir=ab35.HF_CACHE,
                                              local_files_only=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        ab35.MODEL_NAME, torch_dtype=torch.bfloat16,
        cache_dir=ab35.HF_CACHE, local_files_only=True
    ).to(device).eval()
    tokenizer = processor.tokenizer
    model_dtype = next(model.parameters()).dtype

    print(f"[GPU{gpu_id}] loading MSSBench-safe eval set...", flush=True)
    eval_records = [json.loads(l) for l in open(SAFE_EVAL) if l.strip()]
    print(f"[GPU{gpu_id}] eval set: {len(eval_records)} samples", flush=True)

    img_end_ref = {"v": 0}

    def _generate_ablated(image, prompt):
        inputs = processor(text=prompt, images=image.convert("RGB"),
                           return_tensors="pt").to(device, torch.bfloat16)
        img_end_ref["v"] = ab35._get_image_end(inputs["input_ids"])
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                use_cache=True, pad_token_id=tokenizer.pad_token_id,
            )
        return processor.decode(out_ids[0, input_len:], skip_special_tokens=True).strip()

    for feat_i, row in enumerate(feature_rows):
        L = int(row["layer"]); F = int(row["feature"])
        cat = row["selected_for_category"]; is_ctrl = int(row["is_control"])
        resp_path = OUT_DIR / f"responses_L{L}_F{F}.jsonl"
        if resp_path.exists():
            print(f"[GPU{gpu_id}] L{L}/F{F}: done, skip", flush=True); continue

        ckpt = ab35.CHECKPOINT_DIR / f"text-only_layer_{L}.pt"
        sae = initialize_jumprelu_sae(L, checkpoint_path=str(ckpt),
                                      device=device, cache_dir=ab35.HF_CACHE)
        sae.eval()
        feature_vec = sae.W_dec[F].detach().to(model_dtype).to(device)
        feature_vec = feature_vec / feature_vec.norm().clamp(min=1e-8)
        del sae; torch.cuda.empty_cache()

        print(f"[GPU{gpu_id}] [{feat_i+1}/{len(feature_rows)}] L{L}/F{F} "
              f"cat={cat} ctrl={is_ctrl}", flush=True)

        hooks = ab35.AblationHooks(model, feature_vec, img_end_ref)
        hooks.register()
        try:
            with open(resp_path, "w") as outf:
                for rec in eval_records:
                    iid = rec["instruction_id"]
                    prompt = f"answer en {rec['instruction']}"
                    try:
                        img = Image.open(rec["image_path"])
                        text = _generate_ablated(img, prompt)
                        out_rec = {**rec, "response": text, "status": "ok"}
                    except Exception as e:
                        out_rec = {**rec, "response": None, "status": f"error: {e}"}
                    outf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    outf.flush()
        finally:
            hooks.remove()
        torch.cuda.empty_cache(); gc.collect()

    print(f"[GPU{gpu_id}] all features done", flush=True)


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=list(range(N_GPUS)))
    ap.add_argument("--features", type=str, default=str(FEATURES_CSV))
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    OUT_DIR = Path(args.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats_path = Path(args.features)
    df = pd.read_csv(feats_path)
    print(f"[MAIN] features={feats_path.name}  out={OUT_DIR}")
    print(f"[MAIN] {len(df)} features × 100 MSSBench-safe samples")

    shards = [[] for _ in args.gpus]
    for i, row in df.iterrows():
        shards[i % len(args.gpus)].append(row.to_dict())
    for g, s in zip(args.gpus, shards):
        print(f"  GPU{g}: {len(s)} features")

    mp.set_start_method("spawn", force=True)
    procs = []
    for gpu_id, shard in zip(args.gpus, shards):
        if not shard: continue
        p = mp.Process(target=_safe_worker, args=(gpu_id, shard, str(OUT_DIR)))
        p.start(); procs.append(p)
    for p in procs: p.join()
    print("[MAIN] all workers done")


if __name__ == "__main__":
    main()
