#!/usr/bin/env python3
"""
Step 5 (hallucination variant) — per-token SAE firing on HallusionBench INCORRECT samples.

Mirrors 30_firing_vlsbench_unsafe.py; swaps VLSBench→HallusionBench and UNSAFE→INCORRECT.

Outputs:
  analysis_hallucination/_greedy/firing_incorrect_pertoken/firing_halluc_layer_L.json
  analysis_hallucination/_greedy/firing_incorrect_by_subcat/firing_halluc_<SUBCAT>_layer_L.json
"""
import json, os, sys, gc, warnings
from pathlib import Path
from collections import defaultdict
import torch
import torch.multiprocessing as mp
import numpy as np

warnings.filterwarnings("ignore")

MODEL_NAME = "google/paligemma2-3b-mix-448"
N_LAYERS = 26
D_SAE = 16384
N_GPUS = 8

ROOT = Path("/data1/vlm_scope_sae_mix448_textonly")
CHECKPOINT_DIR = ROOT / "checkpoints"
HF_CACHE = "/data1/vlm_scope_sae_docci/hf_cache/hub"
ANALYSIS_DIR = ROOT / "analysis_hallucination/_greedy"
JUDGE_FILE   = ROOT / "analysis_hallucination" / "judgments" / "mix448_hallusionbench_judgments.jsonl.greedy_backup"
OUT_DIR      = ANALYSIS_DIR / "firing_incorrect_pertoken"
OUT_SC_DIR   = ANALYSIS_DIR / "firing_incorrect_by_subcat"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SC_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_DATASETS_CACHE"] = "/data1/vlm_scope_sae_mix448_textonly/hf_datasets_cache"
os.environ["HF_HOME"] = "/data1/hbatra/mmdiff/hf_cache"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


def load_incorrect_ids():
    out = []
    with open(JUDGE_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r.get("pred_label") == "INCORRECT":
                out.append((r["uid"], r.get("subcategory", "")))
    return out


def _uid(s):
    return f"{s['category']}_{s['subcategory']}_set{s['set_id']}_fig{s['figure_id']}_q{s['question_id']}"


def _firing_worker(gpu_id, layer_indices):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    sys.path.insert(0, "/home/hbatra/vlm_scope_backup/vlm_scope/finetune/paligemma2")
    from utils import initialize_jumprelu_sae, process_vlm_inputs, get_image_token_positions
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from nnsight import NNsight
    from datasets import load_dataset

    remaining = [l for l in layer_indices if not (OUT_DIR / f"firing_halluc_layer_{l}.json").exists()]
    if not remaining:
        print(f"[Firing GPU{gpu_id}] All layers done, skipping", flush=True); return

    print(f"[Firing GPU{gpu_id}] Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE, local_files_only=True)
    model_raw = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, cache_dir=HF_CACHE, local_files_only=True
    ).to(device).eval()
    nns_model = NNsight(model_raw)

    print(f"[Firing GPU{gpu_id}] Loading HallusionBench...", flush=True)
    ds = load_dataset("lmms-lab/HallusionBench", split="image")
    incorrect = load_incorrect_ids()
    incorrect_ids = {uid for uid, _ in incorrect}
    incorrect_sc = {uid: sc for uid, sc in incorrect}
    keep_idx = [i for i in range(len(ds)) if _uid(ds[i]) in incorrect_ids]
    print(f"[Firing GPU{gpu_id}] {len(keep_idx)}/{len(ds)} INCORRECT samples", flush=True)

    for layer_idx in remaining:
        ckpt_path = CHECKPOINT_DIR / f"text-only_layer_{layer_idx}.pt"
        sae = initialize_jumprelu_sae(layer_idx, checkpoint_path=str(ckpt_path),
                                      device=device, cache_dir=HF_CACHE)
        sae.eval()

        fire_total = np.zeros(D_SAE, dtype=np.int64)
        n_tokens_total = 0
        n_samples_total = 0
        fire_by_sc  = defaultdict(lambda: np.zeros(D_SAE, dtype=np.int64))
        n_tok_by_sc = defaultdict(int)
        n_samp_by_sc = defaultdict(int)
        n_errors = 0

        for idx in keep_idx:
            try:
                sample = ds[idx]
                uid = _uid(sample)
                sc = incorrect_sc[uid]
                image = sample["image"].convert("RGB")
                prompt = f"answer en {sample['question']}"

                input_ids, attn_mask, pixel_values = process_vlm_inputs(
                    image, prompt, processor, model_raw, device=device,
                )
                img_s, img_e = get_image_token_positions(input_ids)

                with torch.no_grad():
                    with nns_model.trace(
                        input_ids=input_ids, attention_mask=attn_mask,
                        pixel_values=pixel_values, use_cache=False,
                    ) as tr:
                        layer_out = nns_model.model.language_model.layers[layer_idx].output[0].save()

                act = layer_out.detach().squeeze(0).float()
                with torch.no_grad():
                    codes = sae.encode(act).detach()

                seq_len = codes.shape[0]
                txt_mask = torch.ones(seq_len, dtype=torch.bool, device=codes.device)
                if img_e > img_s:
                    txt_mask[img_s:img_e] = False

                txt_codes = codes[txt_mask]
                if txt_codes.shape[0] > 0:
                    fired = (txt_codes > 0).sum(dim=0).cpu().numpy().astype(np.int64)
                    fire_total += fired
                    n_tokens_total += txt_codes.shape[0]
                    fire_by_sc[sc] += fired
                    n_tok_by_sc[sc] += txt_codes.shape[0]
                    n_samp_by_sc[sc] += 1
                n_samples_total += 1
            except Exception as e:
                n_errors += 1
                if n_errors <= 5:
                    print(f"[Firing GPU{gpu_id}] L{layer_idx} err idx={idx}: {e}", flush=True)
                continue

            if n_samples_total % 50 == 0 and gpu_id == 0:
                print(f"  L{layer_idx}: {n_samples_total}/{len(keep_idx)}", flush=True)

        total_data = {
            "layer": int(layer_idx), "n_samples": int(n_samples_total),
            "n_tokens": int(n_tokens_total), "n_errors": int(n_errors),
            "dataset": "hallusionbench_incorrect",
            "fire_count": fire_total.tolist(),
        }
        with open(OUT_DIR / f"firing_halluc_layer_{layer_idx}.json", "w") as f:
            json.dump(total_data, f)
        for sc in fire_by_sc:
            sc_tag = sc.replace(" ", "_").replace("/", "_")
            sc_data = {
                "layer": int(layer_idx), "n_samples": int(n_samp_by_sc[sc]),
                "n_tokens": int(n_tok_by_sc[sc]),
                "dataset": "hallusionbench_incorrect",
                "subcategory": sc,
                "fire_count": fire_by_sc[sc].tolist(),
            }
            with open(OUT_SC_DIR / f"firing_halluc_{sc_tag}_layer_{layer_idx}.json", "w") as f:
                json.dump(sc_data, f)

        active_1pct = (fire_total > max(n_tokens_total, 1) * 0.01).sum()
        print(f"[Firing GPU{gpu_id}] L{layer_idx}: {n_samples_total} samples, "
              f"{n_tokens_total} tokens, features >1%: {active_1pct}", flush=True)
        del sae; torch.cuda.empty_cache(); gc.collect()


def main():
    mp.set_start_method("spawn", force=True)
    layer_indices = list(range(N_LAYERS))
    shards = [[] for _ in range(N_GPUS)]
    for i, l in enumerate(layer_indices):
        shards[i % N_GPUS].append(l)
    print(f"[MAIN] Launching {N_GPUS} workers, shards:")
    for g, s in enumerate(shards): print(f"  GPU{g}: {s}")
    procs = []
    for g in range(N_GPUS):
        p = mp.Process(target=_firing_worker, args=(g, shards[g]))
        p.start(); procs.append(p)
    for p in procs: p.join()
    print("[MAIN] All workers done.")


if __name__ == "__main__":
    main()
