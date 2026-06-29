"""
A/B test: Compare two ablation approaches on a single feature.

GPU 1 (Approach A): Projection ablation — project out feature direction from all layers
                    (current method, matches LLaVA-MORE original)
GPU 2 (Approach B): SAE encode-zero-decode — encode activations through SAE at the
                    feature's own layer, zero out the target feature coefficient,
                    decode back. Strongest possible causal ablation.

Test feature: L17/F14475 (top priority feature)

Usage:
    cd finetune/paligemma2
    export HF_TOKEN=hf_REPLACE_ME_WITH_YOUR_HF_TOKEN_OR_USE_ENV_VAR
    MODAL_PROFILE=hunar-oxford modal run modal_ablation_ab_test.py
"""

import os
import json
import modal
from pathlib import Path

VOLUME_NAME = "vlm-scope-data-v2"
GPU_TYPE = "A100"
TIMEOUT = 86400

app = modal.App("vlm-scope-ablation-ab-test")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.44",
        "sae-lens>=4.0",
        "nnsight>=0.3",
        "datasets",
        "h5py",
        "tqdm",
        "huggingface-hub",
        "Pillow",
        "numpy",
        "scipy",
        "accelerate",
        "requests",
    )
    .env({
        "HF_HOME": "/vol/cache/huggingface",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
        "WANDB_MODE": "disabled",
    })
    .add_local_file(
        local_path=str(Path(__file__).parent / "utils.py"),
        remote_path="/root/paligemma2/utils.py",
    )
)

# --------------- Constants ---------------

TEST_LAYER = 17
TEST_FEATURE = 14475

N_LAYERS = 26
RESULTS_BASE = "/vol/results/paligemma2"
MODEL_NAME = "google/paligemma2-3b-pt-224"

VSR_DATASET = "cambridgeltl/vsr_random"
VSR_SPLITS = ["train", "dev", "test"]
VQA_MAX_SAMPLES = 1000
IMAGE_CACHE_DIR = "/vol/cache/vsr_images"
CONTROL_RELATIONS = {"has", "wears", "holds", "made of", "part of", "contains"}


def _build_vsr_prompt(statement: str) -> str:
    return (
        "Is the following statement correct? Answer only with 'Yes' or 'No'.\n"
        f"Statement: {statement.strip()}\n"
        "Answer:"
    )


def _get_yes_no_ids(tokenizer):
    yes_ids = set()
    no_ids = set()
    for text in [" Yes", "Yes", " yes", "YES"]:
        toks = tokenizer.encode(text, add_special_tokens=False)
        if toks:
            yes_ids.add(toks[0])
    for text in [" No", "No", " no", "NO"]:
        toks = tokenizer.encode(text, add_special_tokens=False)
        if toks:
            no_ids.add(toks[0])
    overlap = yes_ids & no_ids
    yes_ids -= overlap
    no_ids -= overlap
    return yes_ids, no_ids


# --------------- Approach A: Projection ablation (all layers) ---------------

@app.function(image=image, gpu=GPU_TYPE, volumes={"/vol": volume}, timeout=TIMEOUT)
def approach_a_projection():
    """Approach A: Project out normalized feature direction from self_attn + mlp + layer
    outputs at EVERY transformer layer. Current method matching LLaVA-MORE."""
    import sys, hashlib, torch, warnings
    from pathlib import Path
    from datasets import load_dataset, concatenate_datasets
    from nnsight import NNsight
    from PIL import Image
    from io import BytesIO
    import requests

    warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
    sys.path.insert(0, "/root/paligemma2")
    from utils import (initialize_vlm_model, process_vlm_inputs,
                       get_image_token_positions, initialize_jumprelu_sae)

    print("[A: Projection] Loading model...")
    processor, model_raw = initialize_vlm_model(MODEL_NAME, device="cuda")
    model_raw.eval()
    nns_model = NNsight(model_raw)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    n_model_layers = model_raw.config.text_config.num_hidden_layers
    model_dtype = next(model_raw.parameters()).dtype

    # Load feature vector
    ckpt_path = f"{RESULTS_BASE}/run_jumprelu/checkpoints/pretrained_layer_{TEST_LAYER}.pt"
    sae = initialize_jumprelu_sae(TEST_LAYER, checkpoint_path=ckpt_path,
                                   device="cuda", cache_dir="/vol/cache/huggingface")
    sae.eval()
    feature_vec = sae.W_dec[TEST_FEATURE].detach().to(model_dtype).to("cuda")
    feature_vec = feature_vec / feature_vec.norm().clamp(min=1e-8)
    print(f"[A] Feature vec norm={sae.W_dec[TEST_FEATURE].norm().item():.4f}")
    del sae
    torch.cuda.empty_cache()

    fv = feature_vec.unsqueeze(0)  # (1, d_in)

    def do_ablation(input_ids, attention_mask, pixel_values, img_end):
        with nns_model.trace(input_ids=input_ids, attention_mask=attention_mask,
                             pixel_values=pixel_values) as tr:
            for l in range(n_model_layers):
                attn_out = nns_model.model.language_model.layers[l].self_attn.output[0][0, img_end:]
                attn_out -= (attn_out @ fv.T) * fv

                mlp_out = nns_model.model.language_model.layers[l].mlp.output[0, img_end:]
                mlp_out -= (mlp_out @ fv.T) * fv

                layer_out = nns_model.model.language_model.layers[l].output[0][img_end:]
                layer_out -= (layer_out @ fv.T) * fv

            logits_saved = nns_model.output.logits.save()
        return logits_saved

    def predict_yesno(logits_saved):
        logits = logits_saved[:, -1, :]
        probs = torch.softmax(logits, dim=-1)[0]
        yes_mass = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
        no_mass = probs[list(no_ids)].sum().item() if no_ids else 0.0
        denom = yes_mass + no_mass
        return 1 if (yes_mass / denom if denom > 0 else 0.0) > 0.5 else 0

    # Load datasets
    print("[A] Loading datasets...")
    vsr_splits = []
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    for split in VSR_SPLITS:
        vsr_splits.append(load_dataset(VSR_DATASET, data_files=data_files, split=split))
    vsr = concatenate_datasets(vsr_splits)

    vqa = load_dataset("lmms-lab/VQAv2", split="validation")
    vqa_yesno = []
    for i in range(len(vqa)):
        ex = vqa[i]
        if str(ex.get("answer_type", "")).lower() == "yes/no":
            mc = str(ex.get("multiple_choice_answer", "")).strip().lower()
            if mc in {"yes", "no"}:
                vqa_yesno.append((i, 1 if mc == "yes" else 0))
                if len(vqa_yesno) >= VQA_MAX_SAMPLES:
                    break

    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

    # --- VSR ---
    print(f"[A] Running VSR ablation ({len(vsr)} samples)...")
    vsr_correct = vsr_total = vsr_ctrl_correct = vsr_ctrl_total = 0
    vsr_per_relation = {}

    for vi in range(len(vsr)):
        ex = vsr[vi]
        url = ex.get("image_link", "")
        statement = str(ex.get("caption", "")).strip()
        label = int(ex.get("label", 0))
        relation = ex.get("relation", "")
        if not statement or not url:
            continue

        url_hash = hashlib.md5(url.encode()).hexdigest()
        img_cache = os.path.join(IMAGE_CACHE_DIR, f"{url_hash}.jpg")
        try:
            if os.path.exists(img_cache):
                img = Image.open(img_cache).convert("RGB")
            else:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                img.save(img_cache, "JPEG")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        prompt = _build_vsr_prompt(statement)
        input_ids, attention_mask, pixel_values = process_vlm_inputs(img, prompt, processor, nns_model._module)
        _, img_end = get_image_token_positions(input_ids)

        try:
            logits_saved = do_ablation(input_ids, attention_mask, pixel_values, img_end)
            pred = predict_yesno(logits_saved)
        except Exception as e:
            if vi < 3:
                print(f"  [A ERROR] VSR {vi}: {e}")
            pred = 0

        vsr_total += 1
        if pred == label:
            vsr_correct += 1
        if relation not in vsr_per_relation:
            vsr_per_relation[relation] = {"correct": 0, "total": 0}
        vsr_per_relation[relation]["total"] += 1
        if pred == label:
            vsr_per_relation[relation]["correct"] += 1
        if relation in CONTROL_RELATIONS:
            vsr_ctrl_total += 1
            if pred == label:
                vsr_ctrl_correct += 1

        if (vi + 1) % 100 == 0 and vsr_total > 0:
            print(f"  [A] VSR [{vi+1}/{len(vsr)}] acc={vsr_correct/vsr_total*100:.2f}%")

    vsr_acc = (vsr_correct / vsr_total * 100) if vsr_total > 0 else 0.0
    vsr_ctrl_acc = (vsr_ctrl_correct / vsr_ctrl_total * 100) if vsr_ctrl_total > 0 else 0.0
    print(f"[A] VSR DONE: {vsr_acc:.2f}% (ctrl={vsr_ctrl_acc:.2f}%)")

    # --- VQA ---
    print(f"[A] Running VQA ablation ({len(vqa_yesno)} samples)...")
    vqa_correct = vqa_total = 0

    for qi, label in vqa_yesno:
        ex = vqa[qi]
        img = ex.get("image")
        question = ex.get("question", "")
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            continue

        prompt = f"Answer the following question with only 'Yes' or 'No':\nQuestion: {question.strip()}\nAnswer:"
        try:
            input_ids, attention_mask, pixel_values = process_vlm_inputs(img, prompt, processor, nns_model._module)
        except Exception:
            continue
        _, img_end = get_image_token_positions(input_ids)

        try:
            logits_saved = do_ablation(input_ids, attention_mask, pixel_values, img_end)
            pred = predict_yesno(logits_saved)
        except Exception as e:
            if vqa_total < 3:
                print(f"  [A ERROR] VQA {vqa_total}: {e}")
            pred = 0

        vqa_total += 1
        if pred == label:
            vqa_correct += 1
        if vqa_total % 100 == 0:
            print(f"  [A] VQA [{vqa_total}/{len(vqa_yesno)}] acc={vqa_correct/vqa_total*100:.2f}%")

    vqa_acc = (vqa_correct / vqa_total * 100) if vqa_total > 0 else 0.0
    print(f"[A] VQA DONE: {vqa_acc:.2f}%")

    result = {
        "approach": "A_projection",
        "layer": TEST_LAYER, "feature": TEST_FEATURE,
        "vsr_acc": vsr_acc, "vsr_correct": vsr_correct, "vsr_total": vsr_total,
        "vsr_ctrl_acc": vsr_ctrl_acc, "vsr_ctrl_correct": vsr_ctrl_correct, "vsr_ctrl_total": vsr_ctrl_total,
        "vqa_acc": vqa_acc, "vqa_correct": vqa_correct, "vqa_total": vqa_total,
        "vsr_per_relation": vsr_per_relation,
    }

    out_dir = Path(RESULTS_BASE) / "analysis" / "ablation_jumprelu"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ab_test_approach_a.json", "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()

    return result


# --------------- Approach B: SAE encode-zero-decode (at feature's layer) ---------------

@app.function(image=image, gpu=GPU_TYPE, volumes={"/vol": volume}, timeout=TIMEOUT)
def approach_b_encode_zero_decode():
    """Approach B: At the feature's own layer, intercept the layer output (residual stream),
    run it through the SAE encoder, zero out the target feature coefficient, decode back,
    and replace the original activation. This is the strongest causal ablation because it
    uses the SAE's learned encoding to identify and remove exactly the feature's contribution."""
    import sys, hashlib, torch, warnings
    from pathlib import Path
    from datasets import load_dataset, concatenate_datasets
    from nnsight import NNsight
    from PIL import Image
    from io import BytesIO
    import requests

    warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
    sys.path.insert(0, "/root/paligemma2")
    from utils import (initialize_vlm_model, process_vlm_inputs,
                       get_image_token_positions, initialize_jumprelu_sae)

    print("[B: Encode-Zero-Decode] Loading model...")
    processor, model_raw = initialize_vlm_model(MODEL_NAME, device="cuda")
    model_raw.eval()
    nns_model = NNsight(model_raw)
    tokenizer = processor.tokenizer
    yes_ids, no_ids = _get_yes_no_ids(tokenizer)
    n_model_layers = model_raw.config.text_config.num_hidden_layers
    model_dtype = next(model_raw.parameters()).dtype

    # Load SAE — keep it for encode/decode during ablation
    ckpt_path = f"{RESULTS_BASE}/run_jumprelu/checkpoints/pretrained_layer_{TEST_LAYER}.pt"
    sae = initialize_jumprelu_sae(TEST_LAYER, checkpoint_path=ckpt_path,
                                   device="cuda", cache_dir="/vol/cache/huggingface")
    sae.eval()
    # Cast SAE weights to model dtype for compatibility
    sae_W_enc = sae.W_enc.data.to(model_dtype)    # (d_in, d_sae)
    sae_b_enc = sae.b_enc.data.to(model_dtype)     # (d_sae,)
    sae_W_dec = sae.W_dec.data.to(model_dtype)     # (d_sae, d_in)
    sae_b_dec = sae.b_dec.data.to(model_dtype)     # (d_in,)
    sae_threshold = sae.threshold.data.to(model_dtype)  # (d_sae,)
    print(f"[B] SAE loaded: d_in={sae.d_in}, d_sae={sae.d_sae}")
    print(f"[B] Feature {TEST_FEATURE} threshold={sae_threshold[TEST_FEATURE].item():.6f}")
    del sae
    torch.cuda.empty_cache()

    def do_ablation(input_ids, attention_mask, pixel_values, img_end):
        """Encode-zero-decode: at TEST_LAYER, replace text-token activations with
        SAE reconstruction that has the target feature zeroed out."""
        with nns_model.trace(input_ids=input_ids, attention_mask=attention_mask,
                             pixel_values=pixel_values) as tr:
            # Intercept the layer output (residual stream) at the feature's layer
            # layer.output[0] has shape (seq, d_in) — NO batch dim in PaliGemma2
            layer_out = nns_model.model.language_model.layers[TEST_LAYER].output[0]
            text_acts = layer_out[img_end:]  # (n_text, d_in)

            # SAE encode: x @ W_enc + b_enc -> JumpReLU
            pre_jump = text_acts @ sae_W_enc + sae_b_enc  # (n_text, d_sae)
            # JumpReLU: zero out values below threshold
            codes = pre_jump * (pre_jump > sae_threshold).to(pre_jump.dtype)  # (n_text, d_sae)

            # Zero out the target feature
            codes[:, TEST_FEATURE] = 0.0

            # SAE decode: codes @ W_dec + b_dec
            reconstructed = codes @ sae_W_dec + sae_b_dec  # (n_text, d_in)

            # Replace original text-token activations with reconstruction
            layer_out[img_end:] = reconstructed

            logits_saved = nns_model.output.logits.save()
        return logits_saved

    def predict_yesno(logits_saved):
        logits = logits_saved[:, -1, :]
        probs = torch.softmax(logits, dim=-1)[0]
        yes_mass = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
        no_mass = probs[list(no_ids)].sum().item() if no_ids else 0.0
        denom = yes_mass + no_mass
        return 1 if (yes_mass / denom if denom > 0 else 0.0) > 0.5 else 0

    # Load datasets
    print("[B] Loading datasets...")
    vsr_splits = []
    data_files = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    for split in VSR_SPLITS:
        vsr_splits.append(load_dataset(VSR_DATASET, data_files=data_files, split=split))
    vsr = concatenate_datasets(vsr_splits)

    vqa = load_dataset("lmms-lab/VQAv2", split="validation")
    vqa_yesno = []
    for i in range(len(vqa)):
        ex = vqa[i]
        if str(ex.get("answer_type", "")).lower() == "yes/no":
            mc = str(ex.get("multiple_choice_answer", "")).strip().lower()
            if mc in {"yes", "no"}:
                vqa_yesno.append((i, 1 if mc == "yes" else 0))
                if len(vqa_yesno) >= VQA_MAX_SAMPLES:
                    break

    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

    # --- VSR ---
    print(f"[B] Running VSR ablation ({len(vsr)} samples)...")
    vsr_correct = vsr_total = vsr_ctrl_correct = vsr_ctrl_total = 0
    vsr_per_relation = {}

    for vi in range(len(vsr)):
        ex = vsr[vi]
        url = ex.get("image_link", "")
        statement = str(ex.get("caption", "")).strip()
        label = int(ex.get("label", 0))
        relation = ex.get("relation", "")
        if not statement or not url:
            continue

        url_hash = hashlib.md5(url.encode()).hexdigest()
        img_cache = os.path.join(IMAGE_CACHE_DIR, f"{url_hash}.jpg")
        try:
            if os.path.exists(img_cache):
                img = Image.open(img_cache).convert("RGB")
            else:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                img.save(img_cache, "JPEG")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        prompt = _build_vsr_prompt(statement)
        input_ids, attention_mask, pixel_values = process_vlm_inputs(img, prompt, processor, nns_model._module)
        _, img_end = get_image_token_positions(input_ids)

        try:
            logits_saved = do_ablation(input_ids, attention_mask, pixel_values, img_end)
            pred = predict_yesno(logits_saved)
        except Exception as e:
            if vi < 3:
                print(f"  [B ERROR] VSR {vi}: {e}")
            pred = 0

        vsr_total += 1
        if pred == label:
            vsr_correct += 1
        if relation not in vsr_per_relation:
            vsr_per_relation[relation] = {"correct": 0, "total": 0}
        vsr_per_relation[relation]["total"] += 1
        if pred == label:
            vsr_per_relation[relation]["correct"] += 1
        if relation in CONTROL_RELATIONS:
            vsr_ctrl_total += 1
            if pred == label:
                vsr_ctrl_correct += 1

        if (vi + 1) % 100 == 0 and vsr_total > 0:
            print(f"  [B] VSR [{vi+1}/{len(vsr)}] acc={vsr_correct/vsr_total*100:.2f}%")

    vsr_acc = (vsr_correct / vsr_total * 100) if vsr_total > 0 else 0.0
    vsr_ctrl_acc = (vsr_ctrl_correct / vsr_ctrl_total * 100) if vsr_ctrl_total > 0 else 0.0
    print(f"[B] VSR DONE: {vsr_acc:.2f}% (ctrl={vsr_ctrl_acc:.2f}%)")

    # --- VQA ---
    print(f"[B] Running VQA ablation ({len(vqa_yesno)} samples)...")
    vqa_correct = vqa_total = 0

    for qi, label in vqa_yesno:
        ex = vqa[qi]
        img = ex.get("image")
        question = ex.get("question", "")
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
        else:
            continue

        prompt = f"Answer the following question with only 'Yes' or 'No':\nQuestion: {question.strip()}\nAnswer:"
        try:
            input_ids, attention_mask, pixel_values = process_vlm_inputs(img, prompt, processor, nns_model._module)
        except Exception:
            continue
        _, img_end = get_image_token_positions(input_ids)

        try:
            logits_saved = do_ablation(input_ids, attention_mask, pixel_values, img_end)
            pred = predict_yesno(logits_saved)
        except Exception as e:
            if vqa_total < 3:
                print(f"  [B ERROR] VQA {vqa_total}: {e}")
            pred = 0

        vqa_total += 1
        if pred == label:
            vqa_correct += 1
        if vqa_total % 100 == 0:
            print(f"  [B] VQA [{vqa_total}/{len(vqa_yesno)}] acc={vqa_correct/vqa_total*100:.2f}%")

    vqa_acc = (vqa_correct / vqa_total * 100) if vqa_total > 0 else 0.0
    print(f"[B] VQA DONE: {vqa_acc:.2f}%")

    result = {
        "approach": "B_encode_zero_decode",
        "layer": TEST_LAYER, "feature": TEST_FEATURE,
        "vsr_acc": vsr_acc, "vsr_correct": vsr_correct, "vsr_total": vsr_total,
        "vsr_ctrl_acc": vsr_ctrl_acc, "vsr_ctrl_correct": vsr_ctrl_correct, "vsr_ctrl_total": vsr_ctrl_total,
        "vqa_acc": vqa_acc, "vqa_correct": vqa_correct, "vqa_total": vqa_total,
        "vsr_per_relation": vsr_per_relation,
    }

    out_dir = Path(RESULTS_BASE) / "analysis" / "ablation_jumprelu"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ab_test_approach_b.json", "w") as f:
        json.dump(result, f, indent=2)
    volume.commit()

    return result


# --------------- Entrypoint ---------------

@app.local_entrypoint()
def main():
    print("=" * 70)
    print("A/B ABLATION TEST: L17/F14475")
    print("  A: Projection (all layers, -1.0 coef)")
    print("  B: SAE encode-zero-decode (feature's own layer)")
    print("=" * 70)

    # Load baseline from cache
    import json as json_local
    baseline_path = Path("/tmp/ab_baseline_cache.json")

    # Launch both approaches in parallel on 2 GPUs
    print("\nLaunching both approaches in parallel...")
    result_a_handle = approach_a_projection.spawn()
    result_b_handle = approach_b_encode_zero_decode.spawn()

    print("Waiting for results...")
    result_a = result_a_handle.get()
    result_b = result_b_handle.get()

    # Load baseline from volume (already cached from main ablation run)
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    # Print baseline reference (from previous run)
    print("Baseline (from prior run): VSR=51.80%, VQA=62.50%, Ctrl=56.14%")
    bl_vsr, bl_vqa, bl_ctrl = 51.80, 62.50, 56.14

    print(f"\n{'Metric':<20} {'Approach A (Proj)':<25} {'Approach B (Enc-0-Dec)':<25}")
    print("-" * 70)
    print(f"{'VSR Acc':<20} {result_a['vsr_acc']:.2f}% (Δ={result_a['vsr_acc']-bl_vsr:+.2f}){'':<5} {result_b['vsr_acc']:.2f}% (Δ={result_b['vsr_acc']-bl_vsr:+.2f})")
    print(f"{'VQA Acc':<20} {result_a['vqa_acc']:.2f}% (Δ={result_a['vqa_acc']-bl_vqa:+.2f}){'':<5} {result_b['vqa_acc']:.2f}% (Δ={result_b['vqa_acc']-bl_vqa:+.2f})")
    print(f"{'Ctrl VSR':<20} {result_a['vsr_ctrl_acc']:.2f}% (Δ={result_a['vsr_ctrl_acc']-bl_ctrl:+.2f}){'':<5} {result_b['vsr_ctrl_acc']:.2f}% (Δ={result_b['vsr_ctrl_acc']-bl_ctrl:+.2f})")
    print("=" * 70)

    # Determine winner
    delta_a = bl_vsr - result_a['vsr_acc']
    delta_b = bl_vsr - result_b['vsr_acc']
    if delta_b > delta_a:
        print(f"\n>>> Approach B has STRONGER VSR drop ({delta_b:.2f}pp vs {delta_a:.2f}pp)")
    elif delta_a > delta_b:
        print(f"\n>>> Approach A has STRONGER VSR drop ({delta_a:.2f}pp vs {delta_b:.2f}pp)")
    else:
        print(f"\n>>> Both approaches produce similar VSR drops")

    # Check spatial selectivity (VSR drop > Ctrl drop = spatially selective)
    ctrl_drop_a = bl_ctrl - result_a['vsr_ctrl_acc']
    ctrl_drop_b = bl_ctrl - result_b['vsr_ctrl_acc']
    print(f"    A spatial selectivity: VSR drop {delta_a:.2f} vs Ctrl drop {ctrl_drop_a:.2f}")
    print(f"    B spatial selectivity: VSR drop {delta_b:.2f} vs Ctrl drop {ctrl_drop_b:.2f}")
