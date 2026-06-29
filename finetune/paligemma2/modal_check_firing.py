"""Check firing results on volume."""
import modal

app = modal.App("check-firing")
volume = modal.Volume.from_name("vlm-scope-data-v2")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")

@app.function(image=image, volumes={"/vol": volume}, timeout=60)
def check():
    import json
    import numpy as np
    from pathlib import Path

    firing_dir = Path("/vol/results/paligemma2/analysis/firing_vsr_jumprelu")
    if not firing_dir.exists():
        print("firing_vsr_jumprelu/ NOT FOUND")
        return

    files = sorted(firing_dir.glob("*.json"))
    print(f"Firing result files: {len(files)}")

    layers_done = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        layer = f.name.replace("firing_vsr_layer_", "").replace(".json", "")
        n_vqa = data.get("n_vqa", 0)
        n_vsr = data.get("n_vsr", 0)
        fire_vqa = np.array(data.get("fire_count_vqa", []))
        fire_vsr = np.array(data.get("fire_count_vsr", []))
        n_vqa_firing = np.count_nonzero(fire_vqa) if len(fire_vqa) > 0 else 0
        n_vsr_firing = np.count_nonzero(fire_vsr) if len(fire_vsr) > 0 else 0

        print(f"  L{layer}: n_vqa={n_vqa}, n_vsr={n_vsr}, "
              f"vqa_firing={n_vqa_firing}, vsr_firing={n_vsr_firing}")
        if n_vqa > 0 and n_vsr > 0:
            layers_done.append(int(layer))

    print(f"\nLayers with data: {sorted(layers_done)} ({len(layers_done)} total)")
    missing = set(range(26)) - set(layers_done)
    if missing:
        print(f"Missing layers: {sorted(missing)}")

@app.local_entrypoint()
def main():
    check.remote()
