"""Check VSR activation files and firing results on volume."""
import modal

app = modal.App("ls-vsr")
volume = modal.Volume.from_name("vlm-scope-data-v2")
image = modal.Image.debian_slim(python_version="3.11").pip_install("h5py", "numpy")

@app.function(image=image, volumes={"/vol": volume}, timeout=300)
def ls_vsr():
    from pathlib import Path
    import json
    import h5py

    base = Path("/vol/results/paligemma2")

    # 1. Count total samples across all H5 chunks
    vsr_dir = base / "run" / "activations_vsr"
    print("=== VSR H5 Sample Counts (layer_0) ===")
    total_samples = 0
    if vsr_dir.exists():
        files = sorted(vsr_dir.glob("*.h5"))
        for f in files:
            try:
                with h5py.File(str(f), "r") as hf:
                    grp = hf.get("layer_0")
                    if grp:
                        n = len([k for k in grp.keys() if k.startswith("sample_")])
                        total_samples += n
                        print(f"  {f.name}: {n} samples, {f.stat().st_size/1024/1024:.0f} MB")
                    else:
                        print(f"  {f.name}: NO layer_0")
            except Exception as e:
                print(f"  {f.name}: ERROR {e}")
        print(f"  TOTAL: {total_samples} samples across {len(files)} chunks")

    # 2. Check VSR firing JSON format
    print("\n=== VSR Firing JSONs (first file full dump) ===")
    firing_dir = base / "analysis" / "firing_vsr_jumprelu"
    if firing_dir.exists():
        jsons = sorted(firing_dir.glob("*.json"))
        if jsons:
            with open(jsons[0]) as fp:
                d = json.load(fp)
            # Print all keys and non-list values
            for k, v in d.items():
                if isinstance(v, list):
                    print(f"  {k}: list[{len(v)}], sum={sum(v)}, nonzero={sum(1 for x in v if x != 0)}")
                else:
                    print(f"  {k}: {v}")

@app.local_entrypoint()
def main():
    ls_vsr.remote()
