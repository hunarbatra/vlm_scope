"""Check h5 activation file structure."""
import modal

app = modal.App("check-h5")
volume = modal.Volume.from_name("vlm-scope-data-v2")
image = modal.Image.debian_slim(python_version="3.11").pip_install("h5py", "numpy")

@app.function(image=image, volumes={"/vol": volume}, timeout=60)
def check():
    import h5py
    path = "/vol/results/paligemma2/run/activations/chunk_0_1000.h5"
    with h5py.File(path, "r") as hf:
        print(f"Top-level keys (first 10): {list(hf.keys())[:10]}")
        # Check first key
        first_key = list(hf.keys())[0]
        print(f"\nFirst key: {first_key}")
        item = hf[first_key]
        if hasattr(item, 'keys'):
            print(f"  Sub-keys: {list(item.keys())[:10]}")
            first_sub = list(item.keys())[0]
            print(f"  First sub: {first_sub}, shape={item[first_sub].shape}, dtype={item[first_sub].dtype}")
        else:
            print(f"  Shape: {item.shape}, dtype: {item.dtype}")

@app.local_entrypoint()
def main():
    check.remote()
