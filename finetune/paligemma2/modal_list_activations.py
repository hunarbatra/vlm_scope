"""List activation files on the volume."""
import modal

app = modal.App("list-activations")
volume = modal.Volume.from_name("vlm-scope-data-v2")

@app.function(volumes={"/vol": volume}, timeout=60)
def list_acts():
    from pathlib import Path
    import json

    base = Path("/vol/results/paligemma2")

    # Check activation dirs
    for name in ["run/activations", "run_jumprelu/activations", "activations"]:
        d = base / name
        if d.exists():
            files = sorted(d.glob("*.h5"))
            print(f"\n{d}: {len(files)} h5 files")
            for f in files[:10]:
                print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
        else:
            print(f"\n{d}: NOT FOUND")

    # Check top_samples
    ts = base / "analysis" / "autointerp" / "top_samples.json"
    if ts.exists():
        data = json.loads(ts.read_text())
        print(f"\ntop_samples.json: {len(data)} features")
        for k in sorted(data.keys())[:5]:
            print(f"  {k}: {len(data[k])} samples")
    else:
        print(f"\n{ts}: NOT FOUND")

    # Check what feature CSV has
    csv = base / "analysis" / "final_features_jumprelu" / "final_spatial_visual_features.csv"
    if csv.exists():
        import pandas as pd
        df = pd.read_csv(csv)
        print(f"\nFeature CSV: {len(df)} features")
        print(f"  Layers: {sorted(df['layer'].unique())}")
    else:
        print(f"\n{csv}: NOT FOUND")

    # Check analysis dir structure
    analysis = base / "analysis"
    if analysis.exists():
        for d in sorted(analysis.iterdir()):
            if d.is_dir():
                n = len(list(d.iterdir()))
                print(f"  {d.name}/: {n} items")

@app.local_entrypoint()
def main():
    list_acts.remote()
