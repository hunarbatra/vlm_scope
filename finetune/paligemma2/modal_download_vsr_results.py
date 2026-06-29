"""Download spatial VSR results from Modal volume."""
import modal

app = modal.App("download-vsr-results")
volume = modal.Volume.from_name("vlm-scope-data-v2")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas")

@app.function(image=image, volumes={"/vol": volume}, timeout=120)
def download():
    from pathlib import Path
    import pandas as pd

    base = Path("/vol/results/paligemma2/analysis")
    result = {}

    # List what's in analysis dir
    if base.exists():
        print("Analysis subdirs:", [d.name for d in base.iterdir() if d.is_dir()])

    # Spatial VSR features (Fisher test results)
    spatial_csv = base / "spatial_vsr_jumprelu" / "spatial_features_vsr.csv"
    if spatial_csv.exists():
        df = pd.read_csv(spatial_csv)
        print(f"spatial_features_vsr.csv: {len(df)} features")
        print(f"  Columns: {list(df.columns)}")
        per_layer = df.groupby('layer').size()
        print(f"  Per layer:\n{per_layer}")
        result["spatial"] = df.to_csv(index=False)
    else:
        print("spatial_features_vsr.csv NOT FOUND")

    # Final intersection (adapted ∩ spatial_vsr)
    final_csv = base / "final_features_vsr_jumprelu" / "final_spatial_visual_features.csv"
    if final_csv.exists():
        df2 = pd.read_csv(final_csv)
        print(f"\nfinal_spatial_visual_features.csv: {len(df2)} features")
        print(f"  Columns: {list(df2.columns)}")
        per_layer = df2.groupby('layer').size()
        print(f"  Per layer:\n{per_layer}")
        result["final"] = df2.to_csv(index=False)
    else:
        print("final_spatial_visual_features.csv NOT FOUND")

    return result

@app.local_entrypoint()
def main():
    result = download.remote()
    from pathlib import Path
    out_dir = Path("analysis_results")
    out_dir.mkdir(exist_ok=True)

    if "spatial" in result:
        with open(out_dir / "spatial_features_vsr.csv", "w") as f:
            f.write(result["spatial"])
        print(f"Saved spatial_features_vsr.csv")

    if "final" in result:
        with open(out_dir / "final_spatial_visual_features_vsr.csv", "w") as f:
            f.write(result["final"])
        print(f"Saved final_spatial_visual_features_vsr.csv")
