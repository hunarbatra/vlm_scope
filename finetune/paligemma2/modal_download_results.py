"""Download analysis results from Modal volume to local disk."""
import modal

app = modal.App("download-results")
volume = modal.Volume.from_name("vlm-scope-data-v2")
image = modal.Image.debian_slim(python_version="3.11")

@app.function(image=image, volumes={"/vol": volume}, timeout=120)
def get_results():
    """Read and return all analysis CSVs and JSONs needed for local work."""
    from pathlib import Path
    import json

    base = Path("/vol/results/paligemma2/analysis")
    results = {}

    # 1. Spatial features (final = adapted ∩ spatial)
    f = base / "final_features_vsr_jumprelu" / "final_spatial_visual_features.csv"
    if f.exists():
        results["final_spatial_visual_features.csv"] = f.read_text()

    # 2. All spatial features (before adapted intersection)
    f = base / "spatial_vsr_jumprelu" / "spatial_features_vsr_all.csv"
    if f.exists():
        results["spatial_features_vsr_all.csv"] = f.read_text()

    # 3. Spatial summary
    f = base / "spatial_vsr_jumprelu" / "spatial_vsr_summary.json"
    if f.exists():
        results["spatial_vsr_summary.json"] = f.read_text()

    # 4. Adapted features
    f = base / "adapted_jumprelu" / "adapted_features_results.csv"
    if f.exists():
        results["adapted_features_results.csv"] = f.read_text()

    # 5. VQA firing stats (for reference / local analysis)
    firing_dir = base / "firing_jumprelu"
    if firing_dir.exists():
        for fp in sorted(firing_dir.glob("firing_layer_*.json")):
            results[f"firing_jumprelu/{fp.name}"] = fp.read_text()

    # 6. VSR firing stats
    vsr_dir = base / "firing_vsr_jumprelu"
    if vsr_dir.exists():
        for fp in sorted(vsr_dir.glob("*.json")):
            results[f"firing_vsr_jumprelu/{fp.name}"] = fp.read_text()

    # 7. Cosines
    cos_dir = base / "cosines_jumprelu"
    if cos_dir.exists():
        for fp in cos_dir.glob("*.npy"):
            results[f"cosines_jumprelu/{fp.name}"] = "__NPY__"  # flag for binary

    # 8. FVU table
    f = base / "fvu_table.csv"
    if f.exists():
        results["fvu_table.csv"] = f.read_text()

    print(f"Found {len(results)} files:")
    for k, v in results.items():
        if v == "__NPY__":
            print(f"  {k}: (numpy binary)")
        else:
            print(f"  {k}: {len(v)} bytes")

    return results


@app.local_entrypoint()
def main():
    from pathlib import Path
    import json

    results = get_results.remote()

    # Save to local analysis directory
    local_base = Path("/home/hbatra/vlm_scope_backup/vlm_scope/finetune/paligemma2/analysis_results")
    local_base.mkdir(parents=True, exist_ok=True)

    for name, content in results.items():
        if content == "__NPY__":
            continue  # Skip binary files
        out_path = local_base / name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        print(f"  Saved: {out_path}")

    print(f"\nDownloaded {len([v for v in results.values() if v != '__NPY__'])} files to {local_base}")
