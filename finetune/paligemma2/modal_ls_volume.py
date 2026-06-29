"""Quick script to list what's on the volume."""
import modal

app = modal.App("ls-volume")
volume = modal.Volume.from_name("vlm-scope-data-v2")

@app.function(volumes={"/vol": volume}, timeout=60)
def ls_vol():
    import os
    from pathlib import Path

    base = Path("/vol/results/paligemma2")
    print("=== Top-level ===")
    for p in sorted(base.iterdir()):
        print(f"  {p.name}/")

    # Check run directory
    run_dir = base / "run"
    if run_dir.exists():
        print(f"\n=== run/ ===")
        for p in sorted(run_dir.iterdir()):
            if p.is_dir():
                n = len(list(p.iterdir()))
                print(f"  {p.name}/ ({n} items)")
            else:
                print(f"  {p.name} ({p.stat().st_size})")

    # Check for jumprelu run
    jr_dir = base / "run_jumprelu"
    if jr_dir.exists():
        print(f"\n=== run_jumprelu/ ===")
        for p in sorted(jr_dir.iterdir()):
            if p.is_dir():
                n = len(list(p.iterdir()))
                print(f"  {p.name}/ ({n} items)")
            else:
                print(f"  {p.name}")

    # Check activations directories
    for act_name in ["activations", "activations_vsr", "activations_vqa"]:
        for parent in [base / "run", base / "run_jumprelu", base]:
            act_dir = parent / act_name
            if act_dir.exists():
                files = sorted(act_dir.glob("*.h5"))
                print(f"\n=== {act_dir.relative_to(base)} === ({len(files)} h5 files)")
                for f in files[:5]:
                    print(f"  {f.name} ({f.stat().st_size // 1024}KB)")
                if len(files) > 5:
                    print(f"  ... and {len(files) - 5} more")

    # Check analysis directory
    analysis = base / "analysis"
    if analysis.exists():
        print(f"\n=== analysis/ ===")
        for p in sorted(analysis.iterdir()):
            if p.is_dir():
                n = len(list(p.iterdir()))
                print(f"  {p.name}/ ({n} items)")
            else:
                print(f"  {p.name}")

@app.local_entrypoint()
def main():
    ls_vol.remote()
