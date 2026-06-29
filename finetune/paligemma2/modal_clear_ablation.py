"""Clear stale ablation results from volume."""
import modal

VOLUME_NAME = "vlm-scope-data-v2"
app = modal.App("clear-ablation")
volume = modal.Volume.from_name(VOLUME_NAME)

@app.function(volumes={"/vol": volume}, timeout=60)
def clear():
    import shutil
    from pathlib import Path
    d = Path("/vol/results/paligemma2/analysis/ablation_jumprelu")
    if d.exists():
        shutil.rmtree(d)
        print(f"Removed {d}")
    d.mkdir(parents=True, exist_ok=True)
    volume.commit()
    return "Cleared"

@app.local_entrypoint()
def main():
    print(clear.remote())
