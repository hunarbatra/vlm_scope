"""Upload local analysis results to Modal volume."""
import modal
from pathlib import Path

app = modal.App("upload-results")
volume = modal.Volume.from_name("vlm-scope-data-v2")
image = modal.Image.debian_slim(python_version="3.11")

@app.function(image=image, volumes={"/vol": volume}, timeout=120)
def upload(files: dict):
    """Upload files to Modal volume."""
    base = Path("/vol/results/paligemma2/analysis")
    for path, content in files.items():
        out = base / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content)
        print(f"  Uploaded: {out} ({len(content)} bytes)")
    volume.commit()

@app.local_entrypoint()
def main():
    local_dir = Path("analysis_results/lexical_filter")
    files = {}
    for f in local_dir.glob("*.csv"):
        files[f"lexical_filter_jumprelu/{f.name}"] = f.read_text()
    print(f"Uploading {len(files)} files...")
    upload.remote(files)
    print("Done!")
