"""Download top_samples.json from Modal volume to local."""
import modal

app = modal.App("download-top-samples")
volume = modal.Volume.from_name("vlm-scope-data-v2")

@app.function(volumes={"/vol": volume}, timeout=60)
def download():
    import json
    from pathlib import Path
    p = Path("/vol/results/paligemma2/analysis/autointerp/top_samples.json")
    if not p.exists():
        return {"error": "not found"}
    return json.loads(p.read_text())

@app.local_entrypoint()
def main():
    import json
    data = download.remote()
    if "error" in data:
        print(f"Error: {data['error']}")
        return
    out = "top_samples.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    n = sum(1 for v in data.values() if v)
    print(f"Downloaded {len(data)} features ({n} with samples) to {out}")
