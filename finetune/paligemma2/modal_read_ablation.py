"""Read ablation results — show summary table."""
import modal

VOLUME_NAME = "vlm-scope-data-v2"
app = modal.App("read-ablation-results")
volume = modal.Volume.from_name(VOLUME_NAME)

@app.function(volumes={"/vol": volume}, timeout=60)
def read_results():
    import json
    from pathlib import Path

    results_dir = Path("/vol/results/paligemma2/analysis/ablation_jumprelu")
    if not results_dir.exists():
        return "No ablation_jumprelu directory found"

    output = []

    # Baseline
    bl = results_dir / "baseline_cache.json"
    if bl.exists():
        b = json.loads(bl.read_text())
        output.append(f"BASELINE: VSR={b.get('vsr_acc',0):.2f}% ({b.get('vsr_correct',0)}/{b.get('vsr_total',0)}), "
                      f"VQA={b.get('vqa_acc',0):.2f}% ({b.get('vqa_correct',0)}/{b.get('vqa_total',0)}), "
                      f"Ctrl={b.get('vsr_ctrl_acc',0):.2f}%")
        bl_vsr = b.get('vsr_acc', 0)
        bl_vqa = b.get('vqa_acc', 0)
        bl_ctrl = b.get('vsr_ctrl_acc', 0)
    else:
        output.append("No baseline cache")
        bl_vsr = bl_vqa = bl_ctrl = 0

    output.append(f"\n{'Layer':>5} {'Feat':>6} {'VSR%':>7} {'ΔVSR':>7} {'VQA%':>7} {'ΔVQA':>7} {'Ctrl%':>7} {'ΔCtrl':>7}")
    output.append("-" * 65)

    all_features = []
    for wf in sorted(results_dir.glob("ablation_w*.json")):
        data = json.loads(wf.read_text())
        for feat in data:
            all_features.append(feat)

    # Sort by layer, then feature
    all_features.sort(key=lambda x: (x.get("layer", 0), x.get("feature", 0)))

    for feat in all_features:
        lid = feat.get("layer", "?")
        fid = feat.get("feature", "?")
        vsr = feat.get("vsr_acc", 0)
        vqa = feat.get("vqa_acc", 0)
        ctrl = feat.get("vsr_ctrl_acc", 0)
        dvsr = vsr - bl_vsr
        dvqa = vqa - bl_vqa
        dctrl = ctrl - bl_ctrl
        output.append(f"  L{lid:>2}  F{fid:>5}  {vsr:>6.2f}  {dvsr:>+6.2f}  {vqa:>6.2f}  {dvqa:>+6.2f}  {ctrl:>6.2f}  {dctrl:>+6.2f}")

    output.append(f"\nTotal features completed: {len(all_features)}")
    return "\n".join(output)

@app.local_entrypoint()
def main():
    print(read_results.remote())
