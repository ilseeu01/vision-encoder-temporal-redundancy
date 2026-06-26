"""Recompute analysis from cached frames.npz (no GPU) and patch meta.json's delta_summary.
Used after changing analysis logic (e.g. matched-metric correlation) to avoid re-encoding."""
import os, sys, json, glob
sys.path.insert(0, "/home/hjlee/video_delta_analysis"); import common
for npz in sorted(glob.glob("out/*/*/frames.npz")):
    d = os.path.dirname(npz); meta_p = os.path.join(d, "meta.json")
    meta = json.load(open(meta_p))
    name = meta["delta_summary"]["model"]
    summary = common.run_analysis(npz, d, name)      # rewrites delta_summary.json + plots
    meta["delta_summary"] = summary
    json.dump(meta, open(meta_p, "w"), indent=2)
print("reanalyze done")
