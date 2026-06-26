"""Aggregate every out/<model>/<video>/meta.json into report tables (env: any)."""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "out")
RES_DIR = os.path.join(HERE, "results")
os.makedirs(RES_DIR, exist_ok=True)
TAGS = ["fast", "static"]


def load_all():
    rows = {}
    for mp in sorted(glob.glob(f"{ROOT}/*/*/meta.json")):
        meta = json.load(open(mp))
        sub = mp.split("/")[-3]; tag = mp.split("/")[-2]
        rows.setdefault(sub, {})[tag] = meta
    return rows


def fmt(x, n=4):
    return f"{x:.{n}g}"


def main():
    rows = load_all()
    if not rows:
        print("no results found under", ROOT); return
    out = ["# Vision-encoder overhead & temporal-redundancy analysis\n",
           f"models: {len(rows)}  | videos: fast (downhill MTB POV) vs static (locked forest)\n"]

    # ---------- Table 1: model / encoder spec ----------
    out += ["\n## 1. Model & vision-encoder spec\n",
            "| model | type | total params | vision-enc params | dtype | native res | proc res | patch grid | #tokens | patch dim |",
            "|---|---|--:|--:|---|---|--:|:--:|--:|--:|"]
    for sub, d in rows.items():
        m = d.get("fast") or next(iter(d.values()))
        name = m["delta_summary"]["model"]
        out.append(f"| {name} | {m['model_type']} | {m['total_params']/1e6:.0f}M | "
                   f"{m['vision_encoder_params']/1e6:.0f}M | {m['dtype']} | {m['native_res']} | "
                   f"{m['processing_res']} | {m['patch_grid'][0]}x{m['patch_grid'][1]} | "
                   f"{m['n_vision_tokens']} | {m['patch_dim']} |")

    # ---------- Table 2: encoder timing ----------
    out += ["\n## 2. Vision-encoder latency (per frame, single image, GPU)\n",
            "| model | fast ms/frame | fast fps | static ms/frame | static fps |",
            "|---|--:|--:|--:|--:|"]
    for sub, d in rows.items():
        name = (d.get("fast") or next(iter(d.values())))["delta_summary"]["model"]
        c = []
        for tag in TAGS:
            t = d.get(tag, {}).get("timing")
            c += ([f"{t['per_frame_ms_mean']:.2f}±{t['per_frame_ms_std']:.2f}", f"{t['fps']:.1f}"]
                  if t else ["-", "-"])
        out.append(f"| {name} | {c[0]} | {c[1]} | {c[2]} | {c[3]} |")

    # ---------- Table 3 (headline): Temporal Redundancy Score ----------
    out += ["\n## 3. Temporal Redundancy Score (headline)\n",
            "**TRS = 1 - mean(embedding cosine-distance) over non-cut adjacent frame pairs.**\n",
            "TRS -> 1 means consecutive frames produce nearly identical encoder representations "
            "(high temporal redundancy => strong case for reuse/caching). cos is scale-free, so "
            "TRS is comparable across models. This is the bottom-line 'how much redundancy is there?'.\n",
            "| model | type | TRS (static) | TRS (fast) | drop (static-fast) |",
            "|---|---|--:|--:|--:|"]
    for sub, d in rows.items():
        m0 = d.get("fast") or next(iter(d.values()))
        name = m0["delta_summary"]["model"]; typ = m0["model_type"]
        trs = {}
        for tag in TAGS:
            if tag in d:
                trs[tag] = 1.0 - d[tag]["delta_summary"]["averages_excluding_cuts"]["cos"]["embedding"]
        ts = fmt(trs["static"]) if "static" in trs else "-"
        tf = fmt(trs["fast"]) if "fast" in trs else "-"
        drop = fmt(trs["static"] - trs["fast"]) if ("static" in trs and "fast" in trs) else "-"
        out.append(f"| {name} | {typ} | {ts} | {tf} | {drop} |")

    # ---------- Table 4a: analysis-1 means (pixel vs embedding delta, cuts excluded) ----------
    out += ["\n## 4. Analysis 1 — does pixel change predict embedding change?\n",
            "Purpose: measure how strongly an input-frame change is *transmitted* into the "
            "encoder representation. High correlation = encoder is sensitive to input change; "
            "low = encoder absorbs/ignores part of the change in its representation space.\n",
            "**Cross-model comparison uses cosine-distance only: L1/L2 depend on each model's "
            "feature scale (e.g. Qwen emb-L2 ~1500 vs SAM2 ~0.5) and are NOT comparable.**\n",
            "### 4a. Mean adjacent-frame delta (cuts excluded)\n",
            "| model | video | pix cos | emb cos | pix L2 | emb L2 (own scale) | #cuts |",
            "|---|---|--:|--:|--:|--:|--:|"]
    for sub, d in rows.items():
        for tag in TAGS:
            if tag not in d: continue
            s = d[tag]["delta_summary"]; a = s["averages_excluding_cuts"]
            out.append(f"| {s['model']} | {tag} | {fmt(a['cos']['pixel'])} | {fmt(a['cos']['embedding'])} "
                       f"| {fmt(a['l2']['pixel'])} | {fmt(a['l2']['embedding'])} | {s['num_cuts']} |")

    # ---------- Table 4b: analysis-1 final statistic (correlation, MATCHED metric) ----------
    out += ["\n### 4b. Final statistic — correlation of per-frame **pixel-cos vs embedding-cos** delta\n",
            "**Both axes use the SAME distance metric (cosine-distance)** so x and y are "
            "consistent — no L2-vs-cos labeling mismatch. With matched units the OLS slope is "
            "interpretable as a *sensitivity gain* (d embedding-cos / d pixel-cos). "
            "Pearson r and Spearman rho are the core statistics. (pixL2-vs-embL2 also stored in meta.json.)\n",
            "| model | video | Pearson r | Spearman rho | R^2 | gain (slope) |",
            "|---|---|--:|--:|--:|--:|"]
    for sub, d in rows.items():
        for tag in TAGS:
            if tag not in d: continue
            s = d[tag]["delta_summary"]
            c = s.get("analysis1_correlation", {}).get("pixCOS_vs_embCOS")
            if not c:
                out.append(f"| {s['model']} | {tag} | - | - | - | - |"); continue
            out.append(f"| {s['model']} | {tag} | {fmt(c['pearson_r'])} | {fmt(c['spearman_rho'])} "
                       f"| {fmt(c['r_squared'])} | {fmt(c['ols_slope'])} |")

    # ---------- Table 5: analysis-2 (motion-region concentration) ----------
    out += ["\n## 5. Analysis 2 — is embedding change concentrated in the motion region?\n",
            "Motion patches = pixel-L2 delta above the p90 (top-10%) threshold. "
            "Note: this is a **pixel-difference motion proxy**, not optical flow — global "
            "brightness/exposure shifts can also raise pixel-L2 (acceptable here as fast/static "
            "differ mainly in real motion).\n",
            "**Primary metric = Concentration Factor (CF) = R / area.** "
            "R alone grows with the number of motion patches, so it is normalised by area: "
            "CF > 1 means motion patches carry MORE embedding change than their area share "
            "(=> static patches are redundant, cacheable). CF ~ 1 means change is spread uniformly.\n",
            "(cos used for cross-model comparability; area = motion-patch fraction.)\n",
            "| model | video | area | R(cos) | **CF(cos)** | CF(L2) |",
            "|---|---|--:|--:|--:|--:|"]
    for sub, d in rows.items():
        for tag in TAGS:
            if tag not in d: continue
            s = d[tag]["delta_summary"]; mc = s.get("motion_concentration", {}).get("p90")
            if not mc:
                out.append(f"| {s['model']} | {tag} | - | - | - | - |"); continue
            out.append(f"| {s['model']} | {tag} | {fmt(mc['area_fraction'])} | {fmt(mc['R']['cos'])} "
                       f"| **{fmt(mc['R_over_area']['cos'])}** | {fmt(mc['R_over_area']['l2'])} |")

    report = "\n".join(out) + "\n"
    with open(f"{RES_DIR}/REPORT.md", "w") as f:
        f.write(report)
    print(report)
    print("written ->", f"{RES_DIR}/REPORT.md")


if __name__ == "__main__":
    main()
