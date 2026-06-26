"""
Shared utilities for per-frame pixel / vision-encoder-embedding delta analysis.

Pipeline (same for Qwen3-VL and InternVL3):
  1. Decode ALL original frames, resize each to a fixed square (FIXED_RES) so that
     patch grids correspond 1:1 across frames.
  2. For every consecutive pair (t-1, t) compute:
       - pixel deltas   (on normalized [0,1] RGB)   : L1, L2, cosine-distance
       - embedding deltas (vision-encoder pre-projector patch tokens): L1, L2, cosine-distance
     Each delta is computed PER PATCH -> a (gridH, gridW) heatmap, plus a scalar = mean over patches.
  3. Detect scene cuts from the pixel-L2 scalar series (median + k*MAD).
  4. Average every metric over NON-cut frames.
  5. Plot time-series (shared y-max within model) and composite heatmap pages.

This module is model-agnostic; the two extract_*.py scripts produce the per-frame
embedding tensors and call run_analysis().
"""
import os, json
import numpy as np

FIXED_RES = 448          # every frame resized to FIXED_RES x FIXED_RES
SCENE_CUT_K = 6.0        # MAD multiplier for scene-cut threshold
SCENE_CUT_MIN_REL = 0.0  # extra absolute guard (unused by default)
MOTION_PCTLS = (80, 90, 95)   # analysis-2: pixel-L2 percentile thresholds for "motion" patches
THUMB = 160              # thumbnail edge stored in npz


# ----------------------------------------------------------------------------- metrics
def _block_reduce_mean(arr, gh, gw):
    """Average-pool a (H,W) array into a (gh,gw) grid (H,W divisible by gh,gw)."""
    H, W = arr.shape
    bh, bw = H // gh, W // gw
    arr = arr[: bh * gh, : bw * gw]
    return arr.reshape(gh, bh, gw, bw).mean(axis=(1, 3))


def pixel_delta_maps(f_prev, f_cur, gh, gw):
    """
    f_prev, f_cur : (H,W,3) float32 in [0,1].
    Returns dict of (gh,gw) maps for L1, L2, cosdist (per-pixel then block-pooled
    to the patch grid so they line up with the embedding heatmaps).
    """
    diff = f_cur - f_prev                       # (H,W,3)
    l1_full = np.abs(diff).mean(axis=2)         # (H,W) mean over channels
    l2_full = np.sqrt((diff ** 2).mean(axis=2))
    # per-pixel cosine distance over the 3 RGB channels
    a = f_prev.reshape(-1, 3); b = f_cur.reshape(-1, 3)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    cos_full = (1.0 - num / den).reshape(f_prev.shape[:2])
    return {
        "l1":  _block_reduce_mean(l1_full,  gh, gw),
        "l2":  _block_reduce_mean(l2_full,  gh, gw),
        "cos": _block_reduce_mean(cos_full, gh, gw),
    }


def embed_delta_maps(e_prev, e_cur, gh, gw):
    """
    e_prev, e_cur : (gh*gw, D) patch-token embeddings for two consecutive frames.
    Per-patch L1 (mean |.| over D), L2 (norm over D), cosine distance (1-cos).
    Returns dict of (gh,gw) maps.
    """
    e_prev = e_prev.astype(np.float32); e_cur = e_cur.astype(np.float32)
    diff = e_cur - e_prev
    l1 = np.abs(diff).mean(axis=1)
    l2 = np.linalg.norm(diff, axis=1)
    num = (e_prev * e_cur).sum(1)
    den = np.linalg.norm(e_prev, axis=1) * np.linalg.norm(e_cur, axis=1) + 1e-8
    cos = 1.0 - num / den
    return {
        "l1":  l1.reshape(gh, gw),
        "l2":  l2.reshape(gh, gw),
        "cos": cos.reshape(gh, gw),
    }


def token_norm_means(tok):
    """Mean token magnitude over patches, used to normalise embedding deltas so
    they are comparable across models ('change as a fraction of token size'):
      n1 = mean_p(||e(p)||_1 / D)  -> matches L1 delta (mean |.| over D)
      n2 = mean_p ||e(p)||_2       -> matches L2 delta (Euclidean norm)."""
    t = np.asarray(tok, dtype=np.float32)
    n1 = float((np.abs(t).sum(axis=1) / t.shape[1]).mean())
    n2 = float(np.linalg.norm(t, axis=1).mean())
    return n1, n2


# ----------------------------------------------------------------------------- scene cut
def detect_scene_cuts(series, k=SCENE_CUT_K):
    """
    series : (N-1,) pixel-L2 scalar deltas (delta[i] = frame i+1 vs i).
    A cut at frame index i+1 when delta[i] exceeds median + k*MAD.
    Returns (cut_frame_indices list, threshold).
    """
    s = np.asarray(series, dtype=np.float64)
    med = np.median(s)
    mad = np.median(np.abs(s - med)) + 1e-12
    thr = med + k * 1.4826 * mad
    cut_pair_idx = np.where(s > thr)[0]
    cut_frames = [int(i + 1) for i in cut_pair_idx]   # frame index that STARTS a new shot
    return cut_frames, float(thr)


# ----------------------------------------------------------------------------- model helpers
def get_submodule(model, paths):
    """Return the first attribute path (dotted) that exists on model."""
    for p in paths:
        obj = model; ok = True
        for a in p.split("."):
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False; break
        if ok:
            return obj
    raise AttributeError(f"none of {paths} found on {type(model).__name__}")


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


# ----------------------------------------------------------------------------- frame loading
def _maybe_gray(frames):
    """If env VDA_GRAY is set, convert each (H,W,3) RGB frame to luminance (Rec.601)
    and replicate back to 3 identical channels. This makes BOTH the pixel-difference
    metric AND the encoder input grayscale, since every extractor reads these frames."""
    if not os.environ.get("VDA_GRAY"):
        return frames
    out = []
    for f in frames:
        g = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2])
        g = np.clip(np.round(g), 0, 255).astype(np.uint8)
        out.append(np.repeat(g[..., None], 3, axis=2))
    return out


def load_frames(video_path, res=FIXED_RES):
    """Decode ALL frames, resize each to res x res (RGB uint8). decord -> av fallback.
    Set env VDA_MAXF=N to cap frame count (smoke tests).
    Set env VDA_GRAY=1 to convert frames to grayscale (luminance, replicated to 3ch)."""
    from PIL import Image
    maxf = int(os.environ.get("VDA_MAXF", "0")) or None
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        n = len(vr) if maxf is None else min(maxf, len(vr))
        out = []
        for i in range(n):
            arr = vr[i].asnumpy()
            out.append(np.asarray(Image.fromarray(arr).resize((res, res), Image.BILINEAR),
                                  dtype=np.uint8))
        return _maybe_gray(out)
    except Exception:
        import av
        c = av.open(video_path)
        out = []
        for fr in c.decode(video=0):
            img = fr.to_image().convert("RGB").resize((res, res), Image.BILINEAR)
            out.append(np.asarray(img, dtype=np.uint8))
            if maxf is not None and len(out) >= maxf:
                break
        c.close()
        return _maybe_gray(out)


# ----------------------------------------------------------------------------- encoder timing
def time_forward(forward_gpu, sample_frame, warmup=5, iters=30):
    """
    Measure pure vision-encoder GPU forward latency on a single frame.
    forward_gpu(frame_uint8) must run ONLY the encoder forward and return a CUDA tensor
    (no .cpu()/.numpy()).  Uses CUDA events; returns dict of ms stats.
    """
    import torch
    torch.cuda.synchronize()
    for _ in range(warmup):
        forward_gpu(sample_frame)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record()
        forward_gpu(sample_frame)
        en.record(); torch.cuda.synchronize()
        times.append(st.elapsed_time(en))   # ms
    t = np.asarray(times, dtype=np.float64)
    return {"per_frame_ms_mean": float(t.mean()), "per_frame_ms_std": float(t.std()),
            "per_frame_ms_median": float(np.median(t)), "fps": float(1000.0 / t.mean()),
            "warmup": warmup, "iters": iters}


# ----------------------------------------------------------------------------- analysis 2
def motion_concentration(maps, cut_pair_idx, pctls=MOTION_PCTLS):
    """
    Analysis 2. For each frame pair, threshold the per-patch PIXEL-L2 delta to define a
    'motion mask' (patches whose adjacent-frame change is large), then measure how much of
    the TOTAL embedding change falls inside that mask.

      R(metric)     = sum(emb_delta[mask]) / sum(emb_delta[all])
      area_fraction = |mask| / |patches|
      R / area      = concentration factor (>>1 => embedding change concentrated on motion)

    Thresholds are video-internal percentiles of pixel-L2 (over non-cut pairs), so a static
    video gets its own scale.  Pairs with an empty mask are skipped for R.
    """
    pix_l2 = maps["l2"]["pix"]                      # (P, gh, gw)
    npair = pix_l2.shape[0]
    keep = [i for i in range(npair) if i not in cut_pair_idx]
    flat = np.concatenate([pix_l2[i].ravel() for i in keep]) if keep else np.zeros(1)
    out = {}
    for p in pctls:
        tau = float(np.percentile(flat, p))
        Rvals = {m: [] for m in METRICS}; areas = []; nonempty = 0
        for i in keep:
            mask = pix_l2[i] > tau
            asum = int(mask.sum())
            if asum == 0:
                continue
            nonempty += 1
            areas.append(asum / mask.size)
            for m in METRICS:
                em = maps[m]["emb"][i]
                tot = float(em.sum())
                Rvals[m].append(float(em[mask].sum() / (tot + 1e-12)))
        amean = float(np.mean(areas)) if areas else 0.0
        out[f"p{p}"] = {
            "tau_pixel_l2": tau,
            "area_fraction": amean,
            "nonempty_pairs": nonempty,
            "total_pairs": len(keep),
            "R": {m: (float(np.mean(Rvals[m])) if Rvals[m] else 0.0) for m in METRICS},
            "R_over_area": {m: ((float(np.mean(Rvals[m])) / amean) if (Rvals[m] and amean) else 0.0)
                            for m in METRICS},
        }
    return out


# ----------------------------------------------------------------------------- generic driver
def run_pipeline(video_path, out_dir, model_name, forward_gpu, to_tokens, meta,
                 res=FIXED_RES, time_iters=30):
    """
    End-to-end per-(model, video) pipeline used by every extract_*.py:
      load frames -> time encoder -> encode all -> pixel/embedding deltas -> npz ->
      run_analysis (incl. analysis 2) -> write meta.json (params/res/timing).

      forward_gpu(frame_uint8) -> CUDA tensor (encoder forward only; for timing)
      to_tokens(frame_uint8)   -> np.float16 (P, D) patch tokens on CPU (for deltas)
      meta : static dict (model_type, total_params, vision_encoder_params, native_res,
             patch_grid=[gh,gw] or None, patch_dim, ...). Updated in place with timing.
    """
    from PIL import Image
    if os.environ.get("VDA_GRAY"):
        out_dir = out_dir.replace("/out/", "/out_gray/")
        meta["grayscale"] = True
    os.makedirs(out_dir, exist_ok=True)
    frames = load_frames(video_path, res)
    N = len(frames)
    print(f"[{model_name}] {N} frames @ {res}x{res} from {os.path.basename(video_path)}")

    timing = time_forward(forward_gpu, frames[N // 2], iters=time_iters)
    print(f"[{model_name}] encoder {timing['per_frame_ms_mean']:.2f}+-{timing['per_frame_ms_std']:.2f} ms/frame "
          f"({timing['fps']:.1f} fps)")

    # ---- streaming delta computation (keep only the previous frame's tokens in memory:
    #      SAM2/SAM3 grids are 4096/5184 tokens/frame, too big to store all)
    gh = gw = None
    if meta.get("patch_grid"):
        gh, gw = int(meta["patch_grid"][0]), int(meta["patch_grid"][1])
    scal = {f"{mod}_{m}": [] for mod in ("pix", "emb") for m in METRICS}
    mp = {f"{mod}map_{m}": [] for mod in ("pix", "emb") for m in METRICS}

    prev01 = frames[0].astype(np.float32) / 255.0
    prev_tok = to_tokens(frames[0])                  # (P, D) float16
    if gh is None:
        P = prev_tok.shape[0]; gh = gw = int(round(P ** 0.5))
    P0 = prev_tok.shape[0]
    tok_n1, tok_n2 = [], []
    _n1, _n2 = token_norm_means(prev_tok); tok_n1.append(_n1); tok_n2.append(_n2)
    for i in range(1, N):
        cur01 = frames[i].astype(np.float32) / 255.0
        cur_tok = to_tokens(frames[i])
        # token count must be constant for 1:1 patch correspondence across frames.
        # (Guaranteed here because every frame is fed at the same FIXED_RES, so dynamic-
        #  resolution models like Qwen-VL produce an identical patch grid every frame.)
        assert cur_tok.shape[0] == P0, (
            f"{model_name}: patch count changed {P0}->{cur_tok.shape[0]} at frame {i}; "
            f"dynamic-resolution grid is not constant.")
        _n1, _n2 = token_norm_means(cur_tok); tok_n1.append(_n1); tok_n2.append(_n2)
        pm = pixel_delta_maps(prev01, cur01, gh, gw)
        em = embed_delta_maps(prev_tok, cur_tok, gh, gw)
        for m in METRICS:
            scal[f"pix_{m}"].append(float(pm[m].mean())); mp[f"pixmap_{m}"].append(pm[m])
            scal[f"emb_{m}"].append(float(em[m].mean())); mp[f"embmap_{m}"].append(em[m])
        prev01, prev_tok = cur01, cur_tok
        if i % 50 == 0:
            print(f"  encoded {i}/{N}")

    meta["patch_grid"] = [gh, gw]
    meta["patch_dim"] = int(prev_tok.shape[-1])
    meta["n_vision_tokens"] = int(gh * gw)
    meta["n_frames"] = N
    meta["processing_res"] = res
    meta["timing"] = timing
    meta["token_norm_l1"] = float(np.mean(tok_n1))   # mean token ||.||_1/D  (L1 normaliser)
    meta["token_norm_l2"] = float(np.mean(tok_n2))   # mean token ||.||_2     (L2 normaliser)

    thumbs = np.stack([np.asarray(Image.fromarray(f).resize((THUMB, THUMB)), dtype=np.uint8)
                       for f in frames])
    npz = os.path.join(out_dir, "frames.npz")
    save = {"n_frames": N, "gh": gh, "gw": gw, "thumbs": thumbs,
            "tok_norm_l1": np.asarray(tok_n1, np.float32),
            "tok_norm_l2": np.asarray(tok_n2, np.float32)}
    for k, v in scal.items(): save[k] = np.asarray(v, dtype=np.float32)
    for k, v in mp.items():   save[k] = np.asarray(v, dtype=np.float32)
    np.savez_compressed(npz, **save)

    summary = run_analysis(npz, out_dir, model_name)
    meta["delta_summary"] = summary
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{model_name}] meta.json written -> {out_dir}")
    return summary


# ----------------------------------------------------------------------------- analysis + plots
METRICS = ["l1", "l2", "cos"]
METRIC_LABEL = {"l1": "L1", "l2": "L2", "cos": "cosine-dist"}


def _pearson(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    # rank-transform then Pearson (no scipy dependency)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return _pearson(rx, ry)


def _linfit_slope(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if x.std() < 1e-12:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def analysis1_stats(scal, keep):
    """
    Analysis 1 final statistics: how well does adjacent-frame PIXEL change predict
    vision-encoder EMBEDDING change, over non-cut frame pairs.

    AXIS-METRIC CONSISTENCY: x and y MUST use the SAME distance metric, otherwise the
    correlation mixes apples (pixel-L2) and oranges (embedding-cos) and the slope/units
    are meaningless. We therefore report MATCHED pairs:
      - primary   : pixel cosine-distance  vs  embedding cosine-distance   (cos <-> cos)
      - secondary : pixel L2               vs  embedding L2                (L2  <-> L2)
    cosine is the cross-model representative metric, so pixCOS_vs_embCOS is primary.
    With matched units the OLS slope is now interpretable as a sensitivity gain
    (d emb-change / d pixel-change).
    """
    out = {}
    for mk in ("l1", "l2", "cos"):                 # SAME metric on both axes (L1, L2, cosine)
        x = np.asarray(scal[mk]["pix"])[keep]
        y = np.asarray(scal[mk]["emb"])[keep]
        r = _pearson(x, y)
        out[f"pix{mk.upper()}_vs_emb{mk.upper()}"] = {
            "pearson_r": r, "spearman_rho": _spearman(x, y),
            "r_squared": r * r, "ols_slope": _linfit_slope(x, y)}
    return out


def run_analysis(npz_path, out_dir, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    d = np.load(npz_path, allow_pickle=True)
    N = int(d["n_frames"])
    gh, gw = int(d["gh"]), int(d["gw"])
    thumbs = d["thumbs"]                       # (N, th, tw, 3) uint8
    pair_frames = np.arange(1, N)              # delta[i] corresponds to frame i (vs i-1)

    # scalars: (N-1,) each
    scal = {m: {"pix": d[f"pix_{m}"], "emb": d[f"emb_{m}"]} for m in METRICS}
    # maps: (N-1, gh, gw)
    maps = {m: {"pix": d[f"pixmap_{m}"], "emb": d[f"embmap_{m}"]} for m in METRICS}

    # ---- scene cuts (from pixel L2)
    cut_frames, thr = detect_scene_cuts(scal["l2"]["pix"])
    cut_pair_idx = set(int(c - 1) for c in cut_frames)

    with open(os.path.join(out_dir, "scene_cuts.json"), "w") as f:
        json.dump({"model": model_name, "video_frames": N,
                   "pixel_l2_threshold": thr,
                   "num_cuts": len(cut_frames),
                   "cut_frames": cut_frames}, f, indent=2)
    with open(os.path.join(out_dir, "scene_cuts.txt"), "w") as f:
        f.write(f"# {model_name}  scene cuts (pixel-L2 > {thr:.5f})\n")
        f.write(f"# total {len(cut_frames)} cut(s) among {N} frames\n")
        for c in cut_frames:
            f.write(f"frame {c}  (pixel-L2 delta = {scal['l2']['pix'][c-1]:.5f})\n")

    # ---- averages excluding cut frames
    keep = np.array([i for i in range(N - 1) if i not in cut_pair_idx])
    summary = {"model": model_name, "video_frames": N, "num_cuts": len(cut_frames),
               "averages_excluding_cuts": {}, "averages_all": {}}
    for m in METRICS:
        summary["averages_excluding_cuts"][m] = {
            "pixel": float(scal[m]["pix"][keep].mean()),
            "embedding": float(scal[m]["emb"][keep].mean())}
        summary["averages_all"][m] = {
            "pixel": float(scal[m]["pix"].mean()),
            "embedding": float(scal[m]["emb"].mean())}

    # ---- analysis 1: pixel-change -> embedding-change predictivity (correlation/slope)
    summary["analysis1_correlation"] = analysis1_stats(scal, keep)

    # ---- analysis 2: motion-region concentration of embedding change
    summary["motion_concentration"] = motion_concentration(maps, cut_pair_idx)

    with open(os.path.join(out_dir, "delta_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- analysis-2 plot: R(metric) vs area fraction at each percentile
    mc = summary["motion_concentration"]
    pkeys = list(mc.keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pkeys)); w = 0.2
    for j, m in enumerate(METRICS):
        ax.bar(x + (j - 1) * w, [mc[p]["R"][m] for p in pkeys], w, label=f"R ({METRIC_LABEL[m]})")
    ax.plot(x, [mc[p]["area_fraction"] for p in pkeys], "k--o", lw=1.2, label="area fraction")
    ax.set_xticks(x); ax.set_xticklabels(pkeys)
    ax.set_xlabel("pixel-L2 motion threshold (percentile)")
    ax.set_ylabel("fraction of total embedding change")
    ax.set_title(f"{model_name}  analysis-2: embedding change in motion region vs its area")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "analysis2_concentration.png"), dpi=110)
    plt.close(fig)

    # ---- global maxima (per model) for consistent bar/colorbar scaling (task 5)
    gmax = {m: {"pix": float(scal[m]["pix"].max()),
                "emb": float(scal[m]["emb"].max())} for m in METRICS}
    map_gmax = {m: {"pix": float(maps[m]["pix"].max()),
                    "emb": float(maps[m]["emb"].max())} for m in METRICS}

    # ===================================================================== time-series
    for m in METRICS:
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        for ax, mod, color in ((axes[0], "pix", "#1f77b4"), (axes[1], "emb", "#d62728")):
            ax.bar(pair_frames, scal[m][mod], width=1.0, color=color)
            ax.set_ylim(0, gmax[m][mod] * 1.05)
            ax.set_ylabel(f"{'pixel' if mod=='pix' else 'embedding'}\n{METRIC_LABEL[m]} delta")
            for c in cut_frames:
                ax.axvline(c, color="k", ls="--", lw=0.6, alpha=0.5)
        axes[1].set_xlabel("frame t  (delta vs t-1);  dashed = scene cut")
        axes[0].set_title(f"{model_name}  per-frame {METRIC_LABEL[m]} delta  "
                          f"(pixel top / embedding bottom)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"timeseries_{m}.png"), dpi=110)
        plt.close(fig)

    # normalized overlay: pixel & embedding on a SHARED [0,1] axis (max matched)
    for m in METRICS:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(pair_frames, scal[m]["pix"] / (gmax[m]["pix"] + 1e-12),
                color="#1f77b4", lw=1.0, label="pixel (norm.)")
        ax.plot(pair_frames, scal[m]["emb"] / (gmax[m]["emb"] + 1e-12),
                color="#d62728", lw=1.0, label="embedding (norm.)")
        ax.set_ylim(0, 1.05); ax.set_ylabel(f"{METRIC_LABEL[m]} (norm. to global max)")
        ax.set_xlabel("frame t")
        for c in cut_frames:
            ax.axvline(c, color="k", ls="--", lw=0.6, alpha=0.4)
        ax.set_title(f"{model_name}  {METRIC_LABEL[m]}  pixel vs embedding (max-matched)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"timeseries_norm_{m}.png"), dpi=110)
        plt.close(fig)

    # ===================================================================== composite heatmap pages (task 4)
    # pick characteristic frame pairs from the EMBEDDING L2 series (excluding cuts):
    #   anchor (t=1), then the smallest-change and largest-change frames.
    ref = scal["l2"]["emb"].copy()
    order = [i for i in np.argsort(ref) if i not in cut_pair_idx]
    smallest = order[:4]
    largest = order[::-1][:4]
    picks = sorted(set([0] + list(smallest) + list(largest)))    # pair indices
    pick_frames = [int(i + 1) for i in picks]

    with open(os.path.join(out_dir, "selected_frames.json"), "w") as f:
        json.dump({"selected_frames(t)": pick_frames,
                   "note": "anchor t=1, 4 smallest- and 4 largest-change (embedding L2, cuts excluded)"},
                  f, indent=2)

    for m in METRICS:
        for mod, mlab in (("pix", "pixel"), ("emb", "embedding")):
            vmax = map_gmax[m][mod]
            n = len(picks)
            ncol = n
            fig, axes = plt.subplots(2, ncol, figsize=(2.4 * ncol, 5.4))
            if ncol == 1:
                axes = axes.reshape(2, 1)
            for j, pi in enumerate(picks):
                fr = pi + 1
                # top row: frame thumbnail
                axes[0, j].imshow(thumbs[fr]); axes[0, j].axis("off")
                tag = "  [CUT]" if pi in cut_pair_idx else ""
                axes[0, j].set_title(f"t={fr}{tag}", fontsize=9)
                # bottom row: delta heatmap (shared vmax)
                im = axes[1, j].imshow(maps[m][mod][pi], cmap="inferno",
                                       vmin=0, vmax=vmax)
                axes[1, j].axis("off")
                axes[1, j].set_title(f"{scal[m][mod][pi]:.4f}", fontsize=8)
            cbar = fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.025, pad=0.01)
            cbar.set_label(f"{mlab} {METRIC_LABEL[m]} (vmax={vmax:.4g})", fontsize=8)
            fig.suptitle(f"{model_name}  {mlab} {METRIC_LABEL[m]} delta — characteristic frames "
                         f"(t vs t-1)", fontsize=12)
            fig.savefig(os.path.join(out_dir, f"composite_{mod}_{m}.png"),
                        dpi=120, bbox_inches="tight")
            plt.close(fig)

    print(f"[{model_name}] analysis done -> {out_dir}")
    print(f"  scene cuts: {len(cut_frames)}  | selected frames: {pick_frames}")
    return summary
