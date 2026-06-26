"""Generate RESULTS.md (quantitative findings) from results/gray_summary.json.
Re-run after re-aggregating: python gen_results_md.py
Timing/correlation/CF come from the summary JSON; parameter counts and
resolutions are static model metadata (not stored in the per-frame outputs)."""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, "results", "gray_summary.json")
OUT = os.path.join(HERE, "RESULTS.md")

# static model metadata (params in millions, resolutions, token count)
META = {
    "CLIP-L/14":            dict(total="303M",   venc="303M", native="224",     proc="448", tok="256"),
    "DINOv2-L/14":          dict(total="304M",   venc="304M", native="518",     proc="448", tok="1,024"),
    "SAM2.1-Hiera-B+":      dict(total="81M",    venc="69M",  native="1,024",   proc="448", tok="4,096"),
    "SAM3-ViT":             dict(total="841M",   venc="446M", native="1,008",   proc="448", tok="5,184"),
    "InternVL2.5-8B":       dict(total="8,075M", venc="304M", native="448",     proc="448", tok="1,024"),
    "InternVL3-8B":         dict(total="7,944M", venc="304M", native="448",     proc="448", tok="1,024"),
    "LLaVA-OneVision-0.5B": dict(total="894M",   venc="428M", native="384",     proc="448", tok="729"),
    "Qwen2.5-VL-7B":        dict(total="8,292M", venc="677M", native="동적",    proc="448", tok="1,024"),
    "Qwen3-VL-8B":          dict(total="8,767M", venc="576M", native="동적",    proc="448", tok="784"),
}
rows = json.load(open(SUMMARY))


def f(x, n=3):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"


L = []
L.append("# 정량 결과 (회색조 · L1/L2)\n")
L.append("`results/gray_summary.json`에서 자동 생성된 표입니다 (`python gen_results_md.py`). "
         "컷(장면 전환) 프레임은 모든 평균에서 제외했습니다. 산점도는 `results/`의 PNG, "
         "분석 배경·수식은 [ARCHITECTURE.md](ARCHITECTURE.md), 서술형 리포트는 "
         "[results/REPORT.md](results/REPORT.md) 참고.\n")

# --- Table 0: timing + params ---
L.append("## 분석 0 — 비전 인코더 연산시간 · 파라미터 · 해상도\n")
L.append("| 모델 | 종류 | 전체 / 비전 인코더 파라미터 | 기본→처리 해상도 | 토큰 수 | 연산시간 빠름/정적 (ms) |")
L.append("|---|---|---|---|---|---|")
for r in rows:
    m = META.get(r["name"], {})
    tf = r.get("fast", {}).get("timing_ms"); ts = r.get("static", {}).get("timing_ms")
    L.append(f"| {r['name']} | {r['type']} | {m.get('total','?')} / {m.get('venc','?')} | "
             f"{m.get('native','?')}→{m.get('proc','?')} | {m.get('tok','?')} | "
             f"{f(tf,1)} / {f(ts,1)} |")

# --- Table 1: analysis 1 means + correlation + slope ---
L.append("\n## 분석 1 — 픽셀 변화 vs 인코딩 변화 (평균 · 상관 · 민감도)\n")
L.append("픽셀 변화는 회색조라 L1=L2. `r`=Pearson 상관, `slope`=OLS 기울기(픽셀 1단위당 인코딩 변화, 민감도). "
         "L2 기울기는 임베딩 절대 크기에 비례하므로 모델 간 직접 비교 금지(특히 Qwen).\n")
L.append("| 모델 | 영상 | 픽셀Δ | 인코딩Δ L1 | 인코딩Δ L2 | r(L1) | r(L2) | slope(L1) | slope(L2) |")
L.append("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    for v in ("fast", "static"):
        d = r.get(v)
        if not d:
            continue
        c = d["corr"]
        L.append(f"| {r['name']} | {v} | {f(d['pix']['l1'],4)} | {f(d['emb']['l1'])} | {f(d['emb']['l2'],2)} | "
                 f"{f(c['l1']['pearson_r'])} | {f(c['l2']['pearson_r'])} | "
                 f"{f(c['l1']['ols_slope'],2)} | {f(c['l2']['ols_slope'],1)} |")

# --- Table 2: concentration factor ---
L.append("\n## 분석 2 — 움직임 영역 집중도 (Concentration Factor)\n")
L.append("CF = (움직임 영역 인코딩 변화 비율) / (면적 비율). 상위 10% 픽셀 변화 패치를 움직임 영역으로 본다. "
         "CF≫1이면 변화가 움직임 영역에 집중 → 정적 패치 재사용 여지 큼. CF는 비율이라 L1≈L2.\n")
L.append("| 모델 | 영상 | 움직임 면적 a | CF (L1) | CF (L2) |")
L.append("|---|---|---|---|---|")
for r in rows:
    for v in ("fast", "static"):
        d = r.get(v)
        if not d:
            continue
        L.append(f"| {r['name']} | {v} | {f(d['area'])} | {f(d['cf']['l1'],2)} | {f(d['cf']['l2'],2)} |")

L.append("\n---\n*이 파일은 `gen_results_md.py`로 자동 생성됩니다. 수치를 직접 수정하지 말고 재집계 후 재생성하세요.*")

open(OUT, "w").write("\n".join(L) + "\n")
print("written", OUT)
