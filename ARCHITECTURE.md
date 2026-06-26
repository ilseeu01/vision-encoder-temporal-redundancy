# 코드 구조 설명 — Vision Encoder 오버헤드 & 시간적 중복성 분석

> 서로 다른 LMM/VFM 9종의 vision encoder에 대해, 빠른 영상 vs 정적 영상에서
> ① 인코더 연산시간 ② 인접 프레임 대비 인코딩 변화량(분석1) ③ 움직임 영역에
> 인코딩 변화가 집중되는 정도(분석2)를 측정한다.

---

## 0. 연구 가설 (Hypothesis) — 왜 이 실험을 하는가

본 연구의 궁극적 목적은 **on-device vision encoding의 시간적 중복(temporal redundancy)을
활용한 연산 절감 가능성**을 검증하는 것이다. 분석1·2·TRS는 다음 세 가설을 검증하기 위한
도구다.

- **H1.** 정적/저움직임 영상에서는 인접 프레임의 encoder representation 변화가 **매우 작다**
  (= TRS가 1에 가깝다). → *분석1(작은 입력 변화) + TRS로 검증.*
- **H2.** Representation 변화는 **움직임 영역에 집중**된다 (정적 패치는 거의 안 변함).
  → *분석2의 Concentration Factor(CF≫1)로 검증.*
- **H3.** 따라서 **정적 패치의 encoder 출력을 재사용하는 cache-based vision encoding**이
  가능하며, vision encoder 오버헤드(연산시간 측정)를 줄일 여지가 크다.
  → *연산시간 표 + H1·H2 결합으로 동기 부여.*

> 즉 "분석1·2를 통해 무엇을 증명하려는가?" = **H3 (정적 패치 재사용 가능성)**.
> 분석1=재사용의 시간축 근거(프레임 간 안정성), 분석2=재사용의 공간축 근거(움직임 영역 한정).

---

## 1. 설계 핵심 아이디어

모든 모델이 **동일한 파이프라인**(`common.run_pipeline`)을 공유하고, 모델별 스크립트는
오직 **"한 프레임 → vision encoder forward → 패치 토큰 (P, D)"** 부분만 정의한다.
즉 모델마다 다른 것은:

- 모델 로딩 방법 (env, 클래스, 체크포인트)
- 전처리(해상도/정규화)
- vision encoder의 **임베딩 추출 지점**

이 세 가지뿐이고, 나머지(프레임 디코딩 · delta 계산 · scene cut · 타이밍 · 분석1/2 ·
그래프 · 결과 저장)는 전부 `common.py`가 담당한다.

```
ext_<model>.py  ──(forward 함수 2개 + meta)──▶  common.run_pipeline  ──▶  out/<model>/<video>/
                                                       │
                                                       ├─ common.load_frames      (영상 디코딩)
                                                       ├─ common.time_forward      (연산시간 측정)
                                                       ├─ common.pixel/embed_delta_maps (delta)
                                                       ├─ common.run_analysis      (분석1 + scene cut + 그래프)
                                                       └─ common.motion_concentration (분석2)
```

### 1.1 Representation 선택 근거 (왜 패치 토큰인가)

본 연구는 모델별 **최종 vision representation(projector/merger 이후)**이 아니라,
**vision encoder의 spatial patch token**을 분석 대상으로 한다. 근거:

1. **패치 단위 위치 정보 유지** — 토큰이 영상의 공간 위치에 1:1 대응하므로,
   움직임 영역에 인코딩 변화가 집중되는지(분석2)를 측정할 수 있다.
2. **공정한 cross-model 비교** — projector / merger / pixel-shuffle 이후 표현은
   모델마다 구조·차원·토큰 수가 제각각이라 동일 기준 비교가 불가능하다.
   패치 토큰은 "ViT가 본 그대로"라 비교 단위가 일관된다.
3. **CLS / pooled token 배제** — CLS·pooled 출력은 전역 요약이라 **공간 정보를 잃어**
   분석2를 수행할 수 없다. (그래서 ViT 계열은 모두 `[:,1:]`로 CLS를 떼고 패치만 사용)
4. **LLM 입력 직전 = vision encoder의 "출력"으로 정의** — Qwen 계열은
   `ViT → patch-merge → merger → LLM` 구조인데, 본 연구는 **merge 직전 ViT 패치 토큰**
   (즉 merger의 입력)을 vision encoder 출력으로 정의한다(아래 4절 표·구현 참고).

---

## 2. 파일 구성

| 파일 | 역할 | 실행 env |
|---|---|---|
| **`common.py`** | 공용 엔진 (전 모델 공유). delta 정의, 타이밍, 분석1/2, 그래프, 드라이버 | - |
| **`videos.json`** | 실험 영상 2종(fast/static)의 경로·메타 | - |
| `ext_qwen3.py` | Qwen3-VL-8B 추출 | `qwen3vl` |
| `ext_qwen25.py` | Qwen2.5-VL-7B 추출 | `qwen3vl` |
| `ext_intern.py` | InternVL3 / InternVL2.5 추출 (`argv: 3` 또는 `25`) | `internvl` |
| `ext_llava_ov.py` | LLaVA-OneVision-0.5B 추출 | `qwen3vl` |
| `ext_dinov2.py` | DINOv2-L/14 추출 | `qwen3vl` |
| `ext_clip.py` | CLIP-L/14 추출 | `qwen3vl` |
| `ext_sam2.py` | SAM2.1 Hiera-B+ 추출 | `sam2` |
| `ext_sam3.py` | SAM3 ViT trunk 추출 | `sam3` |
| **`make_tables.py`** | 모든 `out/*/*/meta.json` 취합 → 최종 리포트 4개 표 | 아무 env |
| **`run_all.sh`** | 9개 모델 × 2영상을 4 GPU에 분산 병렬 실행 후 리포트 생성 | - |
| `dl_models.sh` | 신규 모델 4종 HF 다운로드 | - |
| `dl_videos.sh` | YouTube 후보 영상 다운로드(yt-dlp) | - |

> `extract_qwen.py` / `extract_intern.py`는 **이전 버전**(단일 영상, 타이밍·분석2 없음).
> 현재는 `ext_*.py` 9종이 정식이며 기존 파일은 참고용으로 남겨둠.

---

## 3. `common.py` — 공용 엔진 상세

### 3.1 설정 상수
```python
FIXED_RES   = 448          # 픽셀 delta 계산용 프레임 해상도 (모든 모델 공통)
SCENE_CUT_K = 6.0          # scene cut 임계 (median + k·MAD)
MOTION_PCTLS = (80,90,95)  # 분석2: "움직임 영역" 정의용 픽셀-L2 백분위 임계
THUMB = 160                # 썸네일 한 변
```

- **`SCENE_CUT_K = 6`** : robust outlier detection에서 널리 쓰이는 `median + k·MAD`
  기준을 사용한다. 실험 영상에서 일반 움직임은 유지하고 **장면 전환만** 제거하도록
  경험적으로 설정한 값으로, 5·7과 비교했을 때 fast 영상의 컷 검출이 안정적이었다.
- **`MOTION_PCTLS = (80,90,95)`** : 단일 임계의 임의성을 피하기 위한 **민감도 분석**.
  각각 움직임 상위 **20% / 10% / 5%** 영역에 해당하며, 임계를 높여도(영역이 좁아져도)
  집중 경향이 유지되는지 확인한다. 리포트 대표값은 **p90(상위 10%)**.

### 3.2 metric 정의 — 패치 단위 delta
프레임쌍(t-1, t)마다 **패치별로** 세 가지 거리를 계산하여 `(gh, gw)` 히트맵 + 스칼라(평균) 산출.

- `pixel_delta_maps(f_prev, f_cur, gh, gw)` : [0,1] RGB에서 per-pixel L1/L2/cosine →
  `_block_reduce_mean`으로 패치 그리드에 평균풀링 (임베딩 히트맵과 위치 1:1 정렬).
- `embed_delta_maps(e_prev, e_cur, gh, gw)` : 같은 위치 패치 토큰끼리 L1(mean|·|),
  L2(‖·‖), cosine-distance(1−cos).
- cosine은 "클수록 변화 큼"이 되도록 **distance(1−cos)** 로 기록.

### 3.3 `detect_scene_cuts(series)`
픽셀-L2 시계열에서 `median + 6·1.4826·MAD`를 넘는 프레임을 컷으로 판정(robust).
분석1/2의 평균은 컷 프레임을 **제외**하고 낸다(장면 전환이 통계를 오염시키지 않도록).

### 3.4 `time_forward(forward_gpu, sample_frame)` — 연산시간 측정
- **vision encoder forward만** 격리해서 측정 (`forward_gpu`는 GPU 텐서 반환, `.cpu()` 없음).
- warmup 5회 → **CUDA Event**로 30회 측정 → `per_frame_ms_mean/std/median`, `fps` 반환.
- 측정 입력은 영상 중간 프레임 1장(대표값), 해상도는 각 모델이 실제로 쓰는 처리 해상도.

### 3.5 `motion_concentration(maps, cut_pair_idx)` — **분석2**
프레임쌍마다:
1. **움직임 마스크** `M = { 패치 p : pixel_L2_delta_p > τ }`, τ = 영상 내부 픽셀-L2의 백분위(80/90/95).
2. `R(metric) = Σ_{p∈M} emb_delta_p / Σ_{all p} emb_delta_p` — 전체 인코딩 변화 중 움직임 영역 비율.
3. `area_fraction = |M| / |전체 패치|`.
4. **`Concentration Factor (CF) = R / area_fraction` ← 분석2의 primary metric.**

> **Motion proxy 주의**: 여기서 "움직임"은 **optical flow가 아니라 pixel-difference 기반
> motion proxy**다. 따라서 카메라 auto-exposure·조명 변화 같은 **전역 밝기 변화도 motion으로
> 잡힐 수 있다.** 본 실험은 fast/static 두 영상이 *실제 움직임*에서 크게 갈리므로 이 proxy로
> 충분하지만, 일반화 시 optical-flow 기반 마스크가 더 엄밀하다(향후 과제).

> **왜 R이 아니라 CF가 핵심인가** : R 자체는 움직임 패치 개수(area)가 많아지면 당연히
> 커지므로, "정말 집중되어 있는지"를 R만으로 판단할 수 없다. 그래서 **면적 비율로 정규화**한
> CF를 대표 지표로 쓴다.
> - **CF 범위**: 최소 0, **`CF = 1`이면 공간적으로 균등 분포**, **`CF ≫ 1`이면 특정(움직임)
>   영역에 집중**. 상한은 없다(area→0이면 발산 가능).
> - `CF > 1` → 움직임 영역이 자기 면적 몫보다 **더 많은** 인코딩 변화를 담당 → **정적 패치는 중복(캐싱 여지)**.
> - `CF ≈ 1` → 변화가 공간적으로 균등 → 캐싱 이득 없음.
>
> 예: `area_fraction = 0.1`, `R = 0.6` → `CF = 6.0` ⇒ **10% 영역이 전체 변화의 60%를 담당**.

마스크가 빈 프레임쌍은 R 평균에서 제외, 컷도 제외. metric(L1/L2/cos)별로 산출하며,
80/90/95 백분위 모두에 대해 출력해 **민감도(임계 강도)** 를 함께 본다.

### 3.5.1 Temporal Redundancy Score (TRS) — headline 요약 지표
"그래서 중복성이 얼마냐?"에 한 줄로 답하는 대표 지표 (리포트 Table 3, `make_tables`에서 산출).
```
TRS = 1 − mean(embedding cosine-distance)   (컷 제외 인접 프레임쌍 평균)
```
- `TRS → 1` : 인접 프레임의 representation이 거의 동일 = **중복성 높음 → 재사용/캐싱 유리** (H1·H3).
- cos 기반이라 **모델 간 비교 가능**. 예: static TRS 0.99 ⇒ "프레임 간 표현 99% 유지".
- fast/static의 TRS 차이(`drop`)는 **움직임 민감도**를 한눈에 보여준다.

### 3.6 `run_pipeline(...)` — 제너릭 드라이버 (핵심)
모델별 스크립트가 호출하는 단일 진입점.

```python
run_pipeline(video_path, out_dir, model_name, forward_gpu, to_tokens, meta, res=448)
```
- `forward_gpu(frame_uint8)` → CUDA 텐서 `(P, D)` (인코더 forward만; 타이밍용)
- `to_tokens(frame_uint8)`   → `np.float16 (P, D)` (CPU; delta 계산용)
- `meta` : 정적 정보(model_type, total/vision params, native_res, dtype …)

처리 순서:
1. `load_frames`로 영상을 448×448로 전부 디코딩
2. `time_forward`로 연산시간 측정
3. **스트리밍 delta**: 직전 프레임 토큰만 메모리에 유지하며 프레임쌍 delta 계산
   (SAM2=4096, SAM3=5184 토큰/프레임이라 전부 저장 불가 → 메모리 절약)
4. `frames.npz` 저장 → `run_analysis` 호출(분석1 + scene cut + 분석2 + 그래프)
5. `meta.json` 기록 (= 그 모델·영상의 모든 결과 + 스펙)

> **`gh, gw`(패치 그리드)** : meta에 있으면 사용, 없으면 토큰 수 P의 √로 추정.
> 픽셀 delta도 같은 그리드로 풀링하므로 모델마다 그리드가 달라도 영상 내 위치는 정렬됨.

### 3.7 `run_analysis(npz, out_dir, model_name)` — 분석1 + 그래프
- 컷 제외 평균 / 전체 평균 (`averages_excluding_cuts`, `averages_all`)
- **분석1 최종 통계량** (`analysis1_stats` → `summary["analysis1_correlation"]`):
  - **목적(정확한 해석)**: x·y가 (픽셀→풀링→임베딩 구조상) 어느 정도 연관되는 건 당연하므로,
    분석1은 "상관이 있냐"가 아니라 **"입력 프레임 변화가 encoder representation 변화로
    얼마나 *전달(transmit)*되는가"**, 즉 **encoder의 입력 변화 민감도**를 본다.
    높은 correlation = 입력 변화에 민감, 낮은 correlation = 변화 일부를 표현 공간에서 흡수/무시.
  - **축 척도 일치 (라벨링 불일치 해결)**: x·y는 **반드시 같은 거리 척도**를 써야 한다.
    이전 버전은 `x=픽셀-L2`, `y=임베딩-cos`로 **서로 다른 척도를 비교**하는 라벨링 불일치가
    있었다. 현재는 **matched pair**로 계산한다:
    - **primary: `x=픽셀 cosine-distance`, `y=임베딩 cosine-distance` (cos↔cos)**
    - secondary: `x=픽셀 L2`, `y=임베딩 L2` (L2↔L2) — `meta.json`에 함께 저장
  - **핵심 통계량 = Pearson r, Spearman ρ** (+R²). 축 척도가 일치하므로 **OLS slope도 이제
    해석 가능**(= *sensitivity gain* = d 임베딩-cos / d 픽셀-cos)하여 리포트에 gain으로 포함.
- 분석2 결과를 `summary["motion_concentration"]`에 포함
- 산출물:
  - `delta_summary.json`, `scene_cuts.json/.txt`, `selected_frames.json`
  - `timeseries_{l1,l2,cos}.png` (픽셀/임베딩 시계열, 모델 내 max 고정)
  - `timeseries_norm_{m}.png` (픽셀·임베딩 정규화 overlay)
  - `composite_{pix,emb}_{m}.png` (대표 프레임 히트맵)
  - `analysis2_concentration.png` (R vs area 막대)

---

## 4. 모델별 스크립트 패턴 (`ext_*.py`)

모든 스크립트가 동일한 골격:
```python
import common
DEV="cuda:0"; VIDS=json.load(open(".../videos.json"))

def main():
    model = <로딩>
    encoder = common.get_submodule(model, ["후보경로1","후보경로2"])  # vision tower 탐색

    def fwd(fr):  # 인코더 forward만 → GPU 텐서 (P,D)
        ...
        return tokens_on_gpu
    tok = lambda fr: fwd(fr).float().cpu().numpy().astype(np.float16)

    meta = dict(model_type=..., total_params=common.count_params(model),
                vision_encoder_params=common.count_params(encoder), native_res=..., dtype=...)
    for tag, info in VIDS.items():
        common.run_pipeline(info["path"], f"out/<model>/{tag}", NAME, fwd, tok, dict(meta))
```

### 모델별 임베딩 추출 지점 (가장 중요한 차이)

| 모델 | 종류 | 추출 지점 | 처리 해상도 | 그리드 / dim |
|---|---|---|---|---|
| Qwen3-VL-8B | LMM | `visual.merger`에 `forward_pre_hook` → **merger 입력**(merge 전 ViT 패치 토큰) | dynamic | 28×28 / 1152 |
| Qwen2.5-VL-7B | LMM | 동일 (`visual.merger` forward-pre-hook 입력) | dynamic | 동적 / 1280 |
| InternVL3-8B | LMM | `vision_model.last_hidden_state[:,1:]` (CLS 제외) | 448 | 32×32 / 1024 |
| InternVL2.5-8B | LMM | 동일 | 448 | 32×32 / 1024 |
| LLaVA-OV-0.5B | LMM | SigLIP `hidden_states[-2]` (config `mm_vision_select_layer=-2`, patch only; 체크포인트의 fine-tuned vision-tower 가중치 로드, anyres 우회·단일 타일) | 384 | 27×27 / 1152 |
| DINOv2-L/14 | VFM | `last_hidden_state[:,1:]` (interp pos, CLS 제외) | 448 | 32×32 / 1024 |
| CLIP-L/14 | VFM | `last_hidden_state[:,1:]` (CLS 제외) | 224 | 16×16 / 1024 |
| SAM2.1 Hiera-B+ | VFM | `image_encoder(...)["vision_features"]` | 1024 | 64×64 / 256 |
| SAM3 ViT | VFM | `backbone.vision_backbone.trunk` 최종 feature map | 1008 | 72×72 / 1024 |

> **Qwen dynamic resolution 처리 (중요)**: Qwen-VL은 입력 크기에 따라 patch grid가 변하지만,
> 본 파이프라인은 **모든 프레임을 동일한 `FIXED_RES=448`로 먼저 리사이즈한 뒤** 프로세서에
> 넣으므로 smart-resize 결과가 **영상 내내 일정**하다. 실측에서 **Qwen3-VL=28×28(784),
> Qwen2.5-VL=32×32(1024)로 1125/816프레임 전체 동일**했다. `run_pipeline`에는 프레임마다
> 토큰 수가 바뀌면 즉시 멈추는 `assert`(=patch 1:1 대응 보장)가 들어 있다. 즉 표의 그리드는
> "평균/최빈"이 아니라 **모든 프레임에서 동일한 고정값**이다.

> **모델 간 비교는 cosine-distance로 한다 (대표 metric).** 임베딩 절대 스케일(L1/L2)은
> **모델별 feature scale에 크게 의존**(예: Qwen emb-L2 ≈ 1500 vs SAM2 ≈ 0.5)하므로
> cross-model 비교가 불가능하다. **cosine만 scale-free**라 공정 비교가 가능하다.
> (L1/L2는 동일 모델 내부 시계열 비교용으로만 사용.)

> **해상도 공정성 (Method A — native resolution)**: 각 모델은 **실제로 사용하는 처리 해상도**
> (CLIP 224, DINOv2/InternVL 448, LLaVA-OV 384, SAM2 1024, SAM3 1008, Qwen dynamic)에서
> 측정한다. 모든 모델을 한 해상도로 강제(Method B)하지 않은 이유는, **실사용 환경의
> 연산 오버헤드를 그대로 반영**하기 위해서다. 단, 그 결과 **2절의 latency는 해상도·토큰 수가
> 다른 상태의 비교**이므로 절대 시간 직접 비교가 아니라 *모델별 native 운영 비용*으로 해석해야
> 한다 — 비교 가능성은 토큰 수(아래 표 `#tokens`)와 함께 보아야 공정하다.

### 모델별 토큰 수 (연산시간과 가장 직결되는 값)

| 모델 | #tokens | 모델 | #tokens |
|---|--:|---|--:|
| CLIP-L/14 (224) | 256 | InternVL3/2.5 (448) | 1024 |
| LLaVA-OV (384) | 729 | DINOv2-L/14 (448) | 1024 |
| Qwen3-VL | 784 (28×28, 동적) | SAM2.1 (1024) | 4096 |
| Qwen2.5-VL | 동적 | SAM3 (1008) | 5184 |

> 토큰 수 = `patch_grid[0]×patch_grid[1]` (= `meta["n_vision_tokens"]`). 토큰 수는 latency를
> 결정하는 **주요 요인**이지만 **유일한 요인은 아니다** — 실제 연산량은 **attention 구조
> (full vs windowed; 예: SAM2 Hiera는 윈도우 어텐션), encoder depth, hidden dimension**에 따라
> 달라진다. 따라서 latency(2절)는 토큰 수와 함께 해석하되 "토큰 수=시간"으로 단정하지 않는다.

---

## 5. 출력물 구조

```
out/<model>/<fast|static>/
  ├─ meta.json          # 스펙 + 타이밍 + delta_summary(분석1) + motion_concentration(분석2)
  ├─ frames.npz         # 스칼라/히트맵 원자료(재분석용)
  ├─ delta_summary.json
  ├─ scene_cuts.json / scene_cuts.txt / selected_frames.json
  └─ *.png              # 시계열·히트맵·분석2 그래프

results/REPORT.md       # make_tables.py가 생성하는 리포트:
                        #   1)  모델/인코더 스펙 (params·native/proc res·patch grid·#tokens·dim)
                        #   2)  인코더 연산시간 (fast/static, ms·fps)
                        #   3)  Temporal Redundancy Score (headline; TRS, static/fast/drop)
                        #   4a) 분석1 평균 delta (픽셀 vs 임베딩, 컷 제외)
                        #   4b) 분석1 최종 통계량 (Pearson r·Spearman ρ·R²; slope는 appendix)
                        #   5)  분석2 — primary metric = Concentration Factor (CF = R/area)
```

> Table 4·5 머리말에 **cosine-only cross-model 비교**·**분석1 해석(전달/민감도)**·**CF가
> primary metric**·**motion proxy 주의**가 명시되며, Table 3(TRS)가 headline 요약,
> Table 1의 `#tokens`·`proc res` 컬럼이 latency(Table 2) 해석의 기준이 된다.

---

## 6. 실행 방법

```bash
# (A) 개별 모델 — 빠른 테스트는 VDA_MAXF=N 으로 프레임 수 제한
VDA_MAXF=8 ~/miniconda3/envs/qwen3vl/bin/python ext_clip.py

# (B) 전체 — 9 모델 × 2 영상, 4 GPU 병렬 + 리포트 생성
bash run_all.sh        # 끝나면 results/REPORT.md 생성

# (C) 결과만 다시 취합
~/miniconda3/envs/qwen3vl/bin/python make_tables.py
```

`run_all.sh`의 GPU 배치(무거운 SAM을 분산):
- GPU0: SAM3 → CLIP · GPU1: SAM2 → DINOv2
- GPU2: Qwen3-VL → Qwen2.5-VL · GPU3: InternVL3 → InternVL2.5 → LLaVA-OV
