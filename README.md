# Vision Encoder Temporal Redundancy Analysis

비디오를 LMM/VFM에 더 효율적으로 넣기 위한(인코더 출력 재사용·캐싱) 사전 분석.
인접 프레임(t vs t−1) 사이에서 **픽셀**과 **vision encoder 임베딩**이 얼마나 변하는지를
**L1 / L2 / cosine** 으로 측정하고, 9개 모델을 비교한다.

## 동기
비디오는 인접 프레임이 매우 비슷하다. 만약 인코더 출력이 프레임 간 거의 변하지 않는다면
이전 프레임 결과를 재사용해 vision encoder 연산을 줄일 수 있다. 이 저장소는 그 **재사용
가능성**이 실제 데이터에서 얼마나 뒷받침되는지를 세 가지 분석으로 정량화한다.

- **분석 0** — 모델별 vision encoder 연산 시간(오버헤드) 측정
- **분석 1** — 인접 프레임의 픽셀 변화와 인코딩 변화의 평균 비교 및 상관/민감도
- **분석 2** — 움직임이 큰 영역(상위 10% 픽셀 변화)에 인코딩 변화가 집중되는 정도(CF)

## 실험 설정
- **영상 2종** (각 1분 이내, 854×480 → 448×448 통일)
  - `fast` — 다운힐 산악자전거 1인칭 시점(빠른 움직임), 45.0초
  - `static` — 고정 카메라 풍경(거의 정지), 32.7초
- **회색조 변환** — Rec.601 휘도 `Y = 0.299R + 0.587G + 0.114B` (cosine 대신 L1/L2 비교용)

## 대상 모델 (9)
| 모델 | 종류 | 비전 인코더 측정 범위 | 연산시간 정적 (ms/frame) |
|---|---|---|---|
| CLIP-L/14 | VFM | 비전 타워 전체 (CLS 제외) | 9.6 |
| DINOv2-L/14 | VFM | ViT-L/14 백본 전체 | 28.1 |
| SAM2.1-Hiera-B+ | VFM | 이미지 인코더(Hiera+neck) | 102.6 |
| SAM3-ViT | VFM | 이미지 모델 ViT trunk | 115.6 |
| InternVL2.5-8B | LMM | InternViT-300M | 20.8 |
| InternVL3-8B | LMM | InternViT-300M | 19.4 |
| LLaVA-OneVision-0.5B | LMM | SigLIP-so400m/14 (−2층) | 17.4 |
| Qwen2.5-VL-7B | LMM | 비전 ViT (merger 직전) | 52.9 |
| Qwen3-VL-8B | LMM | 비전 ViT (merger 직전) | 23.8 |

## 평가 지표 (`common.py`)
프레임쌍(t−1, t)마다 픽셀과 임베딩을 같은 척도로 측정한다.
- **픽셀 변화**: 회색조 휘도 차이를 화면 전체 화소에 대해 평균. 회색조는 채널이 하나라 **L1 = L2**.
- **임베딩 변화 L1**: 패치 토큰 임베딩의 차원별 절대차 평균을 패치 전체에 대해 평균.
- **임베딩 변화 L2**: 두 임베딩 벡터의 유클리드 거리를 패치 전체에 대해 평균.
- 각 지표는 (patch grid) heatmap과 스칼라(패치 평균) 둘 다 산출. cosine은 distance(1−cos)로 기록.

## 핵심 결과 — 분석 1 (회색조, 공유축 오버레이)
세로축은 **임베딩 변화 ÷ 그 모델의 평균 임베딩 변화**로 정규화해(무차원), 임베딩 절대
크기가 수천 배씩 다른 9개 모델을 같은 축에서 비교한다. 가로축은 회색조 픽셀 변화(모든
모델 동일). `static`(파랑)은 원점 근처, `fast`(주황)는 넓게 퍼지며, OLS선 기울기와 r값으로
입력 변화가 표현 변화로 얼마나 전달되는지를 본다.

**L1 기준**
![Analysis 1 grayscale shared-axis overlay (L1)](results/gray_overlay_shared_l1.png)

**L2 기준**
![Analysis 1 grayscale shared-axis overlay (L2)](results/gray_overlay_shared_l2.png)

> 패널별 자체 스케일(정규화 전, raw) 버전은 `results/gray_analysis1_scatter_l1.png`,
> `results/gray_analysis1_scatter_l2.png` 에 있다.

정적 영상에서 SAM2.1(r≈0.90)·DINOv2·InternVL 계열이 높은 상관을 보여, 작은 입력 변화가
표현에 충실히 전달된다. 또한 움직임이 없는 패치는 인코딩 변화가 거의 없어(분석 2의 높은
집중도) 정적 영역의 인코더 출력을 재사용하는 캐싱이 유망함을 시사한다.

## 저장소 구조
```
common.py              # 핵심 지표·파이프라인 (픽셀/임베딩 L1·L2·cos, 컷 검출, 집중도)
ext_*.py               # 모델별 임베딩 추출기 (clip/dinov2/sam2/sam3/intern/qwen25/qwen3/llava_ov)
run_all.sh             # RGB 기준 9개 모델 × 2영상 전체 실행 (4 GPU 분배)
run_all_gray.sh        # 회색조(VDA_GRAY=1) 기준 전체 실행
aggregate_gray.py      # out_gray/ 결과 → results/gray_summary.json 집계
plot_gray_overlay_shared.py  # 회색조 공유축 오버레이 (위 그래프)
plot_gray_scatter.py         # 회색조 패널별 산점도
plot_analysis1_overlay.py    # RGB·cosine 오버레이
make_tables.py         # 요약 표 생성
results/               # 그래프·요약(JSON)·리포트
ARCHITECTURE.md        # 파이프라인 상세 설계
```
대용량 산출물(`out/`, `out_gray/`, `cache/`, `videos/`)은 `.gitignore`로 제외된다.

## 재현
```bash
# 회색조 전체 실행 (모델별 conda env 필요: qwen3vl / internvl / sam2 / sam3)
bash run_all_gray.sh
# 집계 및 그래프
python aggregate_gray.py
python plot_gray_overlay_shared.py
```
