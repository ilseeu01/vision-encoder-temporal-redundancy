# Vision-encoder overhead & temporal-redundancy analysis

models: 9  | videos: fast (downhill MTB POV) vs static (locked forest)


## 1. Model & vision-encoder spec

| model | type | total params | vision-enc params | dtype | native res | proc res | patch grid | #tokens | patch dim |
|---|---|--:|--:|---|---|--:|:--:|--:|--:|
| CLIP-L/14 | VFM | 303M | 303M | bf16 | 224 | 448 | 16x16 | 256 | 1024 |
| DINOv2-L/14 | VFM | 304M | 304M | bf16 | 518 default; here 448 w/ interp pos-enc | 448 | 32x32 | 1024 | 1024 |
| InternVL2.5-8B | LMM | 8075M | 304M | bf16 | 448 (single tile) | 448 | 32x32 | 1024 | 1024 |
| InternVL3-8B | LMM | 7944M | 304M | bf16 | 448 (single tile) | 448 | 32x32 | 1024 | 1024 |
| LLaVA-OneVision-0.5B | LMM | 894M | 428M | bf16 | 384 (single tile, anyres bypassed) | 448 | 27x27 | 729 | 1152 |
| Qwen2.5-VL-7B | LMM | 8292M | 677M | bf16 | dynamic (Qwen smart-resize) | 448 | 32x32 | 1024 | 1280 |
| Qwen3-VL-8B | LMM | 8767M | 576M | bf16 | dynamic (Qwen smart-resize) | 448 | 28x28 | 784 | 1152 |
| SAM2.1-Hiera-B+ | VFM | 81M | 69M | fp32 | 1024 | 448 | 64x64 | 4096 | 256 |
| SAM3-ViT | VFM | 841M | 446M | bf16-autocast | 1008 | 448 | 72x72 | 5184 | 1024 |

## 2. Vision-encoder latency (per frame, single image, GPU)

| model | fast ms/frame | fast fps | static ms/frame | static fps |
|---|--:|--:|--:|--:|
| CLIP-L/14 | 10.56±1.23 | 94.7 | 10.14±1.21 | 98.6 |
| DINOv2-L/14 | 35.34±7.03 | 28.3 | 35.08±6.80 | 28.5 |
| InternVL2.5-8B | 20.33±0.92 | 49.2 | 20.16±1.60 | 49.6 |
| InternVL3-8B | 20.02±1.41 | 49.9 | 20.97±0.81 | 47.7 |
| LLaVA-OneVision-0.5B | 18.04±0.87 | 55.4 | 17.57±0.25 | 56.9 |
| Qwen2.5-VL-7B | 50.97±0.65 | 19.6 | 53.38±14.93 | 18.7 |
| Qwen3-VL-8B | 23.62±0.24 | 42.3 | 24.13±1.09 | 41.5 |
| SAM2.1-Hiera-B+ | 147.55±16.46 | 6.8 | 132.16±16.02 | 7.6 |
| SAM3-ViT | 114.90±0.24 | 8.7 | 115.84±0.39 | 8.6 |

## 3. Temporal Redundancy Score (headline)

**TRS = 1 - mean(embedding cosine-distance) over non-cut adjacent frame pairs.**

TRS -> 1 means consecutive frames produce nearly identical encoder representations (high temporal redundancy => strong case for reuse/caching). cos is scale-free, so TRS is comparable across models. This is the bottom-line 'how much redundancy is there?'.

| model | type | TRS (static) | TRS (fast) | drop (static-fast) |
|---|---|--:|--:|--:|
| CLIP-L/14 | VFM | 0.9716 | 0.6382 | 0.3334 |
| DINOv2-L/14 | VFM | 0.9925 | 0.6991 | 0.2934 |
| InternVL2.5-8B | LMM | 0.9925 | 0.8144 | 0.1782 |
| InternVL3-8B | LMM | 0.9767 | 0.5896 | 0.3871 |
| LLaVA-OneVision-0.5B | LMM | 0.9746 | 0.5668 | 0.4078 |
| Qwen2.5-VL-7B | LMM | 0.9926 | 0.9624 | 0.03018 |
| Qwen3-VL-8B | LMM | 0.9957 | 0.972 | 0.02372 |
| SAM2.1-Hiera-B+ | VFM | 0.9947 | 0.6398 | 0.3549 |
| SAM3-ViT | VFM | 0.9655 | 0.543 | 0.4225 |

## 4. Analysis 1 — does pixel change predict embedding change?

Purpose: measure how strongly an input-frame change is *transmitted* into the encoder representation. High correlation = encoder is sensitive to input change; low = encoder absorbs/ignores part of the change in its representation space.

**Cross-model comparison uses cosine-distance only: L1/L2 depend on each model's feature scale (e.g. Qwen emb-L2 ~1500 vs SAM2 ~0.5) and are NOT comparable.**

### 4a. Mean adjacent-frame delta (cuts excluded)

| model | video | pix cos | emb cos | pix L2 | emb L2 (own scale) | #cuts |
|---|---|--:|--:|--:|--:|--:|
| CLIP-L/14 | fast | 0.004807 | 0.3618 | 0.08312 | 21.31 | 0 |
| CLIP-L/14 | static | 0.0002092 | 0.02837 | 0.005298 | 4.29 | 11 |
| DINOv2-L/14 | fast | 0.004807 | 0.3009 | 0.08312 | 31.66 | 0 |
| DINOv2-L/14 | static | 0.0002092 | 0.007547 | 0.005298 | 3.792 | 11 |
| InternVL2.5-8B | fast | 0.004807 | 0.1856 | 0.08312 | 19.37 | 0 |
| InternVL2.5-8B | static | 0.0002092 | 0.007457 | 0.005298 | 2.707 | 11 |
| InternVL3-8B | fast | 0.004807 | 0.4104 | 0.08312 | 19.94 | 0 |
| InternVL3-8B | static | 0.0002092 | 0.02327 | 0.005298 | 3.011 | 11 |
| LLaVA-OneVision-0.5B | fast | 0.004388 | 0.4332 | 0.08126 | 57.93 | 0 |
| LLaVA-OneVision-0.5B | static | 0.0002041 | 0.02543 | 0.005299 | 11.45 | 11 |
| Qwen2.5-VL-7B | fast | 0.004807 | 0.03757 | 0.08312 | 4703 | 0 |
| Qwen2.5-VL-7B | static | 0.0002092 | 0.007393 | 0.005298 | 1876 | 11 |
| Qwen3-VL-8B | fast | 0.004807 | 0.02804 | 0.08312 | 4021 | 0 |
| Qwen3-VL-8B | static | 0.0002092 | 0.004322 | 0.005298 | 1022 | 11 |
| SAM2.1-Hiera-B+ | fast | 0.004807 | 0.3602 | 0.08312 | 3.77 | 0 |
| SAM2.1-Hiera-B+ | static | 0.0002092 | 0.005292 | 0.005298 | 0.2994 | 11 |
| SAM3-ViT | fast | 0.004388 | 0.457 | 0.08126 | 39.57 | 0 |
| SAM3-ViT | static | 0.0002041 | 0.03451 | 0.005299 | 6.85 | 11 |

### 4b. Final statistic — correlation of per-frame **pixel-cos vs embedding-cos** delta

**Both axes use the SAME distance metric (cosine-distance)** so x and y are consistent — no L2-vs-cos labeling mismatch. With matched units the OLS slope is interpretable as a *sensitivity gain* (d embedding-cos / d pixel-cos). Pearson r and Spearman rho are the core statistics. (pixL2-vs-embL2 also stored in meta.json.)

| model | video | Pearson r | Spearman rho | R^2 | gain (slope) |
|---|---|--:|--:|--:|--:|
| CLIP-L/14 | fast | 0.192 | 0.1688 | 0.03687 | 4.332 |
| CLIP-L/14 | static | 0.4814 | 0.626 | 0.2318 | 27.51 |
| DINOv2-L/14 | fast | 0.3229 | 0.4034 | 0.1043 | 7.415 |
| DINOv2-L/14 | static | 0.7271 | 0.7413 | 0.5286 | 21.32 |
| InternVL2.5-8B | fast | 0.1216 | 0.1784 | 0.01479 | 1.532 |
| InternVL2.5-8B | static | 0.651 | 0.7687 | 0.4238 | 18.45 |
| InternVL3-8B | fast | 0.1313 | 0.1485 | 0.01724 | 3.339 |
| InternVL3-8B | static | 0.621 | 0.771 | 0.3857 | 39.47 |
| LLaVA-OneVision-0.5B | fast | 0.1507 | 0.1997 | 0.0227 | 3.763 |
| LLaVA-OneVision-0.5B | static | 0.7103 | 0.7369 | 0.5046 | 45.43 |
| Qwen2.5-VL-7B | fast | -0.03971 | -0.05904 | 0.001577 | -0.08575 |
| Qwen2.5-VL-7B | static | 0.3142 | 0.4273 | 0.0987 | 6.343 |
| Qwen3-VL-8B | fast | -0.01816 | -0.1429 | 0.0003297 | -0.03109 |
| Qwen3-VL-8B | static | 0.3634 | 0.4583 | 0.1321 | 3.501 |
| SAM2.1-Hiera-B+ | fast | 0.2821 | 0.3831 | 0.07959 | 8.233 |
| SAM2.1-Hiera-B+ | static | 0.9088 | 0.8093 | 0.8259 | 25.4 |
| SAM3-ViT | fast | 0.305 | 0.4329 | 0.09302 | 7.628 |
| SAM3-ViT | static | 0.795 | 0.7864 | 0.632 | 61.85 |

## 5. Analysis 2 — is embedding change concentrated in the motion region?

Motion patches = pixel-L2 delta above the p90 (top-10%) threshold. Note: this is a **pixel-difference motion proxy**, not optical flow — global brightness/exposure shifts can also raise pixel-L2 (acceptable here as fast/static differ mainly in real motion).

**Primary metric = Concentration Factor (CF) = R / area.** R alone grows with the number of motion patches, so it is normalised by area: CF > 1 means motion patches carry MORE embedding change than their area share (=> static patches are redundant, cacheable). CF ~ 1 means change is spread uniformly.

(cos used for cross-model comparability; area = motion-patch fraction.)

| model | video | area | R(cos) | **CF(cos)** | CF(L2) |
|---|---|--:|--:|--:|--:|
| CLIP-L/14 | fast | 0.1025 | 0.1137 | **1.109** | 0.9719 |
| CLIP-L/14 | static | 0.1003 | 0.1565 | **1.561** | 1.292 |
| DINOv2-L/14 | fast | 0.1019 | 0.1401 | **1.374** | 1.207 |
| DINOv2-L/14 | static | 0.1 | 0.2716 | **2.716** | 1.787 |
| InternVL2.5-8B | fast | 0.1019 | 0.1245 | **1.222** | 1.285 |
| InternVL2.5-8B | static | 0.1 | 0.2517 | **2.517** | 1.751 |
| InternVL3-8B | fast | 0.1019 | 0.1173 | **1.151** | 1.25 |
| InternVL3-8B | static | 0.1 | 0.2168 | **2.168** | 1.678 |
| LLaVA-OneVision-0.5B | fast | 0.1019 | 0.1112 | **1.091** | 1.063 |
| LLaVA-OneVision-0.5B | static | 0.1001 | 0.1888 | **1.886** | 1.447 |
| Qwen2.5-VL-7B | fast | 0.1019 | 0.1005 | **0.9863** | 0.9858 |
| Qwen2.5-VL-7B | static | 0.1 | 0.1138 | **1.138** | 1.037 |
| Qwen3-VL-8B | fast | 0.1018 | 0.1035 | **1.016** | 0.9898 |
| Qwen3-VL-8B | static | 0.1 | 0.1153 | **1.153** | 1.094 |
| SAM2.1-Hiera-B+ | fast | 0.1008 | 0.1803 | **1.789** | 1.426 |
| SAM2.1-Hiera-B+ | static | 0.1 | 0.3109 | **3.109** | 2.054 |
| SAM3-ViT | fast | 0.1007 | 0.1164 | **1.156** | 1.049 |
| SAM3-ViT | static | 0.1 | 0.1866 | **1.866** | 1.536 |
