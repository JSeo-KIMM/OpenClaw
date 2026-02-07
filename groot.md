# NVIDIA Isaac GR00T N1.5 ("GR00T-N1.5") 정리

> 이 문서는 **GR00T N1.5**에 대해 공개된 자료를 바탕으로 핵심을 요약한 것입니다. (2026-02-03 기준)

## 1) 한 줄 요약
**NVIDIA Isaac GR00T N1.5**는 **언어+시각+로봇 상태(고유감각)** 입력을 받아 **로봇 행동(action)을 생성**하는 **VLA(Vision-Language-Action) 기반 로봇 파운데이션 모델**이며, 다양한 로봇 형태(embodiment)에 대해 **후학습(post-training/finetuning)**으로 적응하도록 설계되었습니다.

## 2) GR00T N1 / N1.5 / N1.6 관계
- GR00T N1: 논문(아키텍처/학습 개요)을 통해 공개된 기반 모델 라인업의 출발점에 해당합니다. (VLA + diffusion 기반 action 생성)
- GR00T N1.5: N1의 업데이트 버전(3B 모델이 공개 체크포인트로 제공됨)
- GR00T N1.6: GitHub 리포지토리 기준 최신 버전. N1.5 대비 구조/데이터/성능 개선을 강조하며, N1.5는 별도 브랜치에서 사용하도록 안내합니다.

참고:
- GitHub 리포지토리: N1.6이 기본, **N1.5는 `n1.5-release` 브랜치** 사용 권장
  - https://github.com/NVIDIA/Isaac-GR00T

## 3) 모델이 하는 일(입출력)
Hugging Face 모델 카드 기준(N1.5-3B):
- 입력
  - Vision: 로봇 카메라 이미지 프레임(여러 뷰 가능)
  - State: 로봇 proprioception(관절각/센서 등 수치)
  - Language: 텍스트 명령
- 출력
  - 연속값 벡터 형태의 **행동(action)**

출처: Hugging Face model card
- https://huggingface.co/nvidia/GR00T-N1.5-3B

## 4) 아키텍처(개념)
### (A) N1 계열(VLA + diffusion action) 개요
arXiv 논문(N1)에서는:
- **System 2(vision-language)**가 환경/지시를 이해
- **System 1(diffusion transformer)**가 실시간으로 연속적인 로봇 동작(action)을 생성
- 두 모듈을 end-to-end로 결합/학습한 **Vision-Language-Action (VLA)** 구조를 설명합니다.

출처: arXiv (GR00T N1)
- https://arxiv.org/abs/2503.14734

### (B) N1.5-3B 모델 카드에 나온 구현 디테일(요약)
Hugging Face 모델 카드에는 N1.5-3B가:
- 시각 인코더(예: SigLIP2) / 텍스트 인코더(예: T5)로 관측/지시를 임베딩
- proprioception과 action 시퀀스를 조건부로 모델링하기 위해 **flow-matching transformer(= diffusion transformer 계열)**을 사용
- 학습 시 액션에 노이즈를 섞고, 추론 시 노이즈에서 점진적으로 action을 복원하는 형태를 설명합니다.

출처: Hugging Face model card
- https://huggingface.co/nvidia/GR00T-N1.5-3B

## 5) 공개 체크포인트/라이선스
- 공개 체크포인트: **`nvidia/GR00T-N1.5-3B`**
- 라이선스/이용 조건: Hugging Face 모델 카드에 따르면 **비상업적(non-commercial) 사용 준비됨**으로 표기되어 있으며, NVIDIA 라이선스 문서를 링크합니다.

출처:
- https://huggingface.co/nvidia/GR00T-N1.5-3B

## 6) 실사용 관점: LeRobot와의 연결(왜 중요?)
GitHub 리포지토리에서는 GR00T 사용 흐름을 다음처럼 안내합니다(요약):
1) (비디오, 상태, 액션) 형태의 시연 데이터(demonstration) 수집
2) 이를 **LeRobot 데이터 스키마**로 변환
3) 사전학습 모델로 **zero-shot 평가** 또는 **fine-tuning**
4) 최종적으로 로봇 컨트롤러에 policy를 연결해 실제 로봇 구동

출처:
- https://github.com/NVIDIA/Isaac-GR00T

## 7) 무엇을 “정리/자동화”할 수 있나(예시)
OpenClaw + GR00T 리서치 워크플로우 관점에서 가능한 작업 예:
- 자료 수집: N1.5 관련 공식 링크/모델카드/브랜치/설치법 모아서 문서화
- 설치 체크리스트 생성: (CUDA/드라이버/uv/환경) 등 단계별로 정리
- LeRobot 데이터 준비 문서화: 데이터 포맷/변환 스크립트 흐름 요약
- 실험 로그 템플릿: 실험 설정(embodiment, action horizon, camera views, seed 등) 표준화

## 8) 참고 링크(원문)
- GitHub (Isaac-GR00T): https://github.com/NVIDIA/Isaac-GR00T
  - N1.5 브랜치: https://github.com/NVIDIA/Isaac-GR00T/tree/n1.5-release
- Hugging Face (GR00T-N1.5-3B): https://huggingface.co/nvidia/GR00T-N1.5-3B
- arXiv (GR00T N1 논문): https://arxiv.org/abs/2503.14734
- NVIDIA 개발자 페이지(프로젝트 허브): https://developer.nvidia.com/isaac/gr00t

---

## 다음에 더 채우면 좋은 것(원하면 내가 추가로 조사해서 업데이트 가능)
- 연구 페이지(Research NVIDIA labs)의 N1.5 전용 소개 문서 핵심 요약
- N1.5와 N1.6의 차이점 표(아키텍처/데이터/성능/추론 속도/권장 하드웨어)
- 실제 설치/실행(LeRobot + GR00T) 최소 예제(MVP) 가이드
