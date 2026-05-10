# 09th-gpu-deep-dive

## 01. 스터디 소개
- ChatGPT, Claude 등 AI 서비스들이 동작하기 위한 GPU 하드웨어 구조 및 네트워크를 학습하고, LLM Inference 구조 이해 및 병목 발생 지점, 로컬 LLM(SLM) 서빙, 하드웨어별 튜닝, 양자화 등을 단계적으로 학습합니다. 이를 통해 AI 인프라와 모델에 대한 이해를 높이는 것이 목적입니다.
- 단순히 모델 서빙에서 끝나는 것이 아니라 AI 인프라를 이해하고 각자 보유한 하드웨어에서 모델이 어떻게 동작하는지 관찰 및 병목 지점을 추론함으로써 최적화하는 방법을 학습합니다.

---

## 02. 스터디 목표
- GPU가 AI 워크로드에서 중요한 이유 이해
- GPU 서버 내부 구조와 데이터 이동 흐름 이해
- InfiniBand, RoCE, NCCL 등 AI 네트워크 기본 개념 이해
- LLM 추론 과정과 병목 지점 이해
- GPU, NPU, TPU, Trainium 등 AI 가속기 설계 철학 비교
- 로컬 환경에서 모델을 서빙하고 성능 측정 및 튜닝 수행

---

## 03. 대상
- GPU, HPC, AI 인프라에 관심 있는 분
- GPU 시스템 구조를 이해하고 싶은 분
- LLM 서빙 구조와 병목을 하드웨어 관점에서 이해하고 싶은 사람
- InfiniBand, RoCE, RDMA, NCCL 같은 AI 인프라 네트워크에 관심 있는 분
- 로컬에서 AI 모델을 직접 실행하고 최적화해보고 싶은 분
- NVIDIA, AMD, Apple Silicon, CPU 등 다양한 하드웨어 환경에서 모델 성능을 비교해보고 싶은 분

---

## 04. 커리큘럼
| 주차 | 주제 | 핵심 목표 |
|---|---|---|
| Week 1 | OT 및 환경 공유 | 스터디 방향, 개인별 실습 환경, 진행 방식 정리 |
| Week 2 | GPU 아키텍처와 병렬 연산 구조 이해 | GPU 내부 구조와 병렬 연산 흐름 이해 |
| Week 3 | GPU 서버 시스템 구조 이해 | CPU, Memory, PCIe, GPU, NIC 연결 구조 이해 |
| Week 4 | AI 워크로드 네트워크 이해 | 멀티 GPU/멀티 노드 환경의 네트워크 구조 이해 |
| Week 5 | LLM 추론 구조와 병목 이해 | Token 생성 과정과 GPU/메모리/네트워크 병목 이해 |
| Week 6 | AI 가속기 이해 및 비교 | GPU 외 다양한 AI 가속기의 설계 철학 비교 |
| Week 7 | 기본 모델 서빙 및 벤치마크 | 공통 런타임으로 모델 실행 및 기준 성능 측정 |
| Week 8 | 하드웨어별 서빙 튜닝 및 벤치마크 | 각자 하드웨어에 맞는 튜닝 및 성능 비교 |

---

## 05. 주차별 상세 내용

### Week 2. GPU 아키텍처와 병렬 연산 구조 이해

GPU가 AI 워크로드에서 사용되는 이유를 CPU와 GPU의 구조적 차이 관점에서 이해합니다.

- CPU와 GPU 구조적 차이
- GPU 메모리 계층 구조
- GPU 실행 단위
- GPU 내부 동작 흐름
- Memory Bandwidth와 Compute Throughput 차이


### Week 3. GPU 서버 시스템 구조 이해

GPU 하나만 보는 것이 아니라, 서버 내부에서 CPU, Memory, PCIe, GPU, NIC가 어떻게 연결되는지 이해합니다.

- x86 서버 기본 구조
- CPU Socket, System Memory, PCIe Root Complex, PCIe Switch
- NUMA
- CPU-GPU 데이터 이동
- Host Memory와 GPU Memory
- PCIe, NVLink, NVSwitch
- GPU와 NIC의 물리적 위치가 성능에 미치는 영향

### Week 4. AI 워크로드 네트워크 이해

여러 GPU와 여러 서버를 연결할 때 필요한 네트워크 구조와 프로토콜을 이해합니다.

- AI 워크로드에서 네트워크가 중요한 이유
- 기존 네트워크 방식이 AI 워크로드에 부적절한 이유
- GPU 서버 내부 통신과 서버 간 통신의 차이
- RDMA
- InfiniBand
- RoCE
- NCCL
- Collective Communication
  - AllReduce
  - AllGather
  - ReduceScatter
  - All-to-All
- LLM 학습과 추론에서 네트워크 병목이 발생하는 지점


### Week 5. LLM 추론 구조와 병목 이해

LLM이 실제로 Token을 생성하는 과정을 이해하고, GPU/메모리/네트워크 관점에서 병목을 분석합니다.

- LLM 추론 흐름
- Transformer
- Prefill
- Decode
- KV Cache
- Context Length
- Batch Size
- Concurrent Request
- TTFT, TPOT, Token/sec
- 단일 노드 추론 병목
- 멀티 노드 추론 병목


### Week 6. AI 가속기 이해 및 비교

LLM 추론 병목을 기준으로 다양한 AI 가속기의 설계 철학과 하드웨어 구조를 비교합니다.

- AMD GPU
- NPU
  - FuriosaAI
  - Rebellions
- Google TPU
- AWS Trainium & Inferentia
- Tenstorrent


### Week 7. 실습 환경 구성 및 기본 모델 서빙, 벤치마크

공통 런타임을 이용해 로컬에서 모델을 실행하고, 최적화 전 기준 성능을 측정합니다.

- 실습 환경 구성
  - OS
  - GPU Software Stack
  - Hugging Face
  - llama.cpp
  - nvitop
  - DCGM, dcgm-exporter
- Model Pull & Serving
- 성능 측정
  - TTFT
  - TPOT
  - Token/sec
  - Total Latency
  - VRAM 사용량
  - GPU 사용률


### Week 8. 하드웨어별 서빙 튜닝 및 벤치마크

사전 학습한 내용을 기반으로 각자 하드웨어 특성에 맞는 튜닝 옵션을 적용하고 성능 차이를 측정합니다.

- GPU Offload 조정
- Context Length 조정
- Batch Size 조정
- Output Length 조정
- Concurrency 조정
- 튜닝 전/후 성능 비교
- 하드웨어별 결과 공유