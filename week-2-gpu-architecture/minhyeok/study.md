# SM 내부 구조와 Roofline Model

인트로: https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html

# Part 1. SM 내부 구조 심화

## 1. SM 전체 구조 다시 보기

Week 1에서 SM(Streaming Multiprocessor)이 GPU의 핵심 연산 단위라는 것을 배웠다. H100 SXM 기준 132개의 SM이 있고, 각 SM 안에 128개의 CUDA Core가 있다는 것까지 다뤘다.
> 출처: [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core), p.25 — "A full GH100 GPU ... 144 SMs ... The H100 SXM5 GPU ... includes 132 SMs."

하지만 SM 안에는 CUDA Core만 있는 것이 아니다. SM 내부는 여러 종류의 연산 유닛, 스케줄러, 메모리가 정교하게 배치된 하나의 작은 프로세서다. 이번 장에서는 이 내부를 열어본다.

## 2. Processing Block: SM의 내부 분할

SM은 내부적으로 **4개의 Processing Block(= Sub-partition, Quad)** 으로 나뉜다. 이 구조는 Volta(V100) 이후 모든 NVIDIA 데이터센터 GPU에서 유지되고 있다.
> 출처: [NVIDIA V100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), p.14 — "Each SM is partitioned into four processing blocks" / [NVIDIA A100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), p.18 — 동일 구조 확인.

A100 기준 SM 하나의 내부 구조:

```
SM (1개)
│
├── Processing Block 0
│   ├── Warp Scheduler (1개)
│   ├── Dispatch Unit (1개)
│   ├── Register File: 16,384 × 32-bit
│   ├── FP32 CUDA Core: 16개
│   ├── INT32 CUDA Core: 16개
│   ├── FP64 CUDA Core: 8개
│   ├── Tensor Core: 1개 (3세대)
│   ├── Load/Store Unit: 8개
│   └── SFU: 4개
│
├── Processing Block 1  (동일 구성)
├── Processing Block 2  (동일 구성)
├── Processing Block 3  (동일 구성)
│
├── Shared Memory / L1 Cache: 192 KB (4개 블록이 공유)
└── L1 Instruction Cache
```

> 출처: [NVIDIA A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), p.18-22 — SM 내부 Processing Block 구성 상세. Figure 6 "NVIDIA GA100 Streaming Multiprocessor (SM)" 참조.

SM 전체 합산 (A100 기준):
- FP32 CUDA Core: 16 × 4 = **64개/SM**
- INT32 CUDA Core: 16 × 4 = **64개/SM**
- FP64 CUDA Core: 8 × 4 = **32개/SM**
- Tensor Core: 1 × 4 = **4개/SM**
- Warp Scheduler: **4개/SM**

> 출처: 동일 Whitepaper p.18 — "The A100 SM includes ... 64 FP32 CUDA Cores, 64 INT32 CUDA Cores, 32 FP64 CUDA Cores, 4 Tensor Cores, and 4 Warp Schedulers."

**왜 4개로 나누는가?** — SM 전체를 하나의 스케줄러가 관리하면, 매 사이클 수십 개의 Warp 중 하나를 골라 수십 개의 연산 유닛에 명령을 보내야 한다. 이 선택과 배선이 복잡해지면 클럭 속도를 올리기 어렵다. 4개로 분할하면 각 스케줄러가 자기 블록의 연산 유닛만 관리하면 되므로, 제어 로직이 단순해지고 클럭을 높일 수 있다.

---

## 3. 각 연산 유닛의 역할

### 3.1 FP32 CUDA Core

가장 기본적인 부동소수점 연산 유닛이다. 매 클럭마다 하나의 FP32 FMA(Fused Multiply-Add)를 수행한다. FMA는 `a × b + c`를 한 번에 처리하며, 곱셈 1회 + 덧셈 1회 = **2 FLOPs**로 계산한다.
> 출처: [CUDA C++ Programming Guide — Arithmetic Instructions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#arithmetic-instructions) — FMA throughput per SM 정의.

CUDA 커널에서 `float` 타입의 덧셈, 곱셈, 비교 등이 이 유닛에서 실행된다.

### 3.2 INT32 CUDA Core

정수 연산 전용 유닛이다. "왜 정수 연산에 별도 유닛이 필요한가?"라는 질문이 자연스러운데, 답은 **GPU 커널에서 정수 연산이 생각보다 훨씬 많기 때문**이다.

행렬 곱셈 커널 하나를 보면:

```c
__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // ← INT32: 인덱스 계산
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // ← INT32: 인덱스 계산
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {                       // ← INT32: 루프 카운터, 비교
        sum += A[row * K + k] * B[k * N + col];         // ← FP32: 곱셈+덧셈
        //       ^^^^^^^^^^^     ^^^^^^^^^^^
        //       INT32: 주소 계산  INT32: 주소 계산
    }
    C[row * N + col] = sum;                             // ← INT32: 주소 계산
}
```

FP32 연산 1번 할 때마다 INT32 연산이 3~5번 따라다닌다. 배열 인덱스 계산, 루프 카운터 증가, 포인터 산술 — 전부 INT32다.

#### Pre-Volta: 시분할 구조

Volta 이전에는 하나의 CUDA Core가 FP32와 INT32를 번갈아 처리했다. 두 연산 사이에 데이터 의존성이 있으면 직렬화된다.

> 출처: [NVIDIA Volta Tuning Guide](https://docs.nvidia.com/cuda/volta-tuning-guide/) — "Unlike Pascal GPUs, the GV100 SM includes dedicated FP32 and INT32 cores."
> 출처: [NVIDIA V100 Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), p.15 — "Many applications have inner loops that perform pointer arithmetic (integer memory address calculations) combined with floating-point computations that will benefit from simultaneous execution of FP32 and INT32 instructions."

#### Volta 이후: 분리 데이터패스

Volta부터 FP32와 INT32가 별도의 유닛으로 분리되어 **동시 실행**이 가능해졌다:

```
사이클 1: FP32 (현재 반복 곱셈)  ||  INT32 (다음 반복 주소 계산)
사이클 2: FP32 (현재 반복 곱셈)  ||  INT32 (다음 반복 주소 계산)
...
```

한 Warp가 현재 반복의 FP32 곱셈을 하는 동안, 다음 반복의 INT32 인덱스 계산을 미리 처리한다.

> 출처: [NVIDIA V100 Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), p.15 — "Each iteration of a pipelined loop can update addresses (INT32 pointer arithmetic) and load data for the next iteration while simultaneously processing the current iteration in FP32."

#### 트레이드오프

분리는 공짜가 아니다. INT32 ALU를 별도로 추가하면 다이 면적과 전력 소비가 늘어난다. Pre-Volta 시대에는 면적 절약이 우선이었지만, 딥러닝이 메인 워크로드가 되면서 인덱싱이 빈번한 GEMM, Convolution 패턴이 주류가 되었고, 분리의 이득이 비용을 초과하게 됐다.

흥미로운 점은 컨슈머 GPU(RTX 50 시리즈, Blackwell 기반)에서는 모든 CUDA Core가 FP32 또는 INT32를 처리할 수 있지만, 같은 클럭 사이클에는 하나의 타입만 실행 가능한 구조로 되어 있다는 것이다.

> 출처: [NVIDIA RTX PRO Blackwell GPU Architecture Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf) — "However, the unified cores can only operate as either FP32 or INT32 cores in any given clock cycle."

게이밍 워크로드에서는 INT32 비중이 낮아서 유연성을 택한 것이다. 반면 데이터센터 GPU(A100, H100, B200)는 FP32/INT32 분리를 유지한다 — AI 워크로드에서는 분리가 명확히 이득이라는 판단이다.

### 3.3 Tensor Core

Tensor Core는 **행렬 곱셈(MMA, Matrix Multiply-Accumulate)을 하나의 명령으로 처리하는 전용 유닛**이다.

일반 CUDA Core와 비교하면:

```
CUDA Core (FP32):
  스레드 1개가 매 클럭 1 FMA 수행 → 2 FLOPs/clock/core

Tensor Core:
  Warp(32 스레드)가 협력하여 작은 행렬 곱셈을 한 번에 수행
  → 하나의 명령으로 수십~수백 FMA 동시 처리
```

> 출처: [NVIDIA V100 Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), p.17 — "Each Tensor Core performs the following operation: D = A×B + C, where A, B, C, and D are 4×4 matrices."
> 출처: [NVIDIA Volta Tuning Guide](https://docs.nvidia.com/cuda/volta-tuning-guide/) — "Each Tensor Core performs the following operation: D = AxB + C, where A, B, C, and D are 4x4 matrices. The matrix multiply inputs A and B are FP16 matrices, while the accumulation matrices C and D may be FP16 or FP32 matrices."

#### 세대별 발전

| 세대 | 아키텍처 | 지원 정밀도 | 주요 변화 |
|------|---------|-----------|----------|
| 1세대 | Volta (V100) | FP16 | Tensor Core 최초 도입, SM당 8개 |
| 2세대 | Turing (T4) | FP16, INT8, INT4, INT1 | 추론 정밀도 확대 |
| 3세대 | Ampere (A100) | FP16, BF16, TF32, FP64, INT8, INT4 | TF32 기본, 구조적 희소성, SM당 4개로 감소 |
| 4세대 | Hopper (H100) | + FP8 | Transformer Engine, 처리량 2배 |
| 5세대 | Blackwell (B200) | + FP4 | 2세대 Transformer Engine |

> 출처: 각 세대별 아키텍처 Whitepaper — [V100 WP](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) p.17 (1세대, SM당 8개), [A100 WP](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) p.18-20 (3세대, SM당 4개, TF32, 구조적 희소성), [H100 WP](https://resources.nvidia.com/en-us-tensor-core) p.27 (4세대, FP8, Transformer Engine), [B200 Datasheet](https://www.nvidia.com/en-us/data-center/b200/) (5세대, FP4).

SM당 Tensor Core 개수는 Volta/Turing의 8개에서 Ampere/Hopper의 4개로 줄었다. 하지만 개별 Tensor Core의 클럭당 처리량이 더 커서 SM 전체 처리량은 세대마다 증가한다. 적은 수의 더 강력한 유닛으로 가는 방향이다.

#### TF32 (TensorFloat-32)

Ampere(A100)부터 도입된 연산 포맷이다. 기존 FP32 코드를 수정 없이 Tensor Core에서 가속하기 위해 만들어졌다.

```
FP32:  부호(1) + 지수(8) + 가수(23) = 32 bit
FP16:  부호(1) + 지수(5) + 가수(10) = 16 bit
TF32:  부호(1) + 지수(8) + 가수(10) = 19 bit
```

> 출처: [NVIDIA A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), p.20 — "NVIDIA TF32 ... uses the same 10-bit mantissa as the half-precision (FP16) math, shown to have more than sufficient margin for the precision requirements of AI workloads. And TF32 adopts the same 8-bit exponent as FP32."

FP32의 범위(지수 8bit)와 FP16의 정밀도(가수 10bit)를 결합한 형태다. cuBLAS에서 FP32 GEMM을 호출하면 내부적으로 TF32를 사용해 Tensor Core에서 실행하는 것이 기본 동작이다. 사용자 코드 변경이 필요 없다.
> 출처: [NVIDIA A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), p.21 — "TF32 Tensor Core operations are enabled by default in cuBLAS."

#### Transformer Engine (H100~)

FP8과 FP16을 레이어별로 자동 전환하는 하드웨어+소프트웨어 기능이다. 각 레이어의 텐서 값 분포를 실시간으로 추적해서, 정밀도 손실이 적은 레이어는 FP8로 내려 처리량을 높이고, 민감한 레이어는 FP16을 유지한다.
> 출처: [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core), p.30 — Transformer Engine 동작 설명.

### 3.4 SFU (Special Function Unit)

`sin()`, `cos()`, `exp()`, `log()`, `rsqrt()` 같은 초월 함수를 처리하는 유닛이다. Processing Block당 4개.
> 출처: [CUDA C++ Programming Guide — Arithmetic Instructions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#arithmetic-instructions) — SFU throughput 정의. [NVIDIA A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) SM 다이어그램에서 SFU 위치 확인.

AI 워크로드에서는 activation function(GELU, SiLU 등)이나 Softmax의 `exp()` 계산에 사용된다.

### 3.5 Load/Store Unit (LD/ST)

메모리에서 데이터를 읽고(Load) 쓰는(Store) 유닛이다. Processing Block당 8개.
> 출처: [NVIDIA A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), SM 다이어그램 — "8 Load/Store units" per processing block.

전역 메모리(HBM), 공유 메모리, 레지스터 파일 간 데이터 이동을 담당한다.

---

## 4. Warp Scheduler: SM의 지휘자

### 4.1 기본 동작

SM에는 4개의 Warp Scheduler가 있고(Processing Block당 1개), 각 스케줄러가 매 클럭 사이클마다 자신에게 할당된 Warp 풀에서 Ready 상태인 Warp를 찾아 명령어를 발행(issue)한다.

> 출처: [CUDA C++ Programming Guide — Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading) — "At every instruction issue time, a warp scheduler selects a warp that has threads ready to execute its next instruction and issues the instruction to those threads."

SM 하나에 4개의 스케줄러가 있으므로, **매 사이클 최대 4개의 Warp(= 128 스레드)가 동시에 명령어를 실행**할 수 있다.
> 출처: [NVIDIA Developer Forum](https://forums.developer.nvidia.com/t/can-warps-from-different-ctas-be-coscheduled/298871) — NVIDIA 엔지니어 Robert Crovella의 확인: "an SM may have up to 4 warp schedulers. If it has 4 warp schedulers ... one warp scheduler could schedule an instruction from one CTA, whereas in the same clock cycle another warp scheduler schedules an instruction from another CTA."

### 4.2 Warp가 Ready가 아닌 경우: Stall

Warp가 Ready가 아닌 상태를 **Stall**이라 한다. Stall의 주요 원인:

**Long Scoreboard Stall** — 전역 메모리(HBM) 접근을 요청했는데, 데이터가 아직 도착하지 않은 경우. HBM 접근은 수백 cycle이 걸리므로, 이 동안 해당 Warp는 실행 불가.

**Short Scoreboard Stall** — 공유 메모리 접근이나 짧은 파이프라인 의존성 때문에 대기하는 경우.

**Barrier Stall** — `__syncthreads()` 같은 동기화 명령을 만난 경우.

> 출처: [Nsight Compute — Warp Stall Reasons](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-decoder) — Long Scoreboard, Short Scoreboard, Barrier 등 stall 원인 분류.
> 출처: [Dissecting the NVIDIA Hopper GPU Architecture via Microbenchmarking (2025)](https://arxiv.org/abs/2501.05370) — H100 HBM latency ~658 cycles 실측.

### 4.3 Latency Hiding: GPU가 빠른 진짜 이유

CPU는 메모리 대기 시간을 줄이기 위해 큰 캐시, 비순차 실행(OoO) 같은 복잡한 기법을 쓴다. GPU는 전혀 다른 전략을 택한다 — **기다리는 대신 다른 일을 한다.**

한 Warp가 HBM 데이터를 기다리며 stall되면, Warp Scheduler는 즉시 다른 Ready Warp로 전환한다. 이 전환에는 추가 비용이 거의 없다. 각 Warp의 레지스터 상태가 이미 SM 내 Register File에 상주하고 있기 때문이다.

> 출처: [CUDA C++ Programming Guide — Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-multithreading) — "The execution context ... for each warp processed by a multiprocessor is maintained on-chip during the entire lifetime of the warp. Therefore, switching from one execution context to another has no cost."

이것이 **Latency Hiding**이다. 개별 Warp 입장에서는 수백 cycle을 기다리고 있지만, SM 전체 입장에서는 그 동안 다른 Warp가 연산을 수행하므로 파이프라인이 비지 않는다.

### 4.4 Occupancy와 Latency Hiding의 관계

**Occupancy = SM에 상주하는 활성 Warp 수 / SM이 수용 가능한 최대 Warp 수**

A100 기준 SM당 최대 64 Warp(= 2,048 스레드)를 수용할 수 있다.
> 출처: [CUDA C++ Programming Guide — Compute Capabilities Table](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications) — Compute Capability 8.0: "Maximum number of resident warps per SM: 64."

Occupancy가 높을수록 Warp Scheduler가 선택할 수 있는 Ready Warp 후보가 많아지므로, latency hiding이 더 잘 된다. 하지만 **Occupancy가 높다고 항상 성능이 좋은 것은 아니다.**

Occupancy를 제한하는 리소스 세 가지:

1. **레지스터 사용량** — SM당 총 레지스터: 65,536개. 스레드당 레지스터를 많이 쓰면 SM에 올릴 수 있는 총 Warp 수가 줄어든다.
2. **공유 메모리 사용량** — Block당 공유 메모리를 많이 쓰면 SM에 동시에 올릴 수 있는 Block 수가 줄어든다.
3. **Block당 스레드 수** — 32의 배수가 아니면 마지막 Warp에 빈 스레드가 발생한다.

> 출처: [CUDA C++ Best Practices Guide — Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#occupancy) — Occupancy 정의, 제한 요소, 최적화 가이드.

---

## 5. Warp Divergence

### 5.1 문제 상황

Warp 내 32개 스레드가 조건 분기로 다른 경로를 타야 하면, 두 경로를 순차적으로 실행한다. 결과적으로 처리량이 떨어진다. 이것이 **Warp Divergence**다.

> 출처: [CUDA C++ Programming Guide — Control Flow](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#control-flow-instructions) — "If threads of a warp diverge ... the warp serially executes each branch path."

### 5.2 Volta 이후: Independent Thread Scheduling

Volta(V100)부터 **Independent Thread Scheduling**이 도입됐다. 각 스레드가 독립적인 프로그램 카운터(PC)와 호출 스택을 가진다.

> 출처: [NVIDIA Volta Tuning Guide](https://docs.nvidia.com/cuda/volta-tuning-guide/) — "The Volta architecture introduces Independent Thread Scheduling among threads in a warp."
> 출처: [NVIDIA V100 Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf), p.16 — "Volta maintains per-thread scheduling resources such as a program counter (PC) and call stack (S)."

다만 본질은 변하지 않는다 — 분기가 있으면 비활성 스레드가 생기고 처리량이 줄어든다. Independent Thread Scheduling은 이를 더 효율적으로 관리할 뿐이다.

### 5.3 AI 워크로드에서 Divergence가 드문 이유

행렬 곱셈, element-wise 연산(ReLU, GELU), LayerNorm, Softmax 등 AI의 핵심 연산은 모든 스레드가 정확히 같은 연산을 수행한다. 데이터만 다를 뿐 실행 경로는 동일하다. GPU의 SIMT 모델이 AI 연산에 유독 강한 이유 중 하나다.

---

## 6. SM 구조 요약: 전체 그림

A100 SM 하나를 기준으로 정리하면:

```
┌─────────────────────────────── SM ───────────────────────────────┐
│                                                                   │
│  ┌─ Processing Block 0 ─┐  ┌─ Processing Block 1 ─┐             │
│  │ Warp Scheduler (1)    │  │ Warp Scheduler (1)    │             │
│  │ FP32 Core ×16         │  │ FP32 Core ×16         │             │
│  │ INT32 Core ×16        │  │ INT32 Core ×16        │             │
│  │ FP64 Core ×8          │  │ FP64 Core ×8          │             │
│  │ Tensor Core ×1        │  │ Tensor Core ×1        │             │
│  │ LD/ST Unit ×8         │  │ LD/ST Unit ×8         │             │
│  │ SFU ×4                │  │ SFU ×4                │             │
│  │ Register File 64KB    │  │ Register File 64KB    │             │
│  └───────────────────────┘  └───────────────────────┘             │
│                                                                   │
│  ┌─ Processing Block 2 ─┐  ┌─ Processing Block 3 ─┐             │
│  │      (동일 구성)       │  │      (동일 구성)       │             │
│  └───────────────────────┘  └───────────────────────┘             │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │        Shared Memory / L1 Cache: 192 KB (전체 공유)          │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘

GPU 전체 (A100 SXM): 108 SM → FP32 6,912개 / Tensor Core 432개
```
> 출처: [NVIDIA A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf), p.14-22.

---

# Part 2. Roofline Model — GPU 성능의 물리적 한계

## 7. 두 개의 천장: Compute와 Memory

GPU 성능에는 두 가지 물리적 상한선이 존재한다.

**연산 처리량(Compute Throughput)** — GPU가 초당 수행할 수 있는 부동소수점 연산의 최대 횟수. 단위는 FLOP/s.

**메모리 대역폭(Memory Bandwidth)** — GPU가 초당 HBM에서 읽거나 쓸 수 있는 데이터의 최대 양. 단위는 Byte/s.

어떤 연산이 실행될 때, 이 두 상한 중 **더 낮은 쪽이 실제 성능을 결정**한다. 이 두 한계를 하나의 그래프에 표현한 것이 Roofline Model이다.
> 출처: Williams, S., Waterman, A., & Patterson, D. (2009). [Roofline: An Insightful Visual Performance Model for Multicore Architectures.](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf) Communications of the ACM, 52(4), 65-76. — Roofline Model 원본 논문.

---

## 8. Arithmetic Intensity: 연산과 메모리의 비율

### 8.1 정의

```
Arithmetic Intensity (AI) = 수행하는 연산 횟수 (FLOPs) / 이동하는 데이터 양 (Bytes)
단위: FLOP/Byte
```

> 출처: [GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) — NVIDIA 공식 딥러닝 성능 가이드. "The ratio of arithmetic operations to memory operations ... known as arithmetic intensity."

### 8.2 직관적 이해

**Case A: 벡터 덧셈 (C[i] = A[i] + B[i])**

```
원소 하나당:
  연산: 덧셈 1회 = 1 FLOP
  데이터: A[i] 읽기(4B) + B[i] 읽기(4B) + C[i] 쓰기(4B) = 12 Bytes (FP32)

AI = 1 / 12 ≈ 0.083 FLOP/Byte
```

**Case B: 행렬 곱셈 (C = A × B, 모두 N×N)**

```
총 연산: 2N³ FLOPs
총 데이터: 3N² × 4 Bytes (FP32)

AI = 2N³ / (12N²) = N/6 FLOP/Byte
```

N = 4096이면 AI ≈ 683 FLOP/Byte.

### 8.3 핵심 통찰

Arithmetic Intensity가 높아지려면 **데이터 재사용**이 있어야 한다. 행렬 곱셈은 본질적으로 데이터 재사용이 많은 연산이고, 그래서 AI가 높다. GPU 커널 최적화에서 tiling(데이터를 shared memory에 올려놓고 재사용)이 중요한 이유도 여기에 있다.

---

## 9. Roofline 그래프 구축

### 9.1 그래프 구조

X축에 Arithmetic Intensity(FLOP/Byte), Y축에 달성 가능한 성능(FLOP/s). 두 축 모두 **log scale**이다.

```
       ▲ Attainable Performance (FLOP/s)  [log scale]
       │
  Peak │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┬━━━━━━━━━━━━━━━━  ← Compute Ceiling
  FLOP/s                                 │
       │                                ╱│
       │                              ╱  │
       │                            ╱    │
       │                          ╱      │
       │                        ╱        │
       │                      ╱    Memory Bandwidth Slope
       │                    ╱      (기울기 = BW)
       │                  ╱
       │                ╱
       │──────────────╱──────────────────────────────────→ Arithmetic Intensity
                           ↑                               (FLOP/Byte) [log scale]
                      Ridge Point
```

### 9.2 Ridge Point

```
Ridge Point = Peak FLOP/s / Peak Bandwidth
```

AI < Ridge Point → Memory Bound / AI > Ridge Point → Compute Bound

### 9.3 세대별 Ridge Point 비교

| GPU | Peak FP16 Tensor (TFLOP/s) | HBM BW (TB/s) | Ridge Point (FLOP/Byte) |
|-----|---------------------------|---------------|------------------------|
| V100 | 125 | 0.90 | **138.9** |
| A100 80GB | 312 | 2.04 | **152.9** |
| H100 SXM | 1,979 | 3.35 | **590.7** |
| H200 SXM | 1,979 | 4.80 | **412.3** |
| B200 | 2,250 | 8.00 | **281.3** |

> 출처 (각 GPU 스펙):
> - V100: [V100 Datasheet](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r8.pdf) — 125 TFLOP/s FP16 Tensor, 900 GB/s HBM2.
> - A100: [A100 Datasheet](https://www.nvidia.com/en-us/data-center/a100/) — 312 TFLOP/s FP16 Tensor, 2,039 GB/s HBM2e (SXM).
> - H100: [H100 Datasheet](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306) — 1,979 TFLOP/s FP16 Tensor*, 3.35 TB/s HBM3. (* = dense, sparsity 제외)
> - H200: [H200 Datasheet](https://www.nvidia.com/en-us/data-center/h200/) — 1,979 TFLOP/s (H100과 동일), 4.8 TB/s HBM3e.
> - B200: [B200 Datasheet](https://www.nvidia.com/en-us/data-center/b200/) — 2,250 TFLOP/s FP16 Tensor*, 8.0 TB/s HBM3e. (* = dense)

두 가지 관찰:

**H100의 Ridge Point가 590.7로 매우 높다.** Tensor Core 처리량이 A100 대비 6.3배 뛰었는데 대역폭은 1.6배만 늘었기 때문이다.

**H200과 B200에서 Ridge Point가 낮아졌다.** NVIDIA가 Memory Bound 문제를 인식하고 HBM 대역폭을 공격적으로 올리고 있다는 의미다.

---

## 10. Compute Throughput과 Memory Bandwidth의 물리적 기원

### 10.1 Compute Throughput

#### CUDA Core 기반 (FP32)

```
Peak FP32 FLOP/s = SM 수 × FP32 Cores/SM × 2 × Clock Speed

예: A100 SXM
  = 108 × 64 × 2 × 1.41 GHz
  ≈ 19.5 TFLOP/s
```

> 출처: [A100 Datasheet](https://www.nvidia.com/en-us/data-center/a100/) — 108 SM, 19.5 TFLOP/s FP32. [A100 Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) p.18 — 64 FP32 Cores/SM, boost clock 1.41 GHz.

#### Tensor Core 기반

Tensor Core는 CUDA Core가 스레드당 1 FMA인 반면, Warp 단위로 행렬 곱셈을 일괄 처리해서 같은 면적에서 훨씬 높은 처리량을 낸다. 실제 AI 워크로드의 Peak FLOP/s는 CUDA Core가 아닌 **Tensor Core 스펙**으로 결정된다.

### 10.2 Memory Bandwidth: HBM의 물리적 구조

#### GDDR vs HBM

HBM은 DRAM 다이를 수직으로 쌓고(3D stacking), TSV(Through-Silicon Via)로 관통 연결한 뒤, 실리콘 인터포저 위에서 GPU 다이 바로 옆에 배치한다.
> 출처: [JEDEC HBM3 Standard (JESD238)](https://www.jedec.org/standards-documents/docs/jesd238b01) — "HBM3 DRAM uses a wide-interface architecture ... Each channel interface maintains a 64 bit data bus operating at double data rate (DDR)."
> 출처: [Synopsys — What is HBM3](https://www.synopsys.com/glossary/what-is-high-bandwitdth-memory-3.html) — TSV, 인터포저, 3D stacking 구조 설명.

#### HBM 세대별 특성

| HBM 세대 | 스택당 채널 | 채널당 버스 폭 | 사용 GPU |
|---------|-----------|-------------|---------|
| HBM2 | 8 | 128 bit | V100, A100 40GB |
| HBM2e | 8 | 128 bit | A100 80GB |
| HBM3 | 16 | 64 bit | H100 |
| HBM3e | 16 | 64 bit | H200, B200 |

> 출처: [JEDEC HBM3 Announcement (2022)](https://www.jedec.org/news/pressreleases/jedec-publishes-hbm3-update-high-bandwidth-memory-hbm-standard) — "Doubling the number of independent channels from 8 (HBM2) to 16."
> 출처: [Wevolver — HBM3 Engineering Guide](https://www.wevolver.com/article/what-is-high-bandwidth-memory-3-hbm3-complete-engineering-guide-2025) — "HBM3 employs 16 independent 64-bit channels, resulting in a 1024-bit wide interface per stack."

#### Memory Wall 문제

```
A100 → H100:
  Tensor Core 처리량: 312 → 1,979 TFLOP/s  (6.3배 증가)
  HBM 대역폭: 2.04 → 3.35 TB/s            (1.6배 증가)
```

> 출처: 위 GPU별 Datasheet에서 도출한 비율 계산.

---

## 11. 핵심 정리

### Part 1: SM 내부 구조

1. **SM은 4개의 Processing Block으로 나뉜다.** 각 블록에 Warp Scheduler, FP32/INT32 Core, Tensor Core, SFU, LD/ST Unit이 있다.
2. **FP32와 INT32가 분리된 이유는 인덱싱 비용 때문이다.** Volta 이후 분리되어 동시 실행 가능.
3. **Tensor Core는 행렬 곱셈을 Warp 단위로 일괄 처리한다.** 세대마다 지원 정밀도 확대.
4. **Warp Scheduler가 latency hiding의 핵심이다.** 전환 비용 거의 0.
5. **AI 워크로드에서 Warp Divergence는 거의 없다.**

### Part 2: Roofline Model

6. **GPU 성능에는 두 천장이 있다** — Compute Throughput(FLOP/s)과 Memory Bandwidth(Byte/s). 둘 중 낮은 쪽이 실제 성능.
7. **Arithmetic Intensity(FLOP/Byte)가 어떤 천장에 부딪히는지 결정한다.** Ridge Point보다 낮으면 Memory Bound, 높으면 Compute Bound.
8. **연산 처리량 증가 속도가 HBM 대역폭 증가 속도를 앞지르고 있다.** A100→H100에서 Tensor Core는 6.3배, HBM은 1.6배 증가. 이 격차가 Memory Wall 문제이며, 최신 GPU일수록 Memory Bound에 빠지기 쉽다.

### Week 5 연결

이번 주에 배운 Roofline Model은 Week 5 "LLM 추론 구조와 병목 이해"에서 실제 LLM 연산(Prefill, Decode)을 Roofline 위에 올려놓고 병목을 진단하는 데 사용된다. 같은 행렬 곱셈이라도 입력 크기에 따라 Memory Bound ↔ Compute Bound가 바뀌는 것을 직접 계산하게 될 것이다.

---

## 참고 자료

### 핵심 출처 (본문에서 반복 인용)

- [NVIDIA A100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) — SM 내부 구조, Processing Block, FP32/INT32 분리, TF32, 구조적 희소성.
- [NVIDIA V100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) — FP32/INT32 분리 최초 도입, Independent Thread Scheduling, Tensor Core 1세대.
- [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core) — 4세대 Tensor Core, Transformer Engine, FP8.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — Warp, Thread Block, 메모리 모델, Occupancy, Divergence 정의.
- Williams, S. et al. (2009). [Roofline: An Insightful Visual Performance Model for Multicore Architectures.](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf) Comm. of the ACM.

### GPU 스펙 Datasheet

- [V100 Datasheet](https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r8.pdf)
- [A100 Datasheet](https://www.nvidia.com/en-us/data-center/a100/)
- [H100 Datasheet](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306)
- [H200 Datasheet](https://www.nvidia.com/en-us/data-center/h200/)
- [B200 Datasheet](https://www.nvidia.com/en-us/data-center/b200/)

### 성능 분석

- [GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) — NVIDIA 공식. Compute/Memory Bound, Arithmetic Intensity.
- Ivanov, A. et al. (2021). [Data Movement Is All You Need.](https://arxiv.org/abs/2007.00072) MLSys 2021. — Transformer 연산의 데이터 이동 분석.

### HBM 기술

- [JEDEC HBM3 Standard (JESD238)](https://www.jedec.org/standards-documents/docs/jesd238b01) — HBM3 공식 스펙.
- [JEDEC HBM3 Announcement (2022)](https://www.jedec.org/news/pressreleases/jedec-publishes-hbm3-update-high-bandwidth-memory-hbm-standard) — 채널 수 8→16 변경 등.

### 컨슈머 GPU

- [NVIDIA RTX PRO Blackwell GPU Architecture Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf) — 컨슈머 Blackwell SM의 FP32/INT32 통합 구조.

### 심화

- [Dissecting the NVIDIA Hopper GPU Architecture via Microbenchmarking (2025)](https://arxiv.org/abs/2501.05370) — H100 HBM latency ~658 cycles 실측.
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — Occupancy, 메모리 접근 패턴 최적화.
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) — Warp stall 원인 분류, 프로파일링.