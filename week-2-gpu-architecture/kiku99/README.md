# Week 2. GPU 아키텍처와 병렬 연산 구조 이해

## overview

GPU가 AI 워크로드에 강한 이유는 결국 **"같은 연산을 대량의 데이터에 동시에 적용하는" 병렬성**에 최적화된 하드웨어이기 때문이다. 이 주차는 ① CPU와의 구조적 차이, ② 메모리 계층, ③ 실행 단위(Thread~Grid), ④ 커널 실행 흐름 순으로 GPU 내부를 파고든다.

---

## 1. CPU vs GPU 구조적 차이

GPU(Graphics Processing Unit)는 원래 화면의 수많은 픽셀을 동시에 처리하기 위해 태어났다. 2006년 NVIDIA가 **CUDA**를 공개하며 그래픽을 넘어 일반 연산(GPGPU)에 쓰이기 시작했고, AI의 기반인 행렬·텐서 연산이 "같은 연산을 대량 데이터에 반복 적용하는" 문제라 GPU와 잘 맞았다.

| 구분 | CPU | GPU |
|---|---|---|
| 설계 목표 | 지연 시간 최소화 | 처리량 최대화 |
| 코어 구성 | 적은 수의 강한 코어 | 많은 수의 단순한 코어 |
| 강점 | 단일 작업을 빠르게 | 같은 작업을 대량 병렬 처리 |
| 캐시 전략 | 큰 캐시 + 복잡한 제어 로직 | 상대적으로 작은 캐시 + 병렬성 활용 |
| 지연 대응 | 캐시·분기 예측 등 복잡한 제어 | 다른 warp로 전환하며 지연 숨기기 |
| 대표 작업 | OS, 웹 서버, 분기 많은 코드 | 행렬 연산, 이미지 처리, 딥러닝 |

- CPU 철학: **"한 가지 일을 최대한 빨리 끝낸다"**
- GPU 철학: **"동시에 많은 일을 처리한다"** (각 코어는 단순하지만 수가 많다)

### GPU가 병렬 연산에 강한 4가지 이유

1. **데이터 병렬성(Data Parallelism)** — 같은 연산을 많은 데이터에 동시에 적용하기 쉽다.
2. **연속·규칙적 메모리 접근** — 연속 메모리 접근이 많으면 **메모리 병합(Coalescing)** 으로 하나의 트랜잭션으로 묶여 대역폭을 효율적으로 쓴다.
3. **지연 시간 숨기기(Latency Hiding)** — 일부 thread가 메모리를 기다리는 동안 다른 warp를 돌려 지연을 숨긴다.
4. **높은 메모리 대역폭** — HBM 같은 고대역폭 메모리로 CPU보다 훨씬 넓은 데이터 통로를 확보한다.

---

## 2. GPU 메모리 계층 구조

GPU는 **빠르지만 작은 메모리 ↔ 크지만 느린 메모리**가 계층을 이룬다. 핵심 직관: 느린 HBM에서 반복해서 읽는 대신, 자주 쓰는 데이터는 **shared memory나 register 근처로 끌어와 재사용**하는 게 훨씬 유리하다.

> `cycle`은 GPU 내부 클럭 한 번 진행 시간 단위로, "상대적으로 얼마나 빠른/느린가"를 보여주는 표현이다. 실제 값은 아키텍처·클럭·접근 패턴에 따라 달라진다.

| 메모리 | 스코프 | 위치 | 레이턴시 | 크기 (Hopper 기준) | 관리 주체 |
|---|---|---|---|---|---|
| **Register** | Thread 전용 | On-Chip | ~1 cycle | SM당 65,536개 (약 256KB) | 컴파일러 |
| **Local Memory** | Thread 전용 | Off-Chip (DRAM) | ~500 cycle | thread당 최대 512KB | 컴파일러 |
| **Shared Memory** | Block 내 thread 공유 | On-Chip | ~5 cycle | SM당 최대 228KB | **사람이 직접** |
| **L1 Cache** | Block 내 thread 공유 | On-Chip | ~5 cycle | Shared와 통합 256KB | 하드웨어 |
| **Constant Memory** | Grid 전체 (읽기 전용) | Off-Chip (+Cache) | ~5 / ~500 cycle | 64KB | 사람(사전 선언) |
| **Texture Memory** | Grid 전체 (읽기 전용) | Off-Chip (+Cache) | ~5 / ~500 cycle | 가변 | 하드웨어 |
| **Global Memory** | 전체 thread | Off-Chip (HBM) | ~500 cycle | H100 80GB / H200 141GB | 사람(`cudaMalloc`) |
| **L2 Cache** | 전체 thread | On-Chip (모든 SM 공유) | ~200 cycle | H100 기준 50MB | 하드웨어 |

### 계층별 포인트

- **Register**: 가장 빠름. CUDA 커널 내 지역변수 저장. 한 블록이 register를 많이 쓰면 SM에 동시 상주하는 block·warp 수가 줄어 **Occupancy(점유율)** 가 떨어진다.
- **Local Memory**: 이름은 "local"이지만 실제론 Off-Chip. register에 못 담는 큰 데이터/초과 변수(Register Spill)가 여기로 간다. 그래서 register만큼 빠르지 않다.
- **Shared Memory**: 같은 thread block 안에서만 공유. **사람이 직접 채우고 재사용**할 수 있는 영역 → 잘 쓰면 성능 향상의 핵심. 다른 block의 shared memory는 접근 불가.
- **L1 / Shared 통합**: Hopper는 SM당 L1+Shared를 합쳐 **Unified Data Cache(최대 256KB)** 로 관리. shared로 할당하고 남은 영역을 L1이 사용.
- **Constant/Texture**: 읽기 전용 + 전용 캐시. Cache Hit 시 빠르고(~5 cycle), Miss 시 global과 비슷(~500 cycle). Constant는 작고 자주 읽지만 안 바뀌는 값에 적합.
- **Global Memory**: 가장 크고 가장 느림. `cudaMalloc`으로 할당, `cudaMemcpy`로 host↔device 복사하는 대상이 바로 이 영역. 대역폭은 H100 3.35TB/s(HBM3), H200 4.8TB/s(HBM3e).

---

## 3. GPU 하드웨어 및 실행 단위

### SM (Streaming Multiprocessor)

GPU는 SM을 여러 개 묶은 구조다. SM은 여러 개의 CUDA Core를 가진 연산 장치이며, CUDA Core 외에도 **Warp Scheduler, Tensor Core, Register File** 등을 포함한다.

- H100 SXM 기준: GPU당 **132개 SM**, SM당 **128 CUDA Core** → 총 **16,896개** CUDA Core
- 정리: SM은 GPU 내부에서 연산 수행을 위해 **Warp를 스케줄링하고, 이를 위한 메모리·구성요소를 포함하는 하드웨어 블록**

### 실행 계층 (Hopper 기준)

```
Grid                              ← CUDA Kernel Launch 전체 범위
└─ Thread Block Cluster (Hopper)  ← 여러 SM에 걸친 협력 그룹 (최대 16 블록)
   └─ Thread Block / CTA          ← 하나의 SM 위에서 실행
      └─ Warp (32 threads)        ← 실제 실행 단위, SIMT
         └─ Thread                ← 최소 실행 단위
```

| 단위 | 의미 |
|---|---|
| **Thread** | 최소 실행 단위. 고유 register와 데이터 상태를 가짐 |
| **Warp** | 32 thread = 하나의 실제 실행 단위. 32 thread가 같은 명령을 동시에 → **SIMT** (Single Instruction, Multi Threads). thread별 분기는 가능하지만 성능 저하 유발 |
| **Thread Block (CTA)** | 한 SM에서 실행되는 thread 집합. thread 수는 32의 배수, 최대 1024 (warp 32개). 블록 내 thread는 shared memory로 공유 |
| **Thread Block Cluster** | Hopper에서 추가. 가까운 여러 SM에 블록들을 함께 배치해 협력. 서로 다른 SM의 shared memory에 접근 가능 → **분산 공유 메모리(DSMEM)**. global을 거치지 않아 빠름 |
| **Grid** | 하나의 커널 launch를 구성하는 모든 block의 집합 |

> **Thread Block Cluster가 등장한 배경**: 원래 block은 독립 실행·동기화 불가였고, block 하나의 shared memory가 작다는 한계가 있었다. block 간 데이터 교환은 global memory를 거쳐야 했는데 이는 shared 대비 **약 100배 느리다**. 이를 풀기 위해 가까운 SM의 블록들을 묶어 shared 간 직접 교환(DSMEM)을 가능하게 했다.

---

## 4. GPU 내부 동작 흐름

GPU는 마법처럼 "알아서 병렬화"하지 않는다. **프로그래머가 먼저 "최종 결과가 몇 개 필요한가"를 정해야** 한다.

```
1. 데이터 준비   → 출력 크기 / 문제 크기 파악
2. 그리드 결정   → 전체 일을 몇 개 block으로 나눌지 결정
3. 블록 분할     → block 크기와 thread 수 결정
4. SM 할당       → block들을 사용 가능한 SM에 자동 배치
5. 워프 실행     → SM 안에서 warp 단위로 실제 실행
```

### 단계별 핵심

1. **데이터 준비** — 문제 크기와 출력 모양을 먼저 정한다 (벡터 덧셈이면 결과 길이 N, 행렬 곱이면 행×열, 이미지면 출력 픽셀 수).
2. **그리드 결정** — `<<<grid, block>>>`의 grid = "kernel 전체가 몇 개 block으로 구성되는가". 데이터가 많으면 grid가 커진다.
3. **블록 분할** — block당 thread 수 결정. 고려사항:
   - warp 크기(32)의 배수인가?
   - shared memory를 얼마나 쓸 것인가?
   - register pressure가 너무 커지지 않는가?
   - block이 너무 작아 GPU를 덜 채우는 건 아닌가?
4. **SM 할당** — GPU가 block들을 가용 SM에 **자동 배치(block placement)**. 한 SM에 자원이 허용하면 여러 block이 동시 상주 가능.
5. **워프 실행** — block의 thread들이 32개씩 warp로 묶여 실행. Warp Scheduler가 실행 가능한 warp를 골라 큐에 밀어넣음. **어떤 warp가 메모리를 기다리면 다른 warp를 실행** → 이렇게 memory latency를 숨기며 처리량을 유지한다.

---

## 요약

- GPU는 **처리량 최대화**를 위해 단순한 코어를 대량으로 둔 하드웨어다. 강점은 데이터 병렬성, 메모리 병합, latency hiding, 고대역폭이다.
- 메모리는 **빠르고 작은 것(register/shared) ~ 느리고 큰 것(global/HBM)** 의 계층. 자주 쓰는 데이터를 register/shared로 끌어와 재사용하는 것이 최적화의 출발점.
- 실행 단위는 **Thread → Warp(32) → Block → (Cluster) → Grid**. Warp는 SIMT로 동작하므로 분기는 성능을 깎는다.
- 커널 실행은 **프로그래머가 문제 크기를 정의 → GPU가 block을 SM에 배치 → warp 단위 실행 + latency hiding**으로 흐른다.
