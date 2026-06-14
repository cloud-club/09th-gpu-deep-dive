# Week 4. AI 워크로드 네트워크 이해

## overview

GPU 한 장은 자기 계산을 빠르게 끝내지만, 여러 GPU가 하나의 모델을 같이 학습하려면 **중간 결과를 서로 맞춰야** 한다. AI 학습은 대개 동기식(synchronous)이라 **한 rank의 통신 지연이 전체 step을 지연**시킨다. 그래서 AI에서 네트워크는 "계산이 끝난 GPU들을 다음 step으로 보내기 위한 **동기화 장치**"다.

---

## 0. AI 네트워크는 계층적으로 동작한다

PyTorch/Megatron이 직접 packet을 만들지 않는다. 요청은 아래 계층을 따라 내려간다.

```
Framework (PyTorch / JAX / Megatron / vLLM)
  → communication runtime (NCCL / MPI / NVSHMEM)
  → collective operation (AllReduce / AllGather / ReduceScatter / All-to-All)
  → transport surface (RDMA verbs / libfabric / socket)
  → fabric / interconnect (NVLink / PCIe / InfiniBand / RoCE / EFA)
  → GPU / NIC / HCA / Switch
```

| 층 | 핵심 질문 | 대표 용어 |
|---|---|---|
| 프레임워크 | 어느 rank들이 같이 학습/추론하는가 | DDP, FSDP, tensor/expert parallel |
| 통신 런타임 | 어떤 collective를 어떤 알고리즘으로 | NCCL, MPI, NVSHMEM |
| 데이터 이동 모델 | CPU·커널 복사를 얼마나 줄이는가 | RDMA, GPUDirect RDMA, libfabric |
| Fabric | packet이 어떤 네트워크 위를 지나는가 | InfiniBand, RoCEv2, EFA, Ethernet |
| 운영 검증 | 느림·hang·loss가 어느 층에서 생겼나 | `NCCL_DEBUG`, `fi_info`, GID, PFC, ECN, QP state |

> **층을 나누는 이유**: "AllReduce가 느리다"는 증상은 하나지만 원인은 여러 층에 있다. rank 배치, NCCL의 NIC 선택, RoCE의 PFC/ECN 설정 등. 층을 나누면 디버깅 질문도 나뉜다.

---

## 1. AI 워크로드에서 네트워크가 중요한 이유

### GPU를 늘리면 계산만 늘어나는 게 아니다

여러 GPU로 나누는 순간 각 GPU는 자기 몫을 계산한 뒤 다른 GPU와 중간 결과를 맞춰야 한다.
- **Data Parallel**: gradient를 합쳐야 함
- **Tensor Parallel**: layer 안 activation/partial result 교환
- **MoE**: token을 담당 expert가 있는 rank로 보냄

동기식이므로 **한 rank가 늦으면 나머지가 기다린다**. 평균 대역폭이 좋아도 **tail latency**가 길면 step time이 흔들리고, 비싼 GPU가 노는 시간이 늘어난다.

### 기존 네트워크가 AI에 부적절한 이유

| 구분 | 일반 애플리케이션 네트워크 | AI 학습·추론 네트워크 |
|---|---|---|
| 주된 방향 | 사용자↔서버 (**north-south**) | 서버↔서버, GPU↔GPU (**east-west**) |
| 트래픽 모양 | 요청마다 독립적 | step 경계에서 **동기식 burst** |
| 지연 민감도 | 일부 요청 지연 OK | 한 rank 지연 → 전체 step 지연 |
| CPU 역할 | 네트워크 stack 처리 가능 | CPU copy·kernel path가 병목 |
| 네트워크 목표 | 연결성, 처리량, 장애 격리 | 낮은 지연, 높은 대역폭, **낮은 tail, 경로 예측성** |

전통적 TCP/IP는 데이터가 App memory → kernel buffer → NIC buffer를 오가며 **복사**되어 병목이 된다.

> **Gb/s vs GB/s 혼동 주의**: 네트워크 스펙은 보통 **Gb/s**(기가비트), GPU 메모리/tensor는 **GB/s**(기가바이트). 8배 차이다. `400Gb/s` NIC ≈ `50GB/s`, `900GB/s` NVLink ≈ `7,200Gb/s`.

### 네트워크 병목은 GPU idle time으로 보인다

네트워크 문제는 "느립니다" 로그가 아니라 **GPU utilization이 출렁이는** 증상으로 더 자주 나타난다. "GPU를 더 넣었는데 왜 선형으로 빨라지지 않는가"라는 질문은 결국 네트워크 질문으로 바뀐다.

---

## 2. 서버 내부 통신 vs 서버 간 통신

GPU끼리 주고받는다고 모두 같은 네트워크를 타는 게 아니다.

```
서버 내부: GPU → NVLink/NVSwitch/PCIe → GPU
서버 간:   GPU → PCIe → NIC/HCA → fabric → NIC/HCA → PCIe → GPU
```

### 2.1 내부 통신

| 경로 | 내용 | 직관 |
|---|---|---|
| NVLink | GPU↔GPU 고대역폭 직접 연결 | GPU끼리 가까운 전용 통로 |
| NVSwitch | 여러 GPU의 NVLink를 switch처럼 연결 | 서버 내부 GPU fabric |
| PCIe P2P | PCIe topology상 GPU 직접 접근 가능 시 | 같은 서버 내 범용 I/O 경로 |

### 2.2 서버 간 통신

NIC/HCA + Fabric switch가 필요하고, 주소·경로·혼잡 제어·재전송·flow control을 잘 해야 한다. 중요 질문 4가지:
1. GPU memory → NIC/HCA로 데이터가 어떻게 이동하는가
2. NIC/HCA가 **CPU를 거치지 않고** 데이터를 읽고 쓸 수 있는가
3. fabric이 loss·congestion·path imbalance를 어떻게 다루는가
4. NCCL이 실제로 기대한 network path를 골랐는가

### 2.3 Scale-up vs Scale-out

- **Scale-Up**: 서버 1대/Rack-scale 안에서 GPU를 더 촘촘히 잇거나 대역폭 키우기
- **Scale-Out**: 여러 서버를 Fabric으로 묶어 더 큰 클러스터 만들기

일반적 GPU 클러스터 fabric은 **Leaf-Spine(Fat-Tree)** 모양. Leaf는 GPU 서버와 가깝고, Spine은 여러 leaf를 묶어 노드 간 경로를 만든다.

설계 고려사항:
1. **bisection bandwidth** — 클러스터를 상하로 나눴을 때 대역폭이 비대칭이면 All-to-All 같은 전체 확산 트래픽에서 병목.
2. **GPU-NIC/rail locality** — GPU index·NIC·leaf/rail 배치가 맞으면 불필요한 PCIe/NUMA hop과 cross-rail 이동을 줄인다. 어긋나면 같은 링크 속도에서도 성능이 눈에 띄게 떨어진다.

---

## 3. RDMA — CPU를 덜 거치고 메모리 사이를 직접 잇는 모델

**RDMA(Remote Direct Memory Access)**: 원격 시스템 메모리에 직접 접근해 읽고 쓰는 통신 모델. 핵심은 **통신 과정에서 CPU와 kernel copy를 최대한 제거**하는 것. 애플리케이션은 등록된 메모리와 work request만 준비하고, 실제 데이터 이동은 NIC/HCA가 수행한다.

```
전통적 복사 경로:  App mem → kernel buf → NIC buf → network → (원격에서 역순)
RDMA 경로:        Registered mem → NIC/HCA가 직접 읽고 씀 → fabric → remote registered mem
```

### 주요 용어

| 객체 | 의미 | 설명 |
|---|---|---|
| **PD** | Protection Domain | 어떤 QP와 MR이 서로 접근 가능한지 묶는 보호 범위 |
| **MR** | Memory Region | NIC/HCA가 접근하도록 등록한 메모리 |
| **QP** | Queue Pair | 통신 endpoint. Send Queue + Receive Queue |
| **SQ/RQ** | Send/Receive Queue | 보낼/받을 work request·buffer를 올리는 큐 |
| **CQ** | Completion Queue | 작업 완료 결과가 쌓이는 큐 |
| **WR/WC** | Work Request/Completion | 작업 지시 / 처리 결과 |
| **HCA** | Host Channel Adapter | InfiniBand/RDMA 통신 처리 장치 |

### verbs 흐름

```
MR 등록 → QP 생성 → peer 정보 교환(QPN/주소/rkey) → WR post
  → HCA/NIC 전송 → remote HCA/NIC 처리 → CQ에서 완료 확인
```

> 포인트: **"보내기 함수 호출이 끝났다" ≠ "네트워크 작업이 끝났다"**. 완료는 반드시 CQ에서 확인한다.

---

## 4. InfiniBand

InfiniBand는 HPC/AI용 **고속 connection을 위한 전용 고성능 Fabric/프로토콜**. 낮은 지연·높은 대역폭으로 대규모 학습 클러스터에서 자주 등장.

> **RDMA ≠ InfiniBand 구분**
> - RDMA = 원격 메모리 접근 *모델*
> - InfiniBand = RDMA를 낮은 지연·높은 대역폭으로 실어 나르는 *프로토콜·경로*. 데이터 전송뿐 아니라 관리·주소·경로 설정을 위한 **Control plane**도 포함.

### Fabric 구성요소

| 구성요소 | 역할 |
|---|---|
| **HCA** | 서버의 IB endpoint. QP·MR 기반 전송 처리 |
| **IB switch** | fabric 안에서 packet 전달 |
| **Subnet Manager** | 노드 발견, 주소 배정, route 구성 |
| **LID/GID** | fabric 안 endpoint 식별 주소 |

> NDR IB 400Gb/s(≈50GB/s)라 해도, 실제 collective 성능은 link 수, rail 구성, message size, **HCA/GPU locality**, fabric congestion에 좌우된다. "IB를 도입했다"보다 **"각 GPU가 어느 HCA와 얼마나 가까운가"** 와 **"collective traffic이 fabric에 어떻게 퍼지는가"** 가 더 중요하다.

---

## 5. RoCE / RoCEv2 — Ethernet 위의 RDMA

**RoCE(RDMA over Converged Ethernet)**: RDMA semantics를 Ethernet 위에서 제공.

| 항목 | InfiniBand | RoCEv2 |
|---|---|---|
| Fabric 성격 | 전용 switched fabric | Ethernet/IP fabric |
| 관리 모델 | Subnet Manager 중심 | Ethernet/L3 운영 모델 중심 |
| 주소 | LID/GID | GID, IP, UDP encapsulation |
| API 표면 | RDMA verbs | 같은 verbs를 쓰되 주소·경로가 Ethernet/IP 모델에 맞춰 달라짐 |

- RoCEv2는 RDMA operation을 L3에서 전달하기 위해 **UDP/IP encapsulation** 사용. well-known **UDP dst port = 4791**. packet capture·ACL·QoS rule을 볼 때 중요.
- 같은 verbs surface여도 **GID, MTU, PFC, ECN/DCQCN** 같은 fabric contract가 맞아야 제대로 동작.

```
RDMA Work Request → RoCE/RDMA headers → UDP(dst 4791) → IP → Ethernet frame → Ethernet/IP fabric
```

---

## 6. NCCL과 Collective Communication

### NCCL

PyTorch가 "다 같이 gradient를 맞춰"라고 하면, **NCCL**이 실제로 어느 GPU가 누구와 어떤 순서로 데이터를 주고받을지 정하는 **진행 요원**이다. NCCL(NVIDIA Collective Communications Library)은 topology-aware한 inter-GPU 통신을 가속하는 런타임.

```
Framework(DDP/FSDP) → Process group → NCCL communicator
  → Algorithm 선택(ring/tree/channel) → Transport path(NVLink/PCIe/IB Verbs/IP socket/EFA plugin)
```

NCCL이 하는 일:
- 여러 GPU/rank가 참여하는 collective 실행
- GPU·NIC topology를 보고 경로 선택
- 메시지 크기·topology에 따라 ring/tree/channel 실행 방식 선택
- CUDA stream과 통합

### Collective는 모든 rank가 함께 호출해야 완성된다

```
Rank 0~3 모두: ncclAllReduce(... count=N, dtype=float16 ...)
→ 모두 같은 collective에 들어와야 전체 operation이 완성된다
```

### Collective를 Traffic Shape로 이해하기

| Collective | 설명 | 주요 워크로드 |
|---|---|---|
| **AllReduce** | 모든 rank 값을 reduce → 결과를 모두에게 | data parallel gradient sync |
| **AllGather** | 각 rank 조각을 모아 모두에게 | tensor/model parallel 결과 조립 |
| **ReduceScatter** | reduce 결과를 rank별 조각으로 나눠 | optimizer state sharding, FSDP/ZeRO |
| **All-to-All** | 각 rank가 모든 rank에 서로 다른 조각을 | MoE expert dispatch, seq/expert parallel |

- `All` = 모든 rank가 결과를 받음 / `Reduce` = 합·최댓값으로 합침 / `Gather` = 조각 모음 / `Scatter` = 나눠 줌
- 실제 대규모 학습은 한 step 안에서 TP·PP·DP·FSDP·EP가 섞이며 여러 collective가 서로 다른 rank group에서 동시에 나타난다.

### Collective별 데이터 흐름

**AllReduce** — 모든 rank 값을 reduce 후 같은 결과를 모두 받음 (분산 학습에서 가장 흔함)
```
입력 R0:[a0] R1:[a1] R2:[a2] R3:[a3]
결과 모든 rank: [a0+a1+a2+a3]
```

**AllGather** — 각 rank 조각을 모아 전체를 모두에게. 결과 buffer가 rank 수에 비례해 커지므로 메모리·대역폭 비용 주의.
```
입력 R0:[x0] R1:[x1] R2:[x2] R3:[x3]
결과 모든 rank: [x0 x1 x2 x3]
```

**All-to-All** — 각 rank가 목적지별 조각을 가지고, 모든 rank가 자기에게 온 조각을 받음. MoE에서 token을 expert rank로 보낼 때 핵심. 패턴이 복잡하고 경로가 많이 벌어져 **fabric 설계 영향이 크다**.
```
R0:[to0:a0 to1:a1 to2:a2 to3:a3] ... → R0:[a0 b0 c0 d0] (각 rank가 모든 rank에게서 받음)
```

**Ring AllReduce** — 큰 메시지를 chunk로 나눠 ring을 따라 돌림.
```
1단계 ReduceScatter: ring(R0→R1→R2→R3→R0)으로 chunk를 돌리며 같은 shard를 reduce → 각 rank가 reduce된 shard 1개 보유
2단계 AllGather:     reduce된 shard를 다시 ring으로 돌려 모든 rank가 전체 결과 보유
```
일반적으로 **ring은 대역폭을 잘 채우고, tree는 작은 메시지·지연 민감 구간에 유리**. 실제 선택은 구현·topology·메시지 크기에 따라 NCCL runtime이 다룬다. runtime이 고른 경로가 topology와 안 맞으면 성능이 떨어진다.

---

## 요약

1. AI에서 네트워크는 계산이 끝난 GPU들을 다음 step으로 보내는 **동기화 장치**다. 한 rank의 지연이 전체 step time으로 커진다.
2. 대규모 collective에서 전통적 **TCP/IP의 copy·CPU/kernel path가 병목**이 되기 쉽다.
3. **서버 내부 = NVLink/NVSwitch/PCIe**, **서버 간 = NIC/HCA + InfiniBand/RoCE/EFA fabric**.
4. **RDMA**는 CPU 개입·copy를 줄이는 데이터 이동 모델. MR/QP/CQ/WR가 기본 언어.
5. **InfiniBand**는 RDMA와 잘 맞는 전용 고성능 switched fabric. Subnet Manager, LID/GID까지 봐야 한다.
6. **RoCEv2**는 Ethernet/IP(UDP 4791) 위 RDMA. GID·MTU·PFC·ECN/DCQCN fabric contract가 맞아야 한다.
7. **NCCL**은 collective 런타임. AllReduce/AllGather/ReduceScatter/All-to-All은 각각 데이터 이동 모양이 다르고, 어떤 collective가 많은지가 workload 병목을 크게 바꾼다.
