# 4주차 - AI 워크로드 네트워크 정리

AI 워크로드 네트워크는 계층적으로 동작한다. 프레임워크가 직접 패킷을 만들지 않는다.

1. 프레임워크(PyTorch/Megatron)가 collective 요청
2. NCCL 같은 런타임이 GPU/네트워크 topology를 보고 경로 선택
3. RDMA verbs / libfabric / IB / RoCE / EFA 가 실제 전송 담당

| 층 | 핵심 질문 | 용어 |
|---|---|---|
| 프레임워크 | 어느 rank들이 같이 학습? | DDP, FSDP, TP, EP |
| 런타임 | 어떤 collective를 어떤 알고리즘으로? | NCCL, MPI, NVSHMEM |
| 이동 모델 | CPU/커널 복사를 얼마나 줄이나? | RDMA, GPUDirect RDMA |
| fabric | 패킷이 어느 네트워크 위로? | IB, RoCEv2, EFA, Ethernet |
| 운영 검증 | 느림/hang/loss가 어느 층? | NCCL_DEBUG, GID, PFC, ECN |

층을 나누는 이유는, "AllReduce 느림" 증상은 하나여도 원인은 여러 층에 흩어져 있기 때문이다. rank 배치 문제일 수도, NCCL이 NIC를 잘못 골랐을 수도, RoCE PFC/ECN이 안 맞을 수도 있다. 층을 나누면 디버깅 질문도 같이 나뉜다.


## 1. 왜 네트워크가 중요한가

GPU 한 장은 빠르다. 그런데 여러 장이 같은 모델을 학습하면 매 step마다 결과를 맞춰야 한다. 데이터 병렬은 gradient를 합치고, tensor parallel은 activation을 주고받고, MoE는 token을 expert가 있는 rank로 보낸다.

학습이 동기식이라 한 rank가 늦으면 나머지가 다 기다린다. 평균 대역폭이 좋아도 tail latency가 길면 step time이 흔들리고, 비싼 GPU가 놀게 된다. 증상은 "네트워크 느림" 로그가 아니라 GPU utilization 출렁임으로 나타난다. "GPU 더 넣었는데 왜 선형으로 안 빨라지나"가 사실은 네트워크 질문이다.

기존 네트워크가 부적절한 이유는, 기존 DC망이 north-south(사용자↔서버) 중심이고 요청끼리 독립적인 데 반해, AI 학습은 east-west(서버↔서버)이고 step 경계에서 동기 burst로 몰리기 때문이다. 게다가 TCP/IP는 App/커널/NIC 버퍼를 오가며 복사해서 CPU가 병목이 된다.

단위 주의: 네트워크는 Gb/s, GPU 메모리는 GB/s로 표기된다. 8배 차이다. 400Gb/s NIC ≈ 50GB/s, 900GB/s NVLink ≈ 7200Gb/s.


## 2. 서버 내부 통신 vs 서버 간 통신

GPU끼리 데이터를 주고받아도 다 같은 길을 타는 게 아니다.

서버 내부 (같은 서버 GPU끼리)
- NVLink: GPU-GPU 고대역폭 직접 연결 (전용 통로)
- NVSwitch: 여러 GPU의 NVLink를 스위치처럼 묶음 (내부 GPU fabric)
- PCIe P2P: PCIe상 GPU끼리 직접 접근 (범용 I/O)

서버 간 (다른 서버 GPU랑)
- NIC/HCA를 거쳐 fabric(IB/RoCE/EFA) 통과
- 경로: GPU → PCIe → NIC/HCA → fabric → NIC/HCA → PCIe → GPU

NIC/HCA와 fabric은 다르다. NIC/HCA는 서버에 박힌 출입구 부품(endpoint)이고, fabric은 서버들 사이의 도로망(스위치)이다. fabric은 보통 Leaf-Spine(Fat-Tree) 구조로, Leaf는 서버와 가깝고 Spine은 leaf들을 묶는다.

서버 간 통신에서 봐야 할 것
1. GPU 메모리에서 NIC/HCA까지 데이터가 어떻게 이동하나
2. NIC/HCA가 CPU를 안 거치고 읽고 쓰나
3. fabric이 loss/congestion/경로 불균형을 어떻게 다루나
4. NCCL이 기대한 경로를 골랐나

Scale-up은 서버/rack 안에서 GPU를 더 촘촘히 잇고 대역폭을 키우는 방향, Scale-out은 여러 서버를 fabric으로 묶어 큰 클러스터를 만드는 방향이다.

설계 고려사항
- bisection bandwidth: 클러스터를 반 갈랐을 때 그 사이 총 대역폭. 부족하면 All-to-All처럼 전체로 퍼지는 트래픽에서 병목
- GPU-NIC/rail locality: GPU index/NIC/leaf 배치가 잘 맞으면 불필요한 PCIe/NUMA hop을 줄임. 어긋나면 같은 링크 속도, 같은 NCCL 호출인데도 성능이 떨어짐


## 3. RDMA

RDMA(Remote Direct Memory Access)는 원격 메모리에 직접 접근해 읽고 쓰는 통신 모델이다. 핵심은 CPU와 커널 복사를 통신에서 최대한 제거하는 것이다.

- 전통 경로: App 메모리 → 커널 버퍼 → NIC 버퍼 → 네트워크 → (원격도 역순 복사)
- RDMA 경로: 등록 메모리(MR) → NIC/HCA가 직접 읽고 씀 → fabric → 원격 등록 메모리

주요 용어
- PD (Protection Domain): 어떤 QP/MR이 서로 접근 가능한지 묶는 범위
- MR (Memory Region): NIC/HCA가 접근하도록 등록한 메모리
- QP (Queue Pair): 통신 endpoint (Send Queue + Receive Queue)
- CQ (Completion Queue): 작업 완료 결과가 쌓이는 곳
- WR/WC: Work Request(작업 지시) / Work Completion(처리 결과)
- HCA (Host Channel Adapter): IB에서 통신을 처리하는 장치

verbs 흐름: MR 등록 → QP 생성 → peer 정보 교환(QPN/주소/rkey) → WR post → HCA/NIC 전송 → 원격 HCA/NIC 처리 → CQ에서 완료 확인

메모리를 등록(MR)해야만 NIC가 안전하게 접근한다. 완료는 CQ에서 확인하므로 "보내기 함수 끝남"과 "네트워크 작업 끝남"을 구분해야 한다.


## 4. InfiniBand

InfiniBand는 HPC/AI용 고속 connection 프로토콜이자 낮은 지연·높은 대역폭의 전용 고성능 fabric이다.

RDMA와 IB는 구분해야 한다. RDMA는 원격 메모리 접근 모델이고, IB는 그걸 잘 실어나르는 프로토콜·경로다. IB fabric에는 데이터 전송 외에 관리/주소/경로용 control plane도 있다.

구성 요소
- HCA: 서버의 IB endpoint, QP/MR 전송 처리
- IB switch: fabric 안에서 패킷 전달
- Subnet Manager: 노드 발견/주소 배정/route 구성
- LID/GID: fabric에서 endpoint를 식별하는 주소

NDR 400Gb/s는 환산하면 50GB/s지만, 실제 성능은 link 수, rail 구성, 메시지 크기, HCA-GPU locality, 혼잡에 따라 달라진다. 그래서 "IB 도입했다"보다 "GPU가 어느 HCA와 가까운가", "트래픽이 fabric 전체에 어떻게 퍼지나"가 더 중요하다.


## 5. RoCE / RoCEv2

RoCE는 RDMA semantics를 Ethernet 위에서 제공하는 방식이다. "그냥 이더넷"이 아니다.

| | InfiniBand | RoCEv2 |
|---|---|---|
| fabric | 전용 switched | Ethernet/IP |
| 관리 | Subnet Manager | Ethernet/L3 운영 모델 |
| 주소 | LID/GID | GID, IP, UDP encapsulation |
| API | RDMA verbs | verbs 그대로, 주소/경로만 Ethernet/IP식 |

RoCEv2는 RDMA operation을 L3(IP)에서 전달하려고 UDP/IP encapsulation을 쓴다.
패킷: WR → RoCE/RDMA header → UDP(dst 4791) → IP → Ethernet → fabric
well-known UDP port 4791은 packet capture/ACL/QoS를 볼 때 의미가 있다.


## 6. NCCL & Collective

NCCL은 collective communication runtime이다. PyTorch가 "다같이 gradient 맞춰"라고 하면, NCCL이 어느 GPU가 누구와 어떤 순서로 주고받을지 정하고 실행한다. GPU/NIC topology를 보고 경로를 고르고, 메시지 크기/topology에 따라 ring/tree/channel을 선택하며, CUDA stream과 통합된다.

collective는 모든 참여 rank가 같이 호출해야 완성된다. 혼자 보내는 함수가 아니다.

| Collective | 동작 | 사용처 |
|---|---|---|
| AllReduce | 모든 rank 값 reduce → 결과 모두에게 | DP gradient sync (최빈) |
| AllGather | 각 rank 조각 모아 모두에게 | TP/모델병렬 조립 |
| ReduceScatter | reduce 결과를 rank별 조각으로 분배 | optimizer sharding, FSDP/ZeRO |
| All-to-All | 각 rank가 서로 다른 조각 교환 | MoE expert dispatch |

단어: All=모두 받음, Reduce=합침, Gather=모음, Scatter=나눔. 큰 LLM은 TP+PP+DP에 FSDP/expert까지 섞어서 한 step에 여러 collective가 동시에 등장한다.

Ring AllReduce는 큰 메시지를 chunk로 쪼개서, 1단계 ReduceScatter(링을 돌며 같은 shard를 reduce, 각 rank가 shard 하나씩)와 2단계 AllGather(reduce된 shard를 다시 링으로 돌려 모든 rank가 전체 결과)로 수행한다. ring은 대역폭을 잘 채우고 tree는 작은 메시지나 지연 민감 구간에 유리하다. NCCL이 알아서 고르지만 topology와 안 맞으면 느려진다.



## GPU-NIC locality와 NUMA node

문서에서 GPU-NIC/rail locality가 어긋나면 "같은 링크 속도, 같은 NCCL 호출인데도 성능이 떨어진다"고 한 부분의 원인 중 하나가 NUMA다.

### NUMA란

NUMA는 서버 한 대 안에서 CPU가 메모리에 접근하는 거리가 균일하지 않다는 개념이다. 요즘 서버는 CPU 소켓이 2개 이상이고 각 CPU에 메모리가 따로 붙는다.

```
NUMA node 0              NUMA node 1
CPU0 - RAM0              CPU1 - RAM1
   \____ 인터커넥트(UPI 등) ____/
```

CPU0 → RAM0은 가깝고 빠르다(local). CPU0 → RAM1은 다리를 건너야 해서 느리다.

NUMA와 RDMA는 다른 개념이다. NUMA는 서버 한 대 안의 CPU↔로컬 RAM 거리 문제이고, RDMA는 서버와 서버 사이 메모리 전송 문제다. 무대가 다르지만 아래 지점에서 만난다.

### RDMA/GPU 통신과 엮이는 지점

RDMA 경로는 GPU → PCIe → NIC/HCA → fabric → … 인데, 이 GPU/NIC/CPU 메모리가 각각 어느 NUMA node에 붙어 있느냐가 성능을 좌우한다.

```
NUMA node 0          NUMA node 1
CPU0 RAM0            CPU1 RAM1
GPU0                 GPU2
NIC0                 NIC1
```

GPU0 데이터를 NIC0로 보내면(둘 다 node 0) NUMA 다리를 안 건너서 빠르다. GPU0 데이터를 NIC1로 보내면(node 0 → node 1) NUMA 인터커넥트를 건너서 느려진다. 같은 RDMA, 같은 NIC 속도인데 느린 경우의 원인이 이것이다. 즉 문서의 GPU-NIC locality는 사실상 "GPU와 NIC를 같은 NUMA node에 맞춰 붙여라"라는 뜻이다.

