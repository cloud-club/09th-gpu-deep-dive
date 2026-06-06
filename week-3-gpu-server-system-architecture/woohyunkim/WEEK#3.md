# Week#3 : GPU 서버 시스템 아키텍처 이해
---

## 01. GPU Server란

GPU는 단독으로 동작할 수 없으며, 서버에 장착되어 동작.  
GPU 성능을 최대한 활용하기 위해서는 서버 내 CPU, 메모리, PCIe 버스 등과 어떻게 연결돼 있는지, 어디서 병목이 발생하는지를 이해하는 것이 중요.

---

## 02. GPU 서버 시스템 아키텍처 (Dual-CPU)

### 02.01. 구성요소

| 컴포넌트 | 위치 | 설명 |
|---|---|---|
| **PCIe Root Complex** | CPU 칩 내부 | CPU Core ↔ PCIe Bus를 연결하는 인터페이스 |
| **PCIe Root Port** | Root Complex 내 논리 포트 | PCIe 버스가 시작되는 논리적 경계점. 부팅 시 BIOS가 Root Port 단위로 PCI Bus 번호 부여. 각 Port는 독립적 |
| **Memory Controller** | CPU 칩 내부 | CPU Core ↔ DRAM 연결. 독립된 전용 버스로 DRAM과 연결 |
| **Host Memory** | CPU 외부 DRAM | CPU가 사용하는 일반 System Memory |

### 전체 구조 (Dual-CPU 서버 기준)

```
Host Memory #0                    Host Memory #1
      │                                 │
   CPU #0                            CPU #1
   ├─ Memory Controller              ├─ Memory Controller
   └─ PCIe Root Complex  ←── UPI ──→  PCIe Root Complex
        └─ Root Port x4                └─ Root Port x4
               │                              │
    ┌──────────┴──────────┐       ┌───────────┴───────────┐
PCIe Switch A  PCIe Switch B   PCIe Switch C  PCIe Switch D
 ├─ GPU #0       ├─ GPU #2      ├─ GPU #4       ├─ GPU #6
 ├─ GPU #1       ├─ GPU #3      ├─ GPU #5       ├─ GPU #7
 ├─ NIC #0       ├─ NIC #3      ├─ NIC #5       ├─ NIC #8
 └─ NIC #2       └─ NIC #4      └─ NIC #7       └─ NIC #9
        └──── NIC #1 (NODE) ────┘       └──── NIC #6 (NODE) ────┘
```

> NIC1, NIC6은 동일 NUMA Node 내 다른 PCIe Switch와 NODE 관계로 연결된다.

---

## 03. NUMA (Non-Uniform Memory Access)

> 멀티 프로세서 시스템에서 CPU가 메모리에 접근하는 경로 및 시간이 메모리의 물리적 위치에 따라 달라지는 아키텍처

### 03.01. 주요 개념

**1. 지역성 (Locality)**

- 각 CPU는 자신의 Memory Controller에 연결된 Host Memory에 접근하는 것이 더 빠르다.
  - 가장 빠른 경로: `CPU#0 Core → Memory Controller → Host Memory#0`
- CPU#0이 CPU#1에 연결된 Host Memory#1에 접근할 경우, **UPI(Ultra Path Interconnect)를 경유**해야 한다.
  - 결과: 대역폭 감소 + 레이턴시 증가

**2. PCIe 디바이스도 NUMA의 영향을 받는다**

- PCIe 디바이스가 어느 CPU의 PCIe Root Complex에 연결됐는지에 따라 **NUMA Node** 소속이 결정된다.
  - Numa Node는 cpu 칩 단위
- 다른 NUMA Node에 접근 시 UPI를 경유 → 성능 저하 발생.

> **NUMA Node 정의**: 하나의 CPU 소켓과 해당 소켓에 직접 연결된 모든 구성요소의 묶음  
> (CPU 코어, 캐시, 메모리 컨트롤러, 호스트 메모리, 루트 컴플렉스, PCIe 디바이스)

**3. Hop 수가 성능을 결정한다**

- GPU 서버 구성요소 간 통신 시 거치는 홉(Hop) 수가 성능을 직접 결정한다.

### 03.02. 주요 데이터 패스

| 경로 | 홉 구성 |
|---|---|
| CPU - Host Memory (Local) | `CPU#0 Core → Memory Controller → Host Memory#0` |
| CPU - Host Memory (Remote) | `CPU#0 Core → UPI → Memory Controller → Host Memory#1` |
| PCIe 디바이스 → Host Memory | `GPU·NIC → PCIe Switch → PCIe Root Port → PCIe Root Complex → Memory Controller → DRAM` |

---

## 04. PCIe, NVLink, NVSwitch

GPU 서버 내 구성요소들을 연결하는 버스 및 통로.  
각각이 어떤 트래픽 패스에 활용되는지 구분하는 것이 핵심이다.

### 04.01. PCIe

- CPU가 서버 내 장치를 **인식하고 명령하는 기본 통로**
- 서버 내 GPU, NIC, NVMe 같은 장치들은 모두 PCIe로 연결되어 인식된다.
- Host Memory ↔ GPU Memory 간 데이터 이동도 PCIe가 기본 통로.
- **PCIe는 서버 내 CPU가 모든 장치를 감지·제어하는 필수 경로다.**

### 04.02. NVLink

- **주로 GPU 간 연결에 사용되는 GPU 전용 고속 통로**
- 같은 서버 내 GPU 간 데이터 송수신 시 PCIe를 거치지 않고, 고대역폭 NVLink로 더 빠르게 통신한다.

| 세대 | GPU | 총 NVLink 대역폭 | 최대 연결 수 | 라인당 대역폭 |
|---|---|---|---|---|
| Gen 4 | Hopper (H100) | 900 GB/s | 18 Lines | 50 GB/s |
| Gen 5 | Blackwell (B200) | 1,800 GB/s | 18 Lines | 100 GB/s |
| Gen 6 | Rubin | 3,600 GB/s | 36 Lines | 100 GB/s |

### 04.03. NVSwitch

- 여러 NVLink를 Switch로 묶어, **어느 GPU 쌍이든 고속 통신이 가능하도록** 서버 내에서 Fabric을 구성한다.
- GPU 간 All-to-All 연결을 NVLink 속도로 보장.

---

## 05. NUMA Locality

NUMA의 지역성과 홉을 기준으로 GPU·NIC 간 연결 수준을 분류한 기준.  
이를 해석하여 서버 내 디바이스 간 연결 토폴로지를 이해하고, 워크로드에서 어떤 디바이스를 사용해야 할지 판단할 수 있다.

### 레이블 의미표

| Label | 의미 | 예시 |
|---|---|---|
| **PIX** | PCIe Bridge 1개 경유 (동일 PCIe Switch 하위) | GPU0 ↔ NIC0 |
| **PXB** | PCIe Bridge 2개 이상 경유 (동일 PCIe Switch 하위, Host Bridge 경유 없음) | GPU0 ↔ NIC1 |
| **NODE** | 동일 NUMA Node 내, Host Bridge(Root Complex) 경유 (다른 PCIe Switch) | GPU1 ↔ NIC3 |
| **SYS** | 다른 NUMA Node 경유, UPI 인터커넥트 사용 | GPU2 ↔ NIC4 |
| **NV#** | NVLink # 개로 직접 연결, PCIe 우회 | GPU0 ↔ GPU1 |

> 성능 순서: **NV# > PIX > PXB > NODE > SYS**

### 05.01. nvidia-smi topo -m

NVIDIA GPU 서버에서 GPU·NIC 간 토폴로지를 확인하는 명령어.

```bash
$ nvidia-smi topo -m
```
- **GPU0~GPU7 상호간**: 모두 `NV18` → 18개 NVLink 묶음으로 All-to-All 직결
- **GPU0 ↔ NIC0**: `PIX` → 같은 PCIe Switch 하위, GPUDirect RDMA 최적 경로
- **GPU0 ↔ NIC5**: `SYS` → 다른 NUMA Node, UPI 경유, 성능 저하
- **NUMA Affinity**: GPU0~3 → Node 0 (CPU Affinity: 0-55, 112-167), GPU4~7 → Node 1 (CPU Affinity: 56-111, 168-223)

### GPUDIRECT RDMA 최적 쌍 (PIX 기준)

| GPU | 최적 NIC | 관계 | PCIe Switch | NUMA Node |
|---|---|---|---|---|
| GPU 0 | NIC 0 (mlx5_0) | PIX | Switch A | Node 0 |
| GPU 1 | NIC 2 (mlx5_2) | PIX | Switch A | Node 0 |
| GPU 2 | NIC 3 (mlx5_3) | PIX | Switch B | Node 0 |
| GPU 3 | NIC 4 (mlx5_4) | PIX | Switch B | Node 0 |
| GPU 4 | NIC 5 (mlx5_5) | PIX | Switch C | Node 1 |
| GPU 5 | NIC 7 (mlx5_7) | PIX | Switch C | Node 1 |
| GPU 6 | NIC 8 (mlx5_8) | PIX | Switch D | Node 1 |
| GPU 7 | NIC 9 (mlx5_9) | PIX | Switch D | Node 1 |

---
