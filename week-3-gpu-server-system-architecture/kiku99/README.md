# Week 3. GPU 서버 시스템 아키텍처 이해

## overview

GPU는 **단독으로 동작하지 못하고 서버에 장착되어 동작**한다. GPU 성능을 최대로 끌어내려면 서버 안에서 CPU·메모리·PCIe·NIC와 **어떻게 연결되고, 데이터가 어떤 흐름으로 흐르며, 어디서 병목이 생기는지**를 이해해야 한다. 이 주차는 ① Dual-CPU 서버 구조, ② NUMA, ③ PCIe/NVLink/NVSwitch, ④ NUMA Locality(`nvidia-smi topo -m`) 순으로 다룬다.

---

## 1. GPU Server

GPU 서버(예: HPE XD670)는 다수의 GPU + CPU + 대량의 DRAM + 여러 NIC + PCIe 스위치를 한 섀시에 담는다. 핵심 질문은 항상 같다: **데이터가 GPU memory ↔ NIC ↔ 다른 GPU로 갈 때 어떤 경로(hop)를 거치는가.**

---

## 2. GPU 서버 시스템 아키텍처 (Dual-CPU)

```
Host Memory #0          Host Memory #1
     │                       │
   CPU #0                  CPU #1
 (Mem Ctrl)              (Mem Ctrl)
 (PCIe Root Complex)     (PCIe Root Complex)
   │  │  │  │              │  │  │  │   ← Root Ports
   ▼                       ▼
PCIe Switch #0          PCIe Switch #1
 │       │               │       │
NIC#0~3  GPU#0~3        GPU#4~7  NIC#4~7
```

### 주요 구성요소

| 구성요소 | 설명 |
|---|---|
| **PCIe Root Complex** | CPU 칩 내부에 존재. CPU Core ↔ PCIe Bus를 연결하는 인터페이스 |
| **PCIe Root Port** | Root Complex 내 논리적 포트. PCIe 버스가 시작되는 논리적 경계점. 부팅 시 BIOS가 각 Root Port 단위로 PCI Bus 번호를 부여하며, 각 Port는 독립적 |
| **Memory Controller** | CPU 칩 내부에 존재. CPU Core ↔ DRAM을 연결. 독립된 전용 버스로 DRAM과 연결 |
| **Host Memory** | CPU가 사용하는 일반 System Memory (DRAM) |

핵심: GPU·NIC·NVMe 같은 장치는 모두 **PCIe**로 연결되어 인식되고, 각 CPU는 자기 쪽 메모리·PCIe 트리를 거느린다.

---

## 3. NUMA (Non-Uniform Memory Access)

멀티 프로세서 시스템에서 **CPU가 메모리에 접근하는 경로·시간이 메모리의 물리적 위치에 따라 달라지는** 아키텍처.

```
┌─── NUMA Node #0 ──────┐   ┌─── NUMA Node #1 ──────┐
│ Host Mem#0            │   │ Host Mem#1            │
│ CPU#0 (MemCtrl, RC)   │◄─►│ CPU#1 (MemCtrl, RC)   │
│ PCIe Switch#0         │UPI│ PCIe Switch#1         │
│ GPU#0~3, NIC#0~3      │   │ GPU#4~7, NIC#4~7      │
└───────────────────────┘   └───────────────────────┘
```

### 3.1 주요 개념

1. **지역성 (Locality)**
   - 각 CPU는 자신의 Memory Controller에 연결된 Host Memory에 접근하는 게 더 빠르다 (`CPU#0 → MemCtrl → Host Mem#0`).
   - CPU#0이 CPU#1의 Host Mem#1에 접근하는 것도 가능하지만 **UPI를 거쳐야 한다** → 대역폭 줄고 레이턴시 늘어남.
2. **PCIe 디바이스도 NUMA의 영향을 받는다**
   - PCIe 디바이스가 어느 CPU의 Root Complex에 연결됐는지에 따라 **NUMA Node가 결정**된다. 다른 NUMA Node에 접근할 땐 UPI를 거쳐 성능 저하.
   - **NUMA Node** = 하나의 CPU 소켓 + 거기 직접 연결된 모든 구성요소(코어, 캐시, Memory Controller, Host Memory, Root Complex, PCIe 디바이스)의 묶음.
3. **Hop 수가 성능을 결정한다** — GPU 서버 구성요소 간 통신 시 거치는 hop 수가 성능을 좌우.

### 3.2 주요 데이터 패스

| 패스 | 경로 |
|---|---|
| CPU → Host Memory (**Local**) | `CPU#0 Core → Memory Controller → Host Mem#0` |
| CPU → Host Memory (**Remote**) | `CPU#0 Core → UPI → Memory Controller → Host Mem#1` |
| PCIe 디바이스 → Host Memory | `GPU·NIC → PCIe Switch → PCIe Root Port → Root Complex → Memory Controller → DRAM` |

---

## 4. PCIe, NVLink, NVSwitch

서버 내 구성요소를 잇는 Bus·통로들. 각각 어떤 트래픽 패스에 쓰이는지가 핵심.

### 4.1 PCIe

- CPU가 서버 내 장치를 **인식하고 명령하는 기본 통로**.
- GPU, NIC, NVMe 같은 장치는 모두 PCIe로 연결되어 인식된다.
- **Host Memory ↔ GPU Memory 간 데이터 이동**의 기본 통로.
- 서버 내 CPU가 모든 장치를 감지·제어하는 필수 통로.

### 4.2 NVLink

주로 **GPU 간 연결에 쓰는 GPU 전용 고속 통로**. 같은 서버 내 GPU끼리 데이터를 주고받을 때 **PCIe를 거치지 않고** 고대역폭 NVLink로 더 빠르게 통신한다.

| 세대 | 아키텍처 | GPU당 총 대역폭 | 최대 연결 수 | 링크당 대역폭 |
|---|---|---|---|---|
| Gen 4 | Hopper | 900 GB/s | 18 Lines | 50 GB/s |
| Gen 5 | Blackwell | 1,800 GB/s | 18 Lines | 100 GB/s |
| Gen 6 | Rubin | 3,600 GB/s | 36 Lines | 100 GB/s |

### 4.3 NVSwitch

여러 NVLink를 Switch로 묶어, **어느 GPU 쌍이든 고속 통신이 가능하도록 서버 내에서 Fabric을 구성**한다. (DGX-1 P100 → DGX-2 V100 → DGX A100 → DGX H100으로 갈수록 NVSwitch 기반 full-mesh로 발전)

---

## 5. NUMA Locality — `nvidia-smi topo -m`

NUMA의 지역성과 hop을 기준으로 **GPU·NIC 간 연결 수준을 분류**한 기준. 이를 해석하면 서버 내 디바이스 연결 토폴로지를 이해하고, 워크로드에서 **어떤 디바이스를 함께 써야 할지** 판단할 수 있다.

### 연결 수준 레이블 (가까운 순 → 먼 순)

| Label | 의미 | 예시 |
|---|---|---|
| **NV#** | NVLink로 직접 연결 (PCIe 우회, # = 본딩된 NVLink 수) | GPU0 ~ GPU1 |
| **PIX** | 한 개의 PCIe bridge를 지나는 경로 (동일 PCIe Switch 하위) | GPU0 ~ NIC0 |
| **PXB** | 두 개 이상 PCIe Bridge를 지나는 경로 (동일 Switch 하위, Host Bridge 경유) | GPU0 ~ NIC1 |
| **NODE** | 동일 NUMA 노드 내부, Host Bridge(Root Complex) 경유 (다른 PCIe Switch) | GPU1 ~ NIC3 |
| **SYS** | 다른 NUMA Node를 지나는 경로, **UPI 인터커넥트 경유** | GPU2 ~ NIC4 |
| PHB | PCIe Host Bridge(보통 CPU)를 지나는 경로 | — |

> 성능 우선순위: **NV# > PIX > PXB > NODE > SYS**. SYS는 UPI까지 타므로 가장 느리다.

### `nvidia-smi topo -m` 읽는 법

명령은 GPU·NIC 간 토폴로지를 매트릭스로 출력한다. 행렬의 각 칸이 위 레이블(NV18, PIX, PXB, NODE, SYS…)이고, `CPU Affinity`/`NUMA Affinity` 열로 각 GPU가 어느 NUMA Node·코어 범위에 속하는지 보여준다.

자료의 8-GPU/10-NIC 예시에서 얻는 결론:
- 모든 GPU 쌍은 **NV18**(NVLink 18본 본딩)으로 연결 → GPU↔GPU는 NVSwitch full-mesh.
- GPU0~3은 NUMA Node 0(`0-55,112-167`), GPU4~7은 NUMA Node 1(`56-111,168-223`)에 속함.
- **GPUDirect RDMA 최적 쌍(PIX)**: 같은 PCIe Switch + 같은 NUMA Node에 있는 GPU-NIC 쌍이 최적이다.

| GPU | 최적 NIC | 홉 | PCIe Switch | NUMA |
|---|---|---|---|---|
| GPU0 | NIC0 (mlx5_0) | PIX | Switch A | Node 0 |
| GPU1 | NIC2 (mlx5_2) | PIX | Switch A | Node 0 |
| GPU2 | NIC3 (mlx5_3) | PIX | Switch B | Node 0 |
| GPU3 | NIC4 (mlx5_4) | PIX | Switch B | Node 0 |
| GPU4 | NIC5 (mlx5_5) | PIX | Switch C | Node 1 |
| GPU5 | NIC7 (mlx5_7) | PIX | Switch C | Node 1 |
| GPU6 | NIC8 (mlx5_8) | PIX | Switch D | Node 1 |
| GPU7 | NIC9 (mlx5_9) | PIX | Switch D | Node 1 |

→ GPUDirect RDMA를 쓸 때는 이 **PIX 쌍**을 매칭해야 PCIe/NUMA hop과 cross-NUMA(UPI) 이동을 피해 성능을 살린다.

---

## 요약

- GPU는 서버 안에서 **CPU(Root Complex/Memory Controller) — PCIe Switch — GPU/NIC** 트리로 연결된다. 데이터 경로의 hop 수가 성능을 좌우한다.
- **NUMA**: CPU 소켓마다 자기 메모리·PCIe 디바이스를 거느린 NUMA Node가 있고, 다른 노드 접근은 **UPI를 타 느려진다**. GPU/NIC도 NUMA 소속이 정해진다.
- **PCIe**는 제어·인식·Host↔GPU 데이터 이동의 기본 통로, **NVLink**는 GPU↔GPU 전용 고속 통로(900GB/s~3.6TB/s), **NVSwitch**는 이를 서버 내 full-mesh fabric으로 묶는다.
- `nvidia-smi topo -m`으로 **NV# > PIX > PXB > NODE > SYS** 연결 수준을 읽고, GPUDirect RDMA는 같은 Switch·NUMA의 **PIX GPU-NIC 쌍**을 매칭해야 한다.
