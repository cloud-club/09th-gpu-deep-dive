## 1. NUMA와 UPI

### NUMA란

- 멀티 소켓 서버에서 메모리 접근 시간이 균일하지 않은 구조 (Non-Uniform Memory Access)
- 각 CPU가 자기 메모리 컨트롤러와 전용 메모리를 직접 보유
- 로컬 접근 = 빠름 / 원격 접근 = 느림

### UPI

- 두 CPU 소켓을 연결하는 고속 인터커넥트 (Intel)
- 반대편 메모리에 가려면 반드시 UPI를 건너야 함 → 대역폭↓, 지연↑
- 로컬: 메모리 전용 채널로 직접 연결 → 넓음
- 원격: UPI를 거침 → 좁음
- UPI가 좁은 이유: 범용 연결선이라 캐시 일관성 신호 등 다른 트래픽도 함께 처리, 물리적 자원도 제한적

### 원격 접근이 잦으면

데이터 이동 지연 → 연산 유닛이 대기 → 처리량 저하

→ 해결: **NUMA-aware 배치** (프로세스와 메모리를 같은 노드에 묶기)
→ 이 원칙은 GPU·NIC 등 PCIe 장치 배치 전략으로도 확장됨

> NUMA는 메모리 접근이 비균일한 구조이며, UPI를 건너는 원격 접근은 대역폭이 줄고 지연이 늘어 성능을 떨어뜨린다.
> 


---
<br>

## 2. NVLink vs PCIe

### PCIe

PCIe는 본래 **CPU ↔ 장치** 범용 통로. GPU끼리 통신하면

`GPU A → PCIe 스위치 / CPU → PCIe 스위치 → GPU B`

1. 우회 경로 → 지연↑
2. SSD·NIC와 통로 공유 → 경합
3. 대역폭 자체가 낮음

### NVLink (GPU 전용 직결 통로)

- CPU/스위치를 거치지 않고 GPU끼리 직접 연결
- HBM에서 곧바로 다른 GPU로 전송
- GPU 간 메모리 일관성 유지

### 

### NVSwitch — 다수 GPU 연결

- GPU 2~4개: NVLink 직접 연결 가능
- GPU 8개+: 모든 쌍 직접 연결 시 링크 수 폭발 (8개 완전 연결 = 28개 연결 필요)
- NVSwitch: 모든 GPU를 중앙 스위치에 연결 → All-to-All 풀 대역폭 통신

> PCIe는 CPU↔장치 범용 통로라 GPU 간 통신에 우회·공유·저대역폭 문제가 있고, NVLink는 GPU 전용 직결로 약 7배 대역폭을 제공하며, NVSwitch가 이를 다수 GPU로 확장한다.
> 

---
<br>

## 3. UPI vs NVLink


|  | UPI | NVLink |
| --- | --- | --- |
| 연결 대상 | CPU ↔ CPU | GPU ↔ GPU |
| 제조사 | Intel | NVIDIA |
| 주 목적 | 캐시 일관성 유지 | 대량 데이터 전송 |

### 왜 둘 다 필요한가 — 트래픽 성격이 다르다

- **CPU 간**: 데이터양 적음 (제어·분기가 본업) → UPI는 정확하고 일관된 소량 통신에 최적화
- **GPU 간**: 거대한 텐서 통째로 전송 → NVLink는 무조건 넓게에 최적화

<br>

### 대역폭 격차

| 항목 | UPI (Sapphire Rapids) | NVLink 4 (H100) |
| --- | --- | --- |
| 연결 대상 | CPU ↔ CPU | GPU ↔ GPU |
| 링크당 대역폭 | ~20+ GB/s | 50 GB/s |
| 링크 수 | 프로세서당 최대 4개 | GPU당 최대 18개 |
| 총 대역폭 | ~100 GB/s 수준 | 900 GB/s |
- 총 대역폭 기준 NVLink가 UPI의 **약 9배**
- NVLink는 Blackwell 1.8 TB/s로 계속 폭증, UPI는 보수적 증가

<br>

### NUMA 관점 연결 — NVSwitch가 UPI 병목을 우회

`nvidia-smi topo -m`의 `SYS` 라벨 = UPI를 건너야 하는 경로

GPU#0(Node0) ↔ GPU#4(Node1) 통신 시, NVLink가 없다면:

`GPU0 → PCIe → UPI(~100 GB/s) → PCIe → GPU4`

- GPU는 900 GB/s로 쏠 준비가 됐는데 중간 UPI가 100 GB/s → **9배 좁은 구간에서 병목**
- NVSwitch: GPU 통신을 NVLink Fabric 안에서 처리 → 노드 경계와 무관하게 풀 대역폭
- `topo -m`에서 GPU0~7이 전부 `NV18`로 직결된 출력 = UPI 병목을 우회한 상태

<br>


> UPI는 CPU 간 캐시 일관성에 최적화된 좁고 정확한 연결, NVLink는 GPU 간 대량 전송에 최적화된 넓은 연결(약 9배)이며, NVSwitch는 GPU 통신이 느린 UPI를 거치지 않게 우회시켜 NUMA 노드 경계의 병목을 제거한다.
> 
