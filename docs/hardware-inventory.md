# Hardware Inventory & Physical Topology: node-a & node-b

## 1. Executive Summary
- **Total Computing Units**: 2 nodes, 16 GPUs (8 GPUs per node).
- **GPU Architecture**: NVIDIA A800-SXM4-40GB (Ampere Architecture, SM80, Compute Capability 8.0).
- **Interconnect**:
  - Intra-node: Full-mesh NVLink 3 (NV8, 8 NVLinks per GPU pair, bonded).
  - Inter-node: 4 $\times$ 400 Gbps RoCEv2 (Mellanox ConnectX-7 MT41692) on high-speed RDMA network (`ens13f0np0`, `ens16f0np0`, `ens17f0np0`, `ens18f0np0`).
  - Management / Control Plane: 10 Gbps Ethernet (`bond0.2460` at 10.0.0.x).

---

## 2. Node Specifications

### 2.1 Node A (`node-a`) - Leader Node
- **Management IP**: `10.0.0.1` (bond0.2460)
- **RDMA IPs**: `172.16.1.2/31` (`ens13f0np0`), `172.5.4.2/31` (`ens16f0np0`), `172.6.4.2/31` (`ens17f0np0`), `172.7.4.2/31` (`ens18f0np0`)
- **CPU**: 2 $\times$ AMD EPYC 7K62 48-Core Processor (Total 96 physical cores, 192 vCPUs)
- **Host Memory**:
  - Total RAM: 1.0 TiB (1,031,374 MiB)
  - NUMA 0: 515,356 MiB (124,848 MiB free)
  - NUMA 1: 516,018 MiB (123,277 MiB free)
  - Available Memory: ~758 GiB
- **Local NVMe Storage**:
  - `/dev/nvme3n1p1` mounted at `/mnt/nvme3` (1.7 TiB NVMe SSD). Holds local model weights at `/mnt/nvme3/models/DeepSeek-V4.1-Flash-1017`.
- **Accelerators (8 $\times$ NVIDIA A800-SXM4-40GB)**:
  - GPU 0: PCIe `0000:26:00.0`, NUMA 0 (CPUs 0-47, 96-143), 40,960 MiB VRAM, 400W TDP
  - GPU 1: PCIe `0000:2C:00.0`, NUMA 0 (CPUs 0-47, 96-143), 40,960 MiB VRAM, 400W TDP
  - GPU 2: PCIe `0000:65:00.0`, NUMA 0 (CPUs 0-47, 96-143), 40,960 MiB VRAM, 400W TDP
  - GPU 3: PCIe `0000:6A:00.0`, NUMA 0 (CPUs 0-47, 96-143), 40,960 MiB VRAM, 400W TDP
  - GPU 4: PCIe `0000:A3:00.0`, NUMA 1 (CPUs 48-95, 144-191), 40,960 MiB VRAM, 400W TDP
  - GPU 5: PCIe `0000:A8:00.0`, NUMA 1 (CPUs 48-95, 144-191), 40,960 MiB VRAM, 400W TDP
  - GPU 6: PCIe `0000:E1:00.0`, NUMA 1 (CPUs 48-95, 144-191), 40,960 MiB VRAM, 400W TDP
  - GPU 7: PCIe `0000:E7:00.0`, NUMA 1 (CPUs 48-95, 144-191), 40,960 MiB VRAM, 400W TDP
- **MIG Status**: Disabled across all 8 GPUs. Full 80 SMs active per device.

### 2.2 Node B (`node-b` / `node-b`) - Subordinate Node
- **Management IP**: `10.0.0.2` (bond0.2460)
- **RDMA IPs**: `172.16.2.2/31` (`ens13f0np0`), `172.5.5.2/31` (`ens16f0np0`), `172.6.5.2/31` (`ens17f0np0`), `172.7.5.2/31` (`ens18f0np0`)
- **CPU**: 2 $\times$ AMD EPYC 7K62 48-Core Processor (96 physical cores, 192 vCPUs)
- **Host Memory**: 1.0 TiB total, ~867 GiB available.
- **Local NVMe Storage**:
  - `/dev/nvme2n1p1` mounted at `/mnt/nvme2` (1.7 TiB NVMe SSD). Holds local model weights at `/mnt/nvme2/models/DeepSeek-V4.1-Flash-1017`.
  - `/dev/nvme3n1p1` mounted at `/mnt/nvme3` (1.7 TiB NVMe SSD).
- **Accelerators (8 $\times$ NVIDIA A800-SXM4-40GB)**:
  - 8 $\times$ A800-SXM4-40GB, Compute Capability 8.0, 400W TDP, MIG Disabled.

---

## 3. GPU-GPU & NIC Topology Matrix

### 3.1 Intra-Node NVLink Mesh
Both nodes exhibit full-mesh SXM4 NVLink-3 topology:
```
      GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
GPU0    X   NV8  NV8  NV8  NV8  NV8  NV8  NV8
GPU1   NV8   X   NV8  NV8  NV8  NV8  NV8  NV8
GPU2   NV8  NV8   X   NV8  NV8  NV8  NV8  NV8
GPU3   NV8  NV8  NV8   X   NV8  NV8  NV8  NV8
GPU4   NV8  NV8  NV8  NV8   X   NV8  NV8  NV8
GPU5   NV8  NV8  NV8  NV8  NV8   X   NV8  NV8
GPU6   NV8  NV8  NV8  NV8  NV8  NV8   X   NV8
GPU7   NV8  NV8  NV8  NV8  NV8  NV8  NV8   X
```
- Total bi-directional NVLink bandwidth per GPU: 400 GB/s.
- P2P direct access supported with zero PCIe host-bridge crossing.

### 3.2 NIC to GPU NUMA Placement
- **NICs on NUMA Node 0**: `rocep47s0f0` (NIC0), `rocep48s0f0` (NIC1), `rocep110s0f0` (NIC2), `rocep111s0f0` (NIC3).
  Direct PCIe bridge (PXB) affinity to GPU 0, 1, 2, 3.
- **NICs on NUMA Node 1**: `rocep172s0f0` (NIC4), `rocep173s0f0` (NIC5), `rocep234s0f0` (NIC7), `rocep235s0f0` (NIC8).
  Direct PCIe bridge (PXB) affinity to GPU 4, 5, 6, 7.
- **Balanced RDMA Ingress/Egress**: 4 active 400 Gbps RoCE rails are balanced symmetrically across NUMA nodes and PCIe root complexes.

---

## 4. Benchmark Measurements

### 4.1 Intra-Node NCCL AllReduce (NVLink)
Measured inside the runtime container with active background model allocations:
| Message Size | Latency ($\mu$s) | Algorithm Bandwidth (GB/s) | Bus Bandwidth (GB/s) |
|---|---|---|---|
| 14,336 B (1 Token Decode) | 128.32 | 0.11 | 0.20 |
| 28,672 B (2 Tokens Decode) | 129.77 | 0.22 | 0.39 |
| 57,344 B (4 Tokens Decode) | 156.93 | 0.37 | 0.64 |
| 114,688 B (8 Tokens Decode) | 164.70 | 0.70 | 1.22 |
| 229,376 B (16 Tokens Decode) | 168.96 | 1.36 | 2.38 |
| 1,048,576 B (1 MB) | 194.09 | 5.40 | 9.45 |
| 16,777,216 B (16 MB) | 328.79 | 51.03 | 89.30 |

*Note*: The standard PyTorch Ring NCCL baseline exhibits fixed kernel launch + ring-hop overhead of ~125 $\mu$s for small messages. In contrast, vLLM's `CUSTOM` one-shot all-reduce kernel utilizes NVLink direct read/write mailboxes, reducing intra-node latency to 8–15 $\mu$s for 14 KB messages.

### 4.2 Inter-Node Latency (RDMA RoCE)
- Physical Link: 400 Gbps Ethernet (MT41692 ConnectX-7).
- Layer 3 RTT (ICMP Ping over `ens13f0np0` 172.16.1.2 $\to$ 172.16.2.2):
  - Min: 0.125 ms (125 $\mu$s)
  - Avg: 0.203 ms (203 $\mu$s)
  - Max: 0.315 ms (315 $\mu$s)
- P2P RDMA Send/Recv Latency: ~15 $\mu$s for 14 KB transfer.
- Network Throughput: Saturates 400 Gbps line rate per rail with multi-stream queue pairs.
