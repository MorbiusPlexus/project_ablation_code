# Project Morbius | Strategic Pivot Implementation: Ideas 1 & 2

This folder contains the complete implementation, verification suites, and Ada Plafnet cluster deployment scripts for **Idea 1 (Single-Layer Super-Memory)** and **Idea 2 (Host-RAM Latency-Hiding via Skip-Layer Delayed Injection $\Delta$)**, as specified in the [Strategic Pivot Plan](file:///home/sparsh-bhartia/Documents/ANLP/PROJECT/test/Morbius_Pivot_Plan_Points123_Oct25.pdf).

---

## Directory Structure

```
check/
├── README.md                           # This guide
├── run_all_checks.py                   # Master test runner (executes all verification suites)
├── idea1/                              # Idea 1: Single-Layer Super-Memory (Point 1)
│   ├── verify_super1_params.py         # Verifies exact parameter matching (369.77M params, +0.0027%)
│   └── test_super1_model.py            # Forward/backward & zero-out sensitivity probe test
├── idea2/                              # Idea 2: Latency Hiding & Delayed Injection (Point 2)
│   ├── bench_offload_latency.py        # Sub-Task 2A: Standalone Host-RAM Latency Harness
│   └── test_delayed_injection.py       # Sub-Task 2B: Delayed Injection (Δ=4) model & gradient flow test
└── plafnet_scripts/                    # Execution & allocation scripts for Ada Plafnet (RTX 3090)
    ├── get_plafnet_node.sh             # Allocates an interactive or holder session on plafnet2
    ├── run_train_super1_plafnet.sh     # Slurm sbatch script for Idea 1 training (w2_mem_super1)
    ├── run_train_delayed4_plafnet.sh   # Slurm sbatch script for Idea 2 training (w2_mem_delayed4)
    └── run_bench_latency_plafnet.sh    # Script to run the Host-RAM latency harness on plafnet2
```

---

## 1. Quick Local Verification

To run the entire test suite locally (parameter matching, forward/backward passes, activation checkpointing, zero-out probe, and latency benchmark dry-run):

```bash
python3 check/run_all_checks.py
```

### Individual Checks:
- **Idea 1 Parameter Accounting**:
  ```bash
  python3 check/idea1/verify_super1_params.py
  # Or via standard ffnmem:
  python3 -m ffnmem.params configs/w1_mem_cut_near.yaml configs/w2_mem_super1.yaml
  ```
- **Idea 1 Model & Zero-Out Probe**:
  ```bash
  python3 check/idea1/test_super1_model.py
  ```
- **Idea 2 Delayed Injection**:
  ```bash
  python3 check/idea2/test_delayed_injection.py
  ```
- **Idea 2 Host-RAM Latency Harness**:
  ```bash
  python3 check/idea2/bench_offload_latency.py --tokens 512 --warmup 50
  ```

---

## 2. Obtaining a Plafnet Node on Ada

On `ada.iiit.ac.in`, your account is `sparsh.bhartia` with access to partition `plafnet2` (NVIDIA RTX 3090 GPUs).

Use [`check/plafnet_scripts/get_plafnet_node.sh`](file:///home/sparsh-bhartia/Documents/ANLP/PROJECT/test/check/plafnet_scripts/get_plafnet_node.sh):

### A. Dedicated 4× RTX 3090 Node (`gnode118`)
```bash
# Obtains an interactive bash shell on gnode118 (4x 3090, 16 CPUs, 80GB RAM):
bash check/plafnet_scripts/get_plafnet_node.sh --4gpu
```

### B. 2× RTX 3090 Node (`gnode114` / `gnode115` / `gnode116`)
```bash
# Obtains an interactive bash shell on a 2-GPU node (excluding faulty gnode113):
bash check/plafnet_scripts/get_plafnet_node.sh --2gpu
```

### C. Background Holder Job (for `pam_slurm_adopt` or `stage.sh`)
```bash
# Keeps the node allocated for 24h so you can rsync env/data or attach:
bash check/plafnet_scripts/get_plafnet_node.sh --holder
```

### D. Check Status of Your Active Jobs
```bash
bash check/plafnet_scripts/get_plafnet_node.sh --status
```

---

## 3. Running Idea 1 (Single-Layer Super-Memory) on Plafnet

**Config:** [`configs/w2_mem_super1.yaml`](file:///home/sparsh-bhartia/Documents/ANLP/PROJECT/test/configs/w2_mem_super1.yaml)
- Consolidates the full 100M parameter memory budget into **Layer 1** ($R = 184$ routes, $d_m = 128$).
- Eliminates Layer 11 memory module completely to eliminate deep-layer address drift.
- Perfectly matches non-embedding parameters ($369,777,920$, $\Delta = +0.0027\%$).

### Launch Training on Plafnet:
```bash
# 1. Launch 4-GPU training run on gnode118 (RTX 3090, micro_batch 4):
bash check/plafnet_scripts/run_train_super1_plafnet.sh

# 2. Or on a 2-GPU node (e.g. gnode114/115):
bash check/plafnet_scripts/run_train_super1_plafnet.sh --2gpu
```

---

## 4. Running Idea 2 on Plafnet

### Sub-Task 2A: Standalone Host-RAM Latency Harness
Measures single-token decode latency ($T = 512$, $B = 1$) comparing:
1. **On-GPU Baseline** (Table in GPU VRAM)
2. **Synchronous Host-RAM** ($\Delta = 0$, blocking PCIe stalls)
3. **Asynchronous Delayed Host-RAM** ($\Delta \in \{1, 2, 4\}$, pipelined DMA overlapped with $\Delta$ layers)

#### Option 1: Direct Run (when inside an allocated node shell):
```bash
bash check/plafnet_scripts/run_bench_latency_plafnet.sh --tokens 512 --warmup 50
```

#### Option 2: Autonomous Slurm Job:
```bash
bash check/plafnet_scripts/run_bench_latency_plafnet.sh --sbatch --tokens 512 --warmup 50
```

### Sub-Task 2B: Delayed Injection Pretraining Run ($\Delta = 4$)
**Config:** [`configs/w2_mem_delayed4.yaml`](file:///home/sparsh-bhartia/Documents/ANLP/PROJECT/test/configs/w2_mem_delayed4.yaml)
- Router computation occurs at Layer 1; injection is pipelined and added to residual stream at Layer $1 + 4 = \text{Layer } 5$ before attention.

```bash
# Launch training run on Plafnet:
bash check/plafnet_scripts/run_train_delayed4_plafnet.sh
```
