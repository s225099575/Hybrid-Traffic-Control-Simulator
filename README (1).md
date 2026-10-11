# Traffic Control Simulator — Hybrid MPI + OpenMP (Dynamic Scheduling)

SIT315 Concurrent and Distributed Programming — Project (High Distinction task)

A distributed traffic-analytics engine that combines **MPI** (distributed-memory
parallelism across machines) with **OpenMP** (shared-memory parallelism within
each process), and supports two selectable work-distribution strategies:

- **Static mode** — the input file is split into one fixed, even range per MPI
  rank up front, mirroring the prior pure-MPI assignment.
- **Dynamic mode** — rank 0 acts as a dedicated dispatcher that hands out small
  chunks of work to the other ranks on demand (a master-worker / self-scheduling
  pattern), so a faster rank naturally completes more chunks than a slower one.

Within both modes, every rank parallelizes its own share of the data across
OpenMP threads using a manual thread-local-map-then-merge reduction pattern.

📹 **Video demonstration:** [Watch Now](https://youtu.be/AgGFw4jJzRI?si=NNzKw6as11YiZzZ0)

📄 **Full design, correctness evidence, and performance evaluation:** see
`Hybrid Traffic Control Simulator.pdf` in this repository.

---

## Project Structure

```
Hybrid-Traffic-Control-Simulator/
├── traffic_hybrid.cpp               # main hybrid MPI + OpenMP source
├── gen_data_user.py                 # dataset generator (small/medium/large/stress)
├── Hybrid Traffic Control Simulator.pdf   # design, correctness, and evaluation report
└── README.md                        # this file
```

A `hostfile` (listing your cluster's node IPs) and the generated dataset
files (`small.txt`, `medium.txt`, `large.txt`, `stress.txt`) are not
committed — create the hostfile yourself as shown in Section 3, and generate
the datasets locally with `gen_data_user.py`, since `stress.txt` alone is
15,000,000 records.

---

## Requirements

- A C++17 compiler
- An MPI implementation providing `mpic++` / `mpirun` (this project was
  developed and tested against **MPICH / Hydra 3.3a2**)
- OpenMP support (bundled with GCC/Clang by default via `-fopenmp`)
- Python 3 (only needed to regenerate the datasets)

This program is designed to run across multiple Linux machines (or VMs) as a
genuine MPI cluster, but it also runs correctly as a single-machine,
single-node job for quick local testing — just omit the hostfile and use a
smaller `-n`.

---

## 1. Setup — Linux / macOS

Install MPICH and build tools. On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y mpich build-essential python3
```

On macOS (via Homebrew):

```bash
brew install mpich
```

Clone or copy this project folder, then compile:

```bash
mpic++ -O3 -std=c++17 -fopenmp -o traffic_hybrid traffic_hybrid.cpp
```

Generate the datasets:

```bash
python3 gen_data_user.py
# writes small.txt, medium.txt, large.txt, stress.txt
```

Run a quick single-machine test (no hostfile needed):

```bash
OMP_NUM_THREADS=2 mpirun -n 4 ./traffic_hybrid small.txt 3 static
OMP_NUM_THREADS=2 mpirun -n 4 ./traffic_hybrid small.txt 3 dynamic 1000
```

---

## 2. Setup — Windows

MPICH's Hydra process manager and this project's POSIX-oriented file handling
are Linux-native, so on Windows the recommended path is **WSL2 (Windows
Subsystem for Linux)** running an Ubuntu distribution — this gives you an
identical environment to the one this project was built and tested on.

1. Install WSL2 with Ubuntu (if not already installed), from an elevated
   PowerShell prompt:
   ```powershell
   wsl --install -d Ubuntu
   ```
2. Open the Ubuntu shell and follow the **Linux / macOS** instructions above
   exactly — `apt install mpich build-essential python3`, then compile and
   run as shown.
3. If you need a genuine multi-machine cluster rather than a single WSL
   instance, point the hostfile at the IP addresses of your other nodes (see
   Section 3) — WSL2 networking supports this, but native Linux or Linux VMs
   are simpler for a true multi-node cluster.

Native Windows MPI implementations (MS-MPI) are **not** compatible with this
project's `mpic++` build command or Hydra-specific hostfile syntax without
modification, so WSL2 is the supported route on Windows.

---

## 3. Running as a Real Multi-Node Cluster

This is how the project was actually evaluated: two separate Ubuntu machines
(a "Head" node and a "Node" node), each with the compiled binary and every
dataset file at an identical path, communicating over a private network.

**a. Build once, on the Head machine:**
```bash
mpic++ -O3 -std=c++17 -fopenmp -o traffic_hybrid traffic_hybrid.cpp
python3 gen_data_user.py
```

**b. Create a `hostfile`** listing each node's IP address and how many MPI
slots it should run, one per line:
```
<head-ip>:1
<node-ip>:1
```

**c. Copy the binary, datasets, and hostfile to every other node** (adjust
user, IP, and path to match your own cluster):
```bash
scp traffic_hybrid small.txt medium.txt large.txt stress.txt hostfile \
    <user>@<node-ip>:~/path/to/hybridtrafficsim/
```

**d. Run from the Head machine**, pointing `mpirun` at the hostfile:
```bash
OMP_NUM_THREADS=2 mpirun -f hostfile -n 4 ./traffic_hybrid large.txt 5 static
```

Ranks are placed round-robin across the hosts listed in the hostfile. Make
sure passwordless SSH is set up between nodes (`ssh-copy-id`) before running
`mpirun` across machines — Hydra launches remote processes over SSH.

---

## Usage

```
OMP_NUM_THREADS=<threads> mpirun [-f hostfile] -n <processes> ./traffic_hybrid <input_file> <top_N> <static|dynamic> [chunk_size]
```

| Argument | Description |
|---|---|
| `OMP_NUM_THREADS` | OpenMP threads spawned *per MPI rank*. Always set this explicitly — the default varies by system. |
| `-n <processes>` | Total MPI processes (ranks). In dynamic mode, rank 0 is a dispatcher and does not process data, so there must be at least 2 for a true multi-worker run. |
| `<input_file>` | Path to a traffic-record CSV file (see `gen_data_user.py`). |
| `<top_N>` | How many most-congested traffic lights to report per hour block. |
| `static \| dynamic` | Scheduling mode (see overview above). |
| `[chunk_size]` | Dynamic mode only — lines per dispatched chunk. Defaults to 20000. |

**Examples:**
```bash
# Static mode, 4 ranks, 2 OpenMP threads each
OMP_NUM_THREADS=2 mpirun -f hostfile -n 4 ./traffic_hybrid large.txt 5 static

# Dynamic mode, 4 ranks (1 dispatcher + 3 workers), fine 5000-line chunks
OMP_NUM_THREADS=2 mpirun -f hostfile -n 4 ./traffic_hybrid large.txt 5 dynamic 5000
```

See `Hybrid Traffic Control Simulator.pdf` for the exact full set of commands
used in the project's evaluation (correctness check, workload-size scaling,
the static-vs-dynamic head-to-head, chunk-size sensitivity, OpenMP
thread-count tuning, and the stress test), along with the complete results
and discussion.

---

## Notes

- Every run prints a diagnostic table: a **partition balance** table in
  static mode, or a **chunks completed per worker rank** table in dynamic
  mode — this is the direct evidence of how work was actually distributed.
- Dynamic mode automatically falls back to a single-rank sequential-chunk
  mode if run with `-n 1`, since there would otherwise be no separate worker
  to dispatch to.
- See the accompanying report for a full discussion of this design's
  limitations, including the line-based (rather than byte-offset) file
  seeking that dominates dynamic mode's overhead at fine chunk sizes.
