# Parallel Qubit Ablation on Mac Mini M4

## The problem

`lightning.qubit` runs single-threaded at low qubit counts. At 12 qubits the state
vector is only 4,096 complex amplitudes (~64 KB) — too small to saturate even one core,
let alone spread across all 10. The M4's performance cores sit idle while the training
loop executes serially.

Trying to fix this by increasing qubits is counter-productive: each extra qubit doubles
the state vector and the compute, so the work grows exponentially while the parallelism
gain is marginal.

---

## The solution: process-level data parallelism

The ablation study (qubit sweep 4 / 6 / 8 / 10 / 12) runs five fully independent
experiments. Each can be pinned to its own core. Total wall-clock time becomes the
duration of the single longest run rather than the sum of all five.

```
Core 0  →  n_qubits=4   (shortest)
Core 1  →  n_qubits=6
Core 2  →  n_qubits=8
Core 3  →  n_qubits=10
Core 4  →  n_qubits=12  (longest)
────────────────────────────────────
Wall time ≈ runtime(n_qubits=12) only
```

---

## How to run it

### Option A — simple bash (recommended)

```bash
source .venv/bin/activate
mkdir -p results/ablation

for Q in 4 6 8 10 12; do
  python run_benchmark.py \
    --n-qubits $Q \
    --vqc-epochs 100 \
    --cv-folds 0 \
    --no-plots \
    > results/ablation/run_q${Q}.log 2>&1 &
done

wait
echo "All done."
```

Each process writes its own log. `wait` blocks until the last one finishes.

### Option B — with CPU affinity (keeps cores from competing)

```bash
source .venv/bin/activate
mkdir -p results/ablation

QUBITS=(4 6 8 10 12)
for i in "${!QUBITS[@]}"; do
  Q=${QUBITS[$i]}
  taskset -c $i python run_benchmark.py \
    --n-qubits $Q \
    --vqc-epochs 100 \
    --cv-folds 0 \
    --no-plots \
    > results/ablation/run_q${Q}.log 2>&1 &
done

wait
echo "All done."
```

`taskset -c $i` pins each process to a specific core so the OS scheduler doesn't
shuffle them around under load. Requires `util-linux` (`brew install util-linux` on
macOS — or just use Option A, the difference is small).

### Monitor progress

```bash
# Live tail all logs at once
tail -f results/ablation/run_q*.log

# Or check which are still running
ps aux | grep run_benchmark
```

---

## Bonus: adjoint differentiation

Switching from parameter-shift to adjoint differentiation cuts gradient evaluations
from `2 × n_params` circuit calls to a single forward+backward pass. For
`StronglyEntanglingLayers(2, n_qubits)` this means:

| n_qubits | params | parameter-shift evals/step | adjoint evals/step |
|---|---|---|---|
| 4  | 24  | 48  | 1 |
| 8  | 48  | 96  | 1 |
| 12 | 72  | 144 | 1 |

`lightning.qubit` supports adjoint natively. To enable it, change the `@qml.qnode`
decorator in `src/quantum_models.py`:

```python
# before
@qml.qnode(dev, diff_method="parameter-shift")

# after
@qml.qnode(dev, diff_method="adjoint")
```

**Caveat:** adjoint only works with `lightning.qubit` (not `default.mixed`), so it
cannot be used for the noise sweep. For the noiseless ablation it is safe and
recommended.

Combined with the parallel sweep, this is the fastest path to full ablation results
on the M4.

---

## Expected outcome

| Approach | Estimated wall time (12-qubit run) |
|---|---|
| Serial, parameter-shift (current) | baseline |
| Serial, adjoint | ~50–70% of baseline |
| Parallel (5 cores), adjoint | ~50–70% of baseline ÷ 1 (runs concurrently) |

The parallel sweep does not speed up any individual run — it eliminates the wait for
the other four.

---

## Results location

Each run saves its output to the path configured in `run_benchmark.py`. Consider
passing `--output-dir results/ablation/q${Q}` if that flag exists, or redirect
stdout/stderr as shown above and collect the JSON files afterwards.
