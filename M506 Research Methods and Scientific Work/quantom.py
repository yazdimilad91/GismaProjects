!pip install -q "qiskit==1.2.4" "qiskit-aer==0.14.2" "qiskit-algorithms==0.3.0" "qiskit-optimization==0.6.1"


# ============================================================
# QAOA (QUBO) — New Qiskit, no venv, no NumPy downgrades
# Encoded in QUBO (pure quantum objective):
#  (1) Penalize isolated selected nodes
#  (2) Reward only strong edges (W > tau)
#  (3) Ignore weak edges entirely
# Runs on HEART & MUSCLE; saves CSV + PNG
# ============================================================

# --------- install only if needed (keeps Colab's numpy as-is) ----------
try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_algorithms.minimum_eigensolvers import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_aer.primitives import Sampler
except Exception:
    !pip install -q qiskit qiskit-aer qiskit-algorithms qiskit-optimization
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_algorithms.minimum_eigensolvers import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_aer.primitives import Sampler

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from qiskit_algorithms.utils import algorithm_globals
import os

# ---------- Parameters (edit here) ----------
tau     = 0.80   # فقط یال‌های با W > tau وارد مدل می‌شوند
M       = 10.0    # جریمه برای نود ایزوله (بزرگ‌تر = سخت‌گیری بیشتر)
reps    = 3      # تعداد لایه‌های QAOA
shots   = 10000   # تعداد اندازه‌گیری‌ها (بیشتر = نوسان کمتر)
maxiter = 80     # گام‌های بهینه‌ساز کلاسیک COBYLA
top_k   = 10     # چند جواب برتر ذخیره شود
seed    = 42     # برای reproducibility

HEART_FILE  = "corr10x10_heart_leftventricle.csv"
MUSCLE_FILE = "corr10x10_muscle_skeletal.csv"

# ---------- Utils ----------
algorithm_globals.random_seed = seed
np.random.seed(seed)

def normalize_matrix(df: pd.DataFrame) -> np.ndarray:
    W = np.abs(df.values)
    rng = W.max() - W.min()
    return (W - W.min()) / (rng if rng != 0 else 1.0)

def build_qubo(W: np.ndarray, tau: float, M: float) -> QuadraticProgram:
    """
    QUBO (min):
      - sum_{i<j, W_ij>tau} W_ij * x_i x_j     (reward strong edges)
      + eps * sum_i x_i                         (tiny tie-break)
      + isolated penalty: if node i has no strong neighbors at all,
                          then selecting it (x_i=1) costs +M
    """
    n = W.shape[0]
    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(name=f"x{i}")
    linear, quad = {}, {}

    # (1) reward strong edges
    for i in range(n):
        for j in range(i+1, n):
            if W[i,j] > tau:
                quad[(f"x{i}", f"x{j}")] = quad.get((f"x{i}", f"x{j}"), 0.0) - float(W[i,j])

    # (2) tiny tie-break
    eps = 1e-3
    for i in range(n):
        linear[f"x{i}"] = linear.get(f"x{i}", 0.0) + eps

    # (3) isolated node penalty
    for i in range(n):
        neighbors = [j for j in range(n) if j != i and W[i,j] > tau]
        if not neighbors:  # node i has zero strong connections
            linear[f"x{i}"] = linear.get(f"x{i}", 0.0) + M

    qp.minimize(linear=linear, quadratic=quad)
    return qp


def decode_and_score(bitstring: str, W: np.ndarray, tau: float):
    """
    قوانین پست‌فیلتر (منطبق با خواسته):
      - اگر حتی یک نود در subset هیچ یالِ >tau به سایر نودهای subset ندارد → حذف (صفر)
      - امتیاز = مجموع وزن همه‌ی یال‌های قوی داخل subset (نه میانگین؛ چون شرط 4 را حذف کردیم)
    """
    nodes = [i for i, b in enumerate(bitstring[::-1]) if b == "1"]
    if len(nodes) < 2:
        return None
    # عدم ایزوله داخل subset
    for i in nodes:
        if not any((j != i and (j in nodes) and W[i,j] > tau) for j in nodes):
            return None
    # امتیاز = مجموع یال‌های قوی داخل subset
    score = 0.0
    for a in range(len(nodes)):
        for b in range(a+1, len(nodes)):
            i, j = nodes[a], nodes[b]
            if W[i,j] > tau:
                score += float(W[i,j])
    if score <= 0.0:
        return None
    return nodes, score

def run_qaoa_on_matrix(W: np.ndarray, title: str):
    qp   = build_qubo(W, tau, M)
    qubo = QuadraticProgramToQubo().convert(qp)
    op   = qubo.to_ising()[0]

    sampler = Sampler(run_options={"shots": shots})
    qaoa    = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=maxiter), reps=reps)

    res  = qaoa.compute_minimum_eigenvalue(op)
    dist = res.eigenstate.binary_probabilities()

    pairs = sorted(dist.items(), key=lambda x: -x[1])
    rows  = []
    for bitstring, prob in pairs[:max(200, top_k*10)]:
        parsed = decode_and_score(bitstring, W, tau)
        if parsed is None:
            continue
        nodes, score = parsed
        rows.append({
            "nodes": nodes,
            "num_nodes": len(nodes),
            "score_sum_strong": round(score, 4),
            "prob": round(prob, 4),
            "bitstring": bitstring
        })

    rows.sort(key=lambda r: (-r["score_sum_strong"], -r["prob"]))
    top = rows[:top_k]
    df  = pd.DataFrame(top)

    # ذخیره خروجی
    csv_name = f"qaoa_top_{title.lower()}.csv"
    png_name = f"qaoa_top_{title.lower()}.png"
    df.to_csv(csv_name, index=False)

    plt.figure(figsize=(10,4))
    plt.bar(range(len(top)), [r["score_sum_strong"] for r in top])
    plt.xticks(range(len(top)), [",".join(map(str, r["nodes"])) for r in top], rotation=90)
    plt.ylabel("Sum of strong-edge weights (W>tau)")
    plt.title(f"{title} — tau={tau} | reps={reps} | shots={shots}")
    plt.tight_layout()
    plt.savefig(png_name)
    plt.close()

    print(f"Saved: {csv_name}, {png_name}")
    return df

# ---------- IO & run ----------
for p in [HEART_FILE, MUSCLE_FILE]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing file: {p}")

W_heart  = normalize_matrix(pd.read_csv(HEART_FILE, index_col=0))
W_muscle = normalize_matrix(pd.read_csv(MUSCLE_FILE, index_col=0))

print("Running QAOA on HEART ...")
df_heart  = run_qaoa_on_matrix(W_heart,  "HEART")
print("\nRunning QAOA on MUSCLE ...")
df_muscle = run_qaoa_on_matrix(W_muscle, "MUSCLE")

print("\nTop HEART:")
display(df_heart)
print("\nTop MUSCLE:")
display(df_muscle)
