import pandas as pd
import numpy as np
from collections import deque

# ---------- PARAMETERS ----------
tau = 0.80

# Input files (heart & muscle)
FILES = [
    ("corr10x10_heart_leftventricle.csv", "qaoa_top_heart.csv", "qaoa_top_heart_connected.csv"),
    ("corr10x10_muscle_skeletal.csv", "qaoa_top_muscle.csv", "qaoa_top_muscle_connected.csv"),
]

def normalize_matrix(path):
    W = np.abs(pd.read_csv(path, index_col=0).values.astype(float))
    rng = W.max() - W.min()
    return (W - W.min()) / (rng if rng != 0 else 1.0)

def strong_neighbors(W, nodes, tau):
    adj = {u: set() for u in nodes}
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            u, v = nodes[i], nodes[j]
            if W[u,v] > tau or W[v,u] > tau:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def is_connected(adj, nodes):
    if not nodes: return False
    start = nodes[0]
    seen = {start}
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(nodes)

# ---------- RUN ----------
for corr_file, qaoa_file, out_file in FILES:
    W = normalize_matrix(corr_file)
    df = pd.read_csv(qaoa_file)

    # مرحله 1: فقط Connectedness
    connected_rows = []
    for _, row in df.iterrows():
        nodes = eval(row["nodes"])
        if len(nodes) < 2:
            continue
        adj = strong_neighbors(W, nodes, tau)
        if is_connected(adj, nodes):
            connected_rows.append(row)

    # مرحله 2: Deduplication (هر نود فقط یکبار)
    used_nodes = set()
    final_rows = []
    for row in connected_rows:
        nodes = eval(row["nodes"])
        if any(n in used_nodes for n in nodes):
            continue  # این subset نودهای تکراری داره
        final_rows.append(row)
        used_nodes.update(nodes)

    df_filt = pd.DataFrame(final_rows)
    df_filt.to_csv(out_file, index=False)
    print(f"{qaoa_file}: {len(df_filt)} / {len(df)} subsets kept (after dedup) -> {out_file}")
