"""onset_decode.py with LEAVE-ONE-SPOT-OUT: onsets grouped by contiguous runs of her side (a spot); each run held out whole, the decoder
trained on the others (no neighbour of the held-out onsets in training). Control: the runs' side labels permuted (the runs kept whole)"""
import sys, os, numpy as np
DIMS = int(os.environ.get("DIMS", 20))
d = np.load(sys.argv[1], allow_pickle=True); X, B, H = d["X"], d["B"], d["H"].astype(int); BA = d["BANDS"]
on = np.nonzero(np.diff(H) == 1)[0] + 1
def groups(idx):
    g = []; cur = 0
    for j, t in enumerate(idx):
        if j and (np.sign(B[t]) != np.sign(B[idx[j - 1]]) or t - idx[j - 1] > 200): cur += 1
        g.append(cur)
    return np.array(g)
def lgo(R, y, g, lam=100.0):
    hit = 0; n = len(y)
    for k in np.unique(g):
        m = g != k; A = R[m]; mu = A.mean(0); sd = A.std(0) + 1e-6; A = (A - mu) / sd
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[m]); hit += int(np.sum(np.sign(((R[~m] - mu) / sd) @ w) == y[~m]))
    return hit / n
def bal(R, y, g, lam=100.0):   # balanced accuracy over the held-out groups
    pr = np.zeros(len(y))
    for k in np.unique(g):
        m = g != k; A = R[m]; mu = A.mean(0); sd = A.std(0) + 1e-6; A = (A - mu) / sd
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[m]); pr[~m] = np.sign(((R[~m] - mu) / sd) @ w)
    return 0.5 * (np.mean(pr[y > 0] > 0) + np.mean(pr[y < 0] < 0))
def run(Z, name, k):
    idx = np.array([t for t in on if t - 1 >= 0 and t + k < len(Z) and abs(B[t]) > 0.3])
    R = np.array([Z[t + k] - Z[t - 1] for t in idx]); y = np.sign(B[idx]); g = groups(idx)
    R = R - R.mean(0); U, S_, Vt = np.linalg.svd(R, full_matrices=False); R = U[:, :DIMS] * S_[:DIMS]
    a = bal(R, y, g); rng = np.random.default_rng(0); ks = np.unique(g)
    gl = {kk: y[g == kk][0] for kk in ks}; ctrl = []
    for _ in range(60):
        perm = dict(zip(ks, rng.permutation([gl[kk] for kk in ks]))); ctrl.append(bal(R, np.array([perm[kk] for kk in g]), g))
    print(f"{name} +{k}: balanced {a:.2f} | spots permuted {np.mean(ctrl):.2f} ± {np.std(ctrl):.2f} (95th {np.percentile(ctrl, 95):.2f}) | spots {len(ks)}, onsets {len(y)}", flush=True)
for k in (5, 10):
    run(X, "actor input", k)
    for b in (0, 1, 2): run(BA[:, b, :], f"band {b}", k)
