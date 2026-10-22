"""HER SIDE IN THE STATE'S RESPONSE TO HER VOICE (drift-free): at each onset of her voice (heard rising) with her off to a side, the change
in a state from the tick before the onset to K ticks after, decoded for her side by leave-one-out ridge, against a shuffled-label control.
The change removes the slow drift that confounds a decoder of the state itself. onset_decode.py NPZ"""
import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=True); X, B, H = d["X"], d["B"], d["H"].astype(int); BA = d["BANDS"]
on = np.nonzero(np.diff(H) == 1)[0] + 1
print(f"voice onsets {len(on)} in {len(H)} ticks; with her off to a side: {int(np.sum(np.abs(B[on]) > 0.3))}")
def loo(R, y, lam):
    n = len(y); hit = 0
    for i in range(n):
        m = np.ones(n, bool); m[i] = False; A = R[m]; mu = A.mean(0); sd = A.std(0) + 1e-6; A = (A - mu) / sd
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[m]); hit += np.sign(((R[i] - mu) / sd) @ w) == y[i]
    return hit / n
def run(Z, name):
    for k in (2, 5, 10):
        idx = [t for t in on if t - 1 >= 0 and t + k < len(Z) and abs(B[t]) > 0.3]
        if len(idx) < 12: print(name, k, "too few", len(idx)); return
        R = np.array([Z[t + k] - Z[t - 1] for t in idx]); y = np.sign(B[idx])
        R = R - R.mean(0); U, S_, Vt = np.linalg.svd(R, full_matrices=False); R = U[:, :20] * S_[:20]   # (20 leading directions of the change: fast)
        accs = {lam: loo(R, y, lam) for lam in (10, 100, 1000)}
        best = max(accs, key=accs.get); rng = np.random.default_rng(0)
        ctrl = [loo(R, rng.permutation(y), best) for _ in range(100)]
        print(f"{name}, +{k} ticks (n {len(y)}, left {int((y > 0).sum())}): " + " | ".join(f"lam {l}: {a:.2f}" for l, a in accs.items())
              + f" | shuffled {np.mean(ctrl):.2f} ± {np.std(ctrl):.2f} (95th {np.percentile(ctrl, 95):.2f})", flush=True)
run(X, "actor input")
if BA.size:
    for b in range(BA.shape[1]): run(BA[:, b, :], f"band {b}")
