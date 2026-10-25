"""IS THE SHOWN ACT IN THE BRAIN WHEN IT MUST ACT? (P8, 2026-10-22 19:15; the owner's "stalk the model, find the perfect conditions"). Reads
OUT/states/*.pt (P8_LOG_STATE: the cortex's last state C and the PFC ladder's eight bands at each box-family hand-over, 'hand', and 10
ticks on, 'plus10', with the act shown) and decodes the act from each feature set with a cross-validated logistic regression (standardised,
L2), against chance (the commonest act's share). A feature set far above chance holds the clip's act; near chance, the memory lost it.
Usage: decode_states_p8.py [OUT (data/demo/p8)] [min per act (8)]"""
import glob, os, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
OUT = sys.argv[1] if len(sys.argv) > 1 else "/Users/lukehamond/Projects/project/data/demo/p8"
MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rows = []
for p in sorted(glob.glob(os.path.join(OUT, "states", "*.pt"))):
    try:
        rows.append(torch.load(p, map_location="cpu", weights_only=False))
    except Exception:
        pass
for at in ("clip", "hand", "plus10"):
    R = [r for r in rows if r.get("at") == at and r.get("C") is not None]
    fams = sorted(set(r["fam"] for r in R))
    keep = [f for f in fams if sum(1 for r in R if r["fam"] == f) >= MIN]
    R = [r for r in R if r["fam"] in keep]
    if len(keep) < 2:
        print(f"{at}: {len(R)} states, acts with >= {MIN}: {keep} (too few to decode)"); continue
    y = np.array([keep.index(r["fam"]) for r in R])
    chance = max(np.bincount(y)) / len(y)
    k = min(5, int(min(np.bincount(y))))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    def score(X):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        return float(np.mean(cross_val_score(clf, X, y, cv=cv)))
    sets = {"cortex C": np.stack([r["C"].numpy() for r in R])}
    for b in range(R[0]["bands"].shape[0]):
        sets[f"band {b}"] = np.stack([r["bands"][b].numpy() for r in R])
    sets["all bands"] = np.stack([r["bands"].reshape(-1).numpy() for r in R])
    print(f"{at}: {len(R)} states, acts {dict((f, int((y == i).sum())) for i, f in enumerate(keep))}, chance {chance:.2f}, {k}-fold")
    for name, X in sets.items():
        print(f"   {name:10s} decoded {score(X):.2f}")
