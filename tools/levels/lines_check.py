"""THE REQUEST LINES' CHECK (the eye layer's words): each request line's prediction of every thing (the eye layer's pull before priority: the words' associations summed, traces ~1),
from a saved state (a reading only)

    python3 tools/levels/lines_check.py LEGS_STATE.pt
"""
import sys
import torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))))
from body.sim.lang import lexicon as LX  # noqa: E402
st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ids = st["word_forms"]["ids"]; a = st["eyes"]["assoc"]
def wid(w):
    return ids[w] if w in ids else (29 + LX.WORD_ID[w] if w in LX.WORD_ID else None)
things = ["ball", "block", "duck", "cup", "car", "bear", "drum", "can", "bucket"]
lines = [f"go pick up the {t}" for t in ("cup", "block", "duck", "car", "can", "ball", "bear", "drum")] + ["put it in the bucket", "put it down", "look at the bucket", "bucket"]
for ln in lines:
    ws = ln.split(); miss = [w for w in ws if wid(w) is None]
    p = {t: sum(float((a.get(wid(w)) or {}).get(t, 0.0)) for w in ws if wid(w) is not None) for t in things}
    top = sorted(p.items(), key=lambda x: -x[1])[:3]
    named = ws[-1] if ws[-1] in things else None
    ok = "" if named is None else ("OK" if top[0][0] == named else "WRONG")
    print(f"{ln:24s} {ok:5s}", [(t, round(v, 2)) for t, v in top], ("unknown words " + str(miss)) if miss else "")
