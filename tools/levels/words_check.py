"""THE WORDS' CHECK (its seeing's looks and the eye layer's words): a saved state's looks (against Workshop 18's) and the words' strongest associations (a reading only)

    python3 tools/levels/words_check.py LEGS_STATE.pt [word ...]
"""
import sys
import torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))))
from body.sim.lang import lexicon as LX  # noqa: E402
W18 = {"ball": 1.8, "block": 225.7, "duck": 46.9, "cup": 131.8, "car": 25.6, "bear": 25.7, "drum": 157.4, "can": 7.3, "bucket": 73.7}
st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
pr = st["seen"]["proto"]
print("looks:", " ".join(f"{n} {pr[n]['hue']:.0f}({abs((pr[n]['hue'] - r + 180) % 360 - 180):.0f}) w{pr[n]['width']:.2f} n{pr[n]['n']}" for n, r in W18.items() if n in pr))
ids = st["word_forms"]["ids"]; a = st["eyes"]["assoc"]
def wid(w):
    return ids[w] if w in ids else (29 + LX.WORD_ID[w] if w in LX.WORD_ID else None)
words = sys.argv[2:] or ["bucket", "ball", "cup", "block", "duck", "car", "can", "bear", "drum", "put", "pick", "go", "it", "in", "the", "up", "down", "look", "at"]
for w in words:
    d = a.get(wid(w)) or {}
    print(f"  {w:7s}", {k: round(v, 2) for k, v in sorted(d.items(), key=lambda x: -x[1])[:4]})
