"""spelled words in the CHILD'S OWN token channel (not 'word', which is her channel): python3 token_words.py D [L]
A letter run in `token` (letters, then <space>/<rest>/<end>/a birth word/None ends it) is a spelling; counted if it is a word she knows
(lexicon growth words or birth words). ROOM=door; ROOT the tree with body/."""
import os, sys, json, collections
ROOT = os.environ.get("ROOT", "/Users/lukehamond/Projects/project-worktrees/wt_int"); sys.path.insert(0, ROOT)
os.environ.setdefault("ROOM", "door")
from body.sim.lang import lexicon as LX
from body.sim.lang import lexicon as _l
import body.sim.lang as langpkg
try:
    from body.sim.lang import lanes as LN
except Exception:
    LN = None
D = int(sys.argv[1]); L = sys.argv[2] if len(sys.argv) > 2 else "/Users/lukehamond/Projects/project/data/g1_seed1"
t0, t1 = D * 48000 + 1, D * 48000 + 24000
if LN is not None and hasattr(LN, "LX_TO_AN"):
    an2sym = {int(LN.LX_TO_AN[i]): s for i, s in enumerate(LX.TABLE)}
else:
    import numpy as np
    from body.sim.lang import lexicon
    an2sym = {i: s for i, s in enumerate(LX.TABLE)}
letters = set(LX.LETTERS) if hasattr(LX, "LETTERS") else set("abcdefghijklmnopqrstuvwxyz")
birth = set(LX.BIRTH_WORDS)
runs = collections.Counter(); cur = []; rests = 0; n = 0; tok_n = collections.Counter()
with open(L + "/ticks.jsonl") as f:
    for line in f:
        try: r = json.loads(line)
        except Exception: continue
        t = r.get("t", 0)
        if not (t0 <= t <= t1) or r.get("night"): continue
        n += 1
        tk = r.get("token")
        sym = an2sym.get(int(tk)) if isinstance(tk, (int, float)) and tk is not None else (tk if isinstance(tk, str) else None)
        if sym is not None: tok_n[sym] += 1
        # her rule (transcriber 4.9): letters run until a space, the end row, a word token, or 2 rest ticks
        if sym in letters: cur.append(sym); rests = 0
        elif sym is None or sym == "<rest>":
            rests += 1
            if cur and rests >= 2: runs["".join(cur)] += 1; cur = []
        else:
            if cur: runs["".join(cur)] += 1
            cur = []; rests = 0
if cur: runs["".join(cur)] += 1
print(f"day {D}: rows {n}; tokens spoken {sum(tok_n.values())}; distinct {len(tok_n)}")
print("top tokens:", tok_n.most_common(12))
print("letter runs:", sum(runs.values()), "distinct", len(runs))
known = {w for w in runs if (w in birth) or len(w) >= 3}
print("runs >=3 letters or words:", [(w, c) for w, c in runs.most_common() if w in known][:20])
