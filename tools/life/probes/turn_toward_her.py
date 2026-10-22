"""W9b's early ruler: in command blocks on its own legs, how often the brain's OWN turn act (the loco effector's turn digit, the
reflex's step not included) points toward her (her bearing from the body's heading), among ticks it turns and she is off-axis by
more than 0.3 rad; chance is 0.5. Per day."""
import sys, json, math, collections
sys.path.insert(0, "/Users/lukehamond/Projects/project-worktrees/wt_int")
from body.sim.world import act_digits, LOCO_SETTINGS, LOCO_JOINTS, EFFECTOR_REST, LOCO_NAME
T0 = int(sys.argv[1]) if len(sys.argv) > 1 else 7344000
agree = collections.Counter(); tot = collections.Counter(); go = collections.Counter(); n = collections.Counter()
with open("/Users/lukehamond/Projects/project/data/g1_seed1/ticks.jsonl") as f:
    f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 400_000_000)); f.readline()
    for l in f:
        try: r = json.loads(l)
        except Exception: continue
        if r["t"] <= T0: continue
        g = r.get("ghost")
        if not g or len(g) < 8 or g[7] != "own": continue
        her, ch = r.get("her_at"), r.get("child")
        la = (r.get("acts") or {}).get("loco")
        if her is None or ch is None or la is None: continue
        cx, cy = (ch[0], ch[1]) if isinstance(ch, (list, tuple)) else (ch.get("x"), ch.get("y")) if isinstance(ch, dict) else (None, None)
        if cx is None: continue
        hx, hy = her[0], her[1]
        bearing = math.atan2(hy - cy, hx - cx) - float(g[2]); bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        day = (r["t"] - 7104000) // 48000 + 148; n[day] += 1
        if int(la) == EFFECTOR_REST[LOCO_NAME]: continue
        dg = act_digits(int(la), len(LOCO_JOINTS)); turn = LOCO_SETTINGS[1][dg[1]]; spd = LOCO_SETTINGS[0][dg[0]]
        go[day] += spd > 0
        if turn == 0 or abs(bearing) < 0.3: continue
        tot[day] += 1; agree[day] += (turn > 0) == (bearing > 0)
for d in sorted(n):
    print(f"day {d}: own-legs ticks {n[d]}, its turns toward her {agree[d]}/{tot[d]} = {agree[d] / max(1, tot[d]):.2f} (chance 0.50), its 'go' acts {go[d]}")
