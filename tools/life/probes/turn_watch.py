"""C373's ruler: the turn share by call (ghost.turn_log: [tick, turned, share after]) and the forward share by call (go_log) from the
latest save, with the calls since a tick (argv[1], default dawn 192)"""
import pickle, sys
sys.path.insert(0, "/Users/lukehamond/Projects/project-worktrees/wt_int")
from body.sim import world as W
T0 = int(sys.argv[1]) if len(sys.argv) > 1 else 9216000
D = "/Users/lukehamond/Projects/project/data/g1_seed1"
s = pickle.loads(bytes(pickle.load(open(f"{D}/world.pt", "rb"))))
s5 = W._uncanon(s["s5"])
gh = s5.get("ghost") if isinstance(s5, dict) else None
if gh is None:
    print("s5 keys:", list(s5.keys())[:30]); sys.exit(0)
tl = [x for x in gh.get("turn_log", []) if int(x[0]) >= T0]; gl = [x for x in gh.get("go_log", []) if int(x[0]) >= T0]
print(f"turn share now {gh.get('reflex_share')} (go share {gh.get('go_share')}); calls owing a turn since {T0}: {len(tl)}, turned {sum(int(x[1]) for x in tl)}")
print("turn_log [tick, turned, share]:", tl[-20:])
print("go_log [tick, came, share]:", gl[-20:])
