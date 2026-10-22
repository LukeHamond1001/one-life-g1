import argparse, sys, importlib.util, torch, numpy as np, os
ROOT="/Users/lukehamond/Projects/project-worktrees/wt_c268"; sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
torch.set_num_threads(2)
ns = argparse.Namespace(out="/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair", resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room="door", body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
world, eyes, lane, L = sl.build(ns); pm = world.parent; P = pm.plan; m = world.m
print("     x:", " ".join(f"{x:5.1f}" for x in np.arange(2.4, 6.0, 0.3)))
for y in np.arange(0.2, -3.4, -0.3):
    print(f"y {y:5.1f}:", " ".join(f"{float(P.dist[P.cell((x, y))]):5.2f}" if pm._in_plan((x, y)) else "   --" for x in np.arange(2.4, 6.0, 0.3)))
room2 = m.body("room2").id; room = m.body("room").id
for gg in range(m.ngeom):
    if int(m.geom_bodyid[gg]) in (room, room2):
        p = world.d.geom_xpos[gg]
        if p[0] > 2.5: print(m.geom(gg).name, p.round(2), m.geom_size[gg].round(2), "type", int(m.geom_type[gg]))
print("toys", {k: np.round(v, 2).tolist() for k, v in pm._toys_xy().items()} if isinstance(pm._toys_xy(), dict) else str(pm._toys_xy())[:300])
