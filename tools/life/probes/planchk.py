import argparse, sys, importlib.util, torch, numpy as np, os
ROOT=os.environ.get("ROOT","/Users/lukehamond/Projects/project-worktrees/wt_c268"); sys.path.insert(0, ROOT); os.chdir(ROOT)
spec = importlib.util.spec_from_file_location("sl", ROOT+"/tools/life/sim_life.py"); sl = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(sl)
except SystemExit: pass
torch.set_num_threads(2)
S="/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/pair"
for room in ("a","door"):
    ns = argparse.Namespace(out=S, resume=True, d=512, seed=1, voice="fake", extra="book_box_bucket", room=room, body="a", no_eyes=False, set=[], lr0=False, page=False, film=None, film_every=3, save_every=0, days=0, ticks=0, threads=2)
    world, eyes, lane, L = sl.build(ns); pm = world.parent; P = pm.plan
    her = np.asarray(pm.base["at"], float)[:2]; ch = np.asarray(world.d.qpos[:2], float)
    print(room, "grid", P.nx, P.ny, "X0 Y0", P.X0, P.Y0, "free cells", int((P.dist >= 0.35).sum()), "her", her.round(2), "dist", round(float(P.dist[P.cell(her)]),2), "in", pm._in_plan(her), "child", ch.round(2), "dist", round(float(P.dist[P.cell(ch)]),2))
    xs = np.linspace(-2.4, 2.4, 13); print("  dist along y=-1.1:", [round(float(P.dist[P.cell((x, -1.1))]), 2) for x in xs])
    print("  dist along x=0.9:", [round(float(P.dist[P.cell((0.9, y))]), 2) for y in np.linspace(-2.1, 2.1, 11)])
    if room == "door":
        m = world.m
        for nm in ("hall_end","hall_side_a","hall_side_b","door_leaf","room2_bench"):
            g = m.geom(nm).id; print("  ", nm, m.geom_pos[g].round(2), m.geom_size[g].round(2), "xpos", world.d.geom_xpos[g].round(2))
        bad = [(m.geom(gg).name, world.d.geom_xpos[gg].round(2), m.geom_size[gg].round(2)) for gg in range(m.ngeom) if int(m.geom_bodyid[gg]) == m.body("room2").id]
        print("  room2 geoms", bad)
