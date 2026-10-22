"""THE BUG STALKER (2026-10-16, the owner's word: 'stalk the process for bugs and fix them right when they come up'). Every INTERVAL s it
reads the record's rows and run.log's lines since its last look and checks invariants that only a bug breaks; on any breach it prints
the ALERT lines and exits (so the harness wakes the lead). Run in the background; re-arm after each wake. Reads only.
Invariants: numbers finite (dopamine, its gain, stress, mood, the kappas, the gates, the actors' updates, the tick's time); no gap in a
day's rows; no new Traceback / PLAN ERROR / jump guard / WorldFault / NaN line in run.log; her come-here not held by the expectant pause
after C357 lands (dawn 183); an intent refused 25+ times for one reason in a window (a loop); a walk block whose teacher stayed at
full strength with no tick alone (W9g's regression); a command block whose legs followed the turn under 0.2 at full teacher; the actors'
updates zero through a day window (frozen actors); the imagination and the frames' ends not moving through a day window (A180's
regression); pain over 6% or falls over 15 a thousand ticks in a window (three times the day's); the tick over 600 ms through a window."""
import json, os, time, re, math, sys, collections, ast
L = "/Users/lukehamond/Projects/project/data/g1_seed1"; INTERVAL = int(os.environ.get("INTERVAL", 180))
C357_FROM = 8784000
def finite(v):
    if isinstance(v, (int, float)): return math.isfinite(v)
    if isinstance(v, (list, tuple)): return all(finite(x) for x in v)
    if isinstance(v, dict): return all(finite(x) for x in v.values())
    return True
def rows_since(t0):
    out = []
    with open(f"{L}/ticks.jsonl", "rb") as f:
        f.seek(max(0, os.path.getsize(f"{L}/ticks.jsonl") - 60_000_000)); f.readline()
        for l in f:
            if b'"t": ' not in l: continue
            try: r = json.loads(l)
            except Exception: continue
            if r.get("t", 0) > t0: out.append(r)
    return out
def log_lines():
    with open(f"{L}/run.log", "r", errors="replace") as f: return f.read().splitlines()
last_t = int(os.environ["START_T"]) if os.environ.get("START_T") else max([r["t"] for r in rows_since(0)[-50:]] or [0]); n_log = len(log_lines()) - int(os.environ.get("BACK_LOG", 0))
print(f"bug stalker armed at {time.strftime('%H:%M')}: tick {last_t}, run.log line {n_log}", flush=True)
while True:
    time.sleep(INTERVAL)
    rows = rows_since(last_t); lines = log_lines(); new = lines[n_log:]; n_log = len(lines)
    al = []
    for l in new:
        if re.search(r"Traceback|PLAN ERROR|SWALLOWED|jump guard|WorldFault|\bnan\b|\bNaN\b|\binf\b", l.replace("balanced nan", "")): al.append(f"run.log: {l[:220]}")   # (C361's 'balanced nan', a side with no turns: not a number gone bad)
        m = re.search(r"walk block over at tick (\d+): (\[[^\]]*\]) \[strength", l)
        if m:                                                              # (C361 adds a 9th field, the block's median error: parse the list)
            try: rw_ = ast.literal_eval(m.group(2))
            except Exception: rw_ = []
            if len(rw_) >= 8 and float(rw_[0]) >= 0.999 and int(rw_[7]) == 0: al.append(f"walk block at {m.group(1)}: the teacher at full strength the whole block, no tick alone (W9g?)")
            if len(rw_) >= 9 and rw_[8] is not None and float(rw_[8]) > 0.05: al.append(f"walk block at {m.group(1)}: the layer's median error over the block {rw_[8]} (> 0.05: the layer off its teacher, not one fall's spike)")
        m = re.search(r"the legs' teacher ([0-9.]+), the turn followed ([0-9.]+)", l)
        if m and float(m.group(1)) >= 0.999 and float(m.group(2)) < 0.2: al.append(f"command block: the legs' teacher {m.group(1)}, the turn followed {m.group(2)} (the brain's turn not executed)")
    if rows:
        fz_ = []                                                             # C362's regression: the walk layer's error frozen (no lesson: off its
        for r in rows:                                                       # feet) 60+ ticks running with no fall counted (the sofa, day 185)
            g_ = r.get("ghost")
            if isinstance(g_, list) and len(g_) > 8 and g_[7] == "own": fz_.append((r["t"], g_[8], g_[3]))
            elif isinstance(g_, list) and len(g_) > 6 and g_[5] == "walk": fz_.append((r["t"], g_[6], g_[3]))
        run_ = 0
        for (ta, ea, fa), (tb, eb, fb) in zip(fz_, fz_[1:]):
            run_ = run_ + 1 if (tb - ta == 1 and ea == eb and fa == fb) else 0
            if run_ == 60: al.append(f"tick {tb}: the walk layer's error frozen at {eb} for 60 ticks with no fall counted (off its feet, not caught: C362?)")
        day = [r for r in rows if r.get("ep") not in (None, "night")]
        for r in rows:
            for k in ("dop", "dop_shape", "dopg", "stress", "mood", "kj", "kj_loco", "kappa", "gates", "actor_upd", "ms", "reward", "imag_N"):
                if k in r and not finite(r[k]): al.append(f"tick {r['t']}: {k} not finite: {str(r[k])[:80]}"); break
        ts = [r["t"] for r in day]
        gaps = [(a, b) for a, b in zip(ts, ts[1:]) if b - a > 60 and b % 48000 != 0]   # (a gap ending at a dawn is the night: no rows by design)
        if gaps: al.append(f"a gap in the day's rows: {gaps[:3]}")
        ref = collections.Counter(); done = collections.Counter()
        for r in rows:
            x = r.get("refused")
            if isinstance(x, (list, tuple)) and len(x) >= 3:                 # [tick, intent, reason], one a row
                it, why = str(x[1]), str(x[2])
                if not why.startswith("stay put"):                           # (C351's by design)
                    ref[(it, why[:60])] += 1
                if r["t"] >= C357_FROM and it in ("come_call", "come_again") and "expectant pause" in why: al.append(f"tick {r['t']}: {it} held by the expectant pause after C357")
            for a in (r.get("acts_ended") or []):
                if isinstance(a, (list, tuple)) and len(a) > 2 and a[2] == "done": done[str(a[0])] += 1
        for (it, why), n in ref.items():
            if n >= 25: al.append(f"'{it}' refused {n} times for one reason in the window ('{why}'): a refusal loop?")
        if len(day) > 600:
            au = [r.get("actor_upd") for r in day if r.get("actor_upd")]
            if au and all(abs(sum(sum(x) for x in a)) < 1e-12 for a in au) and len(au) > 600: al.append("the actors' updates zero through the window")
            im = [r.get("imag") for r in day if r.get("imag") is not None]; en = [r.get("ends") for r in day if r.get("ends") is not None]
            if len(im) > 600 and im[-1] == im[0] and ts[-1] - ts[0] > 1500: al.append(f"the imagination's count still at {im[0]} through {ts[-1] - ts[0]} day ticks")
            if len(en) > 600 and en[-1] == en[0] and ts[-1] - ts[0] > 1500: al.append(f"the frames' ends still at {en[0]} through {ts[-1] - ts[0]} day ticks")
            pain = sum(1 for r in day if r.get("pain")) / len(day); falls = sum(1 for r in day for e in (r.get("events") or []) if e == "fell" or (isinstance(e, (list, tuple)) and e and e[0] == "fell"))
            if pain > 0.06: al.append(f"pain {100 * pain:.1f}% of {len(day)} day ticks")
            if falls / len(day) * 1000 > 15: al.append(f"falls {falls} in {len(day)} day ticks ({falls / len(day) * 1000:.1f} a thousand; the day runs about 5)")
            ms = sorted(r.get("ms", 0) for r in day); 
            if ms and ms[len(ms) // 2] > 600: al.append(f"the tick's median {ms[len(ms) // 2]:.0f} ms")
        last_t = rows[-1]["t"]
    import shutil as _sh
    free_ = _sh.disk_usage(L).free / 1e9
    if free_ < 4.0: al.append(f"the disk {free_:.1f} GB free: a save needs 1.2 GB of temporary space (2026-10-16 19:05: the disk full, a save failed, the life down)")
    if al:
        print(f"BUG STALK ALERT at {time.strftime('%H:%M')} (ticks to {last_t}):", flush=True)
        for a in dict.fromkeys(al): print("  " + a, flush=True)
        sys.exit(0)
