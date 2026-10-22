# THE TEACHER WATCH (2026-10-08, the owner's word: always watch sim reality). Every INTERVAL s: the record's rows since the last look
# (data/g1_seed1/ticks.jsonl, read from its tail) and run.log's tail -> one status line, and ALERT lines on: a kind of her acts refused
# 5+ times with none done, no act of hers ended in 2000 ticks of day, a Traceback / PLAN ERROR / jump guard line, a tick over 300 ms,
# the runner gone. Appends to p1_tools/out/teacher_watch.out. Reads only.
import json, os, time, collections, subprocess, re, sys
import os as _os; L = _os.environ.get("LIFE", "/Users/lukehamond/Projects/project/data/g1_seed1"); OUT = _os.environ.get("WATCH_OUT", "/Users/lukehamond/Projects/project-worktrees/wt_next/tools/life/probes/out/teacher_watch.out")   # (2026-10-19: the life watched)
INTERVAL = int(os.environ.get("INTERVAL", "600")); TAIL = 40_000_000
pos = None; last_act_t = None; faults0 = None
def rows_since(pos):
    p = L + "/ticks.jsonl"; size = os.path.getsize(p)
    start = size - TAIL if pos is None else pos
    start = max(0, start)
    with open(p, "rb") as f:
        f.seek(start); data = f.read(size - start)
    lines = data.split(b"\n")
    if pos is None: lines = lines[1:]
    out = []
    for l in lines:
        if not l.strip(): continue
        try: out.append(json.loads(l))
        except Exception: pass
    return out, size - (len(lines[-1]) if lines and lines[-1].strip() else 0)
def runlog():
    txt = open(L + "/run.log", "rb").read().decode(errors="ignore")
    return txt
while True:
    try:
        rows, pos = rows_since(pos)
        txt = runlog(); tl = [l for l in txt.split("\n") if l.startswith("tick ")]
        ms = int(re.search(r": (\d+) ms a day tick", tl[-1]).group(1)) if tl and re.search(r": (\d+) ms a day tick", tl[-1]) else None
        if ms is not None and re.search(r"'ticks': 1,", tl[-1]):
            ms = None                                                     # (2026-10-19: a restart's first tick is the load's, not the life's pace)
        faults = (txt.count("Traceback"), txt.count("PLAN ERROR"), txt.count("jump guard:"), txt.count("world fault lived through"))
        done = collections.Counter(); ref = collections.Counter(); why = collections.Counter(); jud = collections.Counter(); ev = collections.Counter()
        heard = 0; heard_look = 0; stood = 0; stepped = 0
        for r in rows:
            for a in (r.get("acts_ended") or []):
                if a[0] == "look": continue
                (done if a[2] == "done" else ref)[a[0]] += 1
                if a[2] != "done": why[(a[0], str(a[3])[:70])] += 1
                last_act_t = r["t"]
            if r.get("ep") in ("walk", "command"):
                last_act_t = r["t"]                                      # (2026-10-19: a walk block's child is on the practice ground and
                                                                         # she stays home, W9; in a command block she stays where she calls
                                                                         # from, C351: her acts there are looks only, by design)
            if r.get("refused"):
                x = r["refused"]; ref[x[1]] += 1; why[(x[1], str(x[2])[:70])] += 1
            for j in (r.get("judged") or []): jud[j[1]] += 1
            for e in (r.get("events") or []): ev[e] += 1
            h = [w for w in (r.get("heard") or []) if w]
            if h and r.get("sees") is not None:
                heard += 1; heard_look += int(bool(r.get("sees")))
        t1 = rows[-1]["t"] if rows else None
        alerts = []
        body_ref = collections.Counter()                                  # her body's refusals (a 'request:' refusal is her conduct's own
        for (k, w), c in why.items():                                     # pacing and reading: not a fault of her body)
            if not (w.startswith("request:") or w.startswith("follow-in:") or w.startswith("not true") or w.startswith("not her word") or w.startswith("the new word")
                    or (os.environ.get("QUIET_SMILE") and "the smile alone" in w)): body_ref[k] += c   # her conduct's own (QUIET_SMILE: C364's smile alone is her design; 'the new word ... not last': one frame of an intent while its word is new, the others said, 2026-10-19)
        for k, n in body_ref.items():
            if n >= 5 and done.get(k, 0) == 0 and not all(w.startswith('stay put') for (kk, w), c in why.most_common() if kk == k): alerts.append(f"ALERT {k}: refused {n}, done 0: {[w for (kk, w), c in why.most_common() if kk == k and not (w.startswith('request:') or w.startswith('follow-in:'))][:2]}")   # (C351 amended three times: her stay-put refusals are by design)
        bad_ = [(kk, w, c) for (kk, w), c in why.most_common() if any(m_ in w for m_ in (("form: lowercase",) if os.environ.get("QUIET_VOCAB") else ("not her word", "form: lowercase"))) or ("no line passes the check" in w and c >= 3 and not (os.environ.get("QUIET_COME") and kk == "come_again") and not (os.environ.get("QUIET_VOCAB") and str(kk).startswith("narrate_")))]   # (QUIET_VOCAB: a narration's frames all carry a growth word not yet hers, 2026-10-19)   # (QUIET_VOCAB, 2026-10-19: a newborn's teacher has her birth words only; her growth words refused is her vocabulary, not a fault)   # (one refusal of an intent is a moment's truth rule, not a frame fault)
        if bad_: alerts.append(f"ALERT teacher frames refused by her own check (C351 amended twice): {bad_[:3]}")
        loops_ = [(k_, c_) for k_, c_ in ref.most_common(3) if c_ > 100]       # (2026-10-19: a line refused over 100 times in a window is
        if loops_: alerts.append(f"ALERT a line refused in a loop (whatever the reason): {loops_}")   # a loop, quiet or not: C378's bucket)
        if rows and last_act_t is not None and t1 - last_act_t > 2000 and (t1 % 48000) < 23850: alerts.append(f"ALERT no act of hers ended in {t1 - last_act_t} ticks")
        if faults0 is not None and faults != faults0: alerts.append(f"ALERT faults moved: traceback/plan error/jump guard {faults0} -> {faults}")
        faults0 = faults
        if ms is not None and ms > 400: alerts.append(f"ALERT slow: {ms} ms a tick")
        day_rows = [r for r in rows if not r.get("night")]                 # 2026-10-09 23:10 (the fork's point 8): a setup that hurts ends within the hour
        if len(day_rows) >= 300:
            pain_rate = sum(1 for r in day_rows if r.get("pain")) / len(day_rows); stress_mean = sum(float(r.get("stress", 0)) for r in day_rows) / len(day_rows)
            if pain_rate > float(os.environ.get("PAIN_ALERT", 0.03)): alerts.append(f"ALERT pain {100*pain_rate:.1f}% of ticks in the window (the life's day runs 1%)")
            if stress_mean > float(os.environ.get("STRESS_ALERT", 4.0)): alerts.append(f"ALERT stress {stress_mean:.1f} mean in the window (the day's mean runs about 2)")
        alive = subprocess.run(["pgrep", "-f", "MacOS/Python.*tools/life/sim_life.py --out " + L], capture_output=True).returncode == 0
        if not alive: alerts.append("ALERT runner not found")
        line = (f"{time.strftime('%H:%M')} t {t1} rows {len(rows)} ms {ms} | done {dict(done)} | refused {dict(ref)} | judged {dict(jud)} | events {dict(ev)} "
                f"| hearings {heard} with the toy in view {heard_look} | faults {faults}")
        with open(OUT, "a") as f:
            f.write(line + "\n")
            for a_ in alerts: f.write("  " + a_ + "\n")
            if why: f.write("  refusals: " + str(why.most_common(4)) + "\n")
    except Exception as e:
        with open(OUT, "a") as f: f.write(f"{time.strftime('%H:%M')} watch error {e!r}\n")
    time.sleep(INTERVAL)
