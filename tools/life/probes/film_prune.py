"""THE FILM'S PRUNE (the disk rule, 2026-10-10): data/g1_seed1/film holds a JPEG every third tick (tools/life/sim_life.py --film-every 3).
For every day older than KEEP_DAYS the frames outside one kept window a day are removed (the window: WINDOW ticks from the day's
offset in film_kept.json's "windows" [day, offset, n], or the day's first WINDOW ticks when it has none); the last KEEP_DAYS days stay
whole, for the demo's cut. Dry by default: FILM_PRUNE_DO=1 removes. Never touches ticks.jsonl, the saves or the voice."""
import json, os, sys, collections
L = "/Users/lukehamond/Projects/project/data/g1_seed1"; FILM = L + "/film"; KEPT = L + "/film_kept.json"
KEEP_DAYS = int(os.environ.get("KEEP_DAYS", "4")); WINDOW = int(os.environ.get("WINDOW", "2400")); DO = os.environ.get("FILM_PRUNE_DO") == "1"
names = sorted(n for n in os.listdir(FILM) if n.endswith(".jpg"))
if not names:
    sys.exit("no frames")
kept = json.load(open(KEPT)) if os.path.exists(KEPT) else {"kept": 0, "removed": 0, "windows": []}
win = {int(d): (int(o), int(n)) for d, o, n in kept.get("windows", [])}
last_day = int(names[-1][:-4]) // 48000
by_day = collections.defaultdict(list)
for n in names:
    by_day[int(n[:-4]) // 48000].append(n)
rm = []; kept_n = 0
for d, fs in sorted(by_day.items()):
    if d > last_day - KEEP_DAYS:
        kept_n += len(fs); continue
    o = win.get(d, (0, 0))[0]
    t0 = d * 48000 + o; t1 = t0 + WINDOW
    for n in fs:
        t = int(n[:-4])
        if t0 <= t < t1 and (t % 48000) < 24000:
            kept_n += 1
        else:
            rm.append(n)
    if d not in win:
        win[d] = (o, sum(1 for n in fs if t0 <= int(n[:-4]) < t1))
size = 0
for n in rm:
    try: size += os.path.getsize(os.path.join(FILM, n))
    except OSError: pass
print(f"film: {len(names)} frames, days {min(by_day)}..{last_day}; keep the last {KEEP_DAYS} days whole and {WINDOW} ticks a day before: remove {len(rm)} frames ({size/1e9:.2f} GB), keep {kept_n}; {'REMOVING' if DO else 'dry run (FILM_PRUNE_DO=1 to remove)'}")
if DO:
    for n in rm:
        try: os.remove(os.path.join(FILM, n))
        except OSError: pass
    kept["kept"] = kept_n; kept["removed"] = int(kept.get("removed", 0)) + len(rm)
    kept["windows"] = sorted([int(d), int(o), int(k)] for d, (o, k) in win.items())
    json.dump(kept, open(KEPT, "w"))
    print("removed; film_kept.json updated")
