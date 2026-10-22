"""the ghost blocks as the record shows them: python3 ghost_live.py [N_rows_back]  (C350; the record's ghost row [s, share, heading, falls, metres])"""
import json, sys, subprocess, collections, numpy as np
L = "/Users/lukehamond/Projects/project/data/g1_seed1/ticks.jsonl"; N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
rows = []
for line in subprocess.run(["tail", "-n", str(N), L], capture_output=True, text=True).stdout.splitlines():
    try: rows.append(json.loads(line))
    except Exception: pass
g = [r for r in rows if r.get("ghost")]
if not g: print(f"no ghost rows in the last {len(rows)} ({rows[0]['t'] if rows else '?'}..{rows[-1]['t'] if rows else '?'})"); sys.exit()
blocks = []; prev = None
for r in g:
    if prev is None or r["t"] - prev["t"] > 5: blocks.append([r])
    else: blocks[-1].append(r)
    prev = r
def kinds(x): return [str(e[0]) if isinstance(e, (list, tuple)) and e else str(e) for e in (x or [])]
for b in blocks:
    s = np.array([r["ghost"][0] for r in b]); sh = np.array([r["ghost"][1] for r in b]); last = b[-1]["ghost"]
    ev = collections.Counter(k for r in b for k in kinds(r.get("events")))
    pain = sum(1 for r in b if r.get("pain")); up = sum(1 for r in b if (r.get("child") or [0,0,""])[2] == "sitting")
    rew = sum(float(r.get("reward") or 0) for r in b if (r.get("reward") or 0) > 0)
    print(f"block {b[0]['t']}..{b[-1]['t']} ({len(b)} ticks): strength {s[0]:.3f} -> {last[0]:.3f} (min {s.min():.3f}, mean {s.mean():.3f}); carried mean {sh.mean():.2f} (<0.3 on {100*np.mean(sh<0.3):.0f}%, >0.6 on {100*np.mean(sh>0.6):.0f}%); "
          f"metres {last[4]:.1f}; falls {last[3]}; stood {ev.get('stood',0)} stepped {ev.get('stepped',0)} fell {ev.get('fell',0)}; trunk up {100*up/len(b):.0f}%; pain {pain} ({100*pain/len(b):.1f}%); reward + {rew:.1f}; stress {b[-1].get('stress',0):.2f}")
    falls_at = [round(a['ghost'][0], 2) for a, r in zip(b, b[1:]) if r['ghost'][3] != a['ghost'][3]]
    xy = np.array([r['child'][:2] for r in b]); print(f"   falls at strength {falls_at[:20]}; room x {xy[:,0].min():.1f}..{xy[:,0].max():.1f} y {xy[:,1].min():.1f}..{xy[:,1].max():.1f}; sees {100*np.mean([1 if r.get('sees') else 0 for r in b]):.0f}%")
    # the strength's path in tenths of the block
    k = max(1, len(b)//10); print("   strength by tenth:", [round(float(s[i:i+k].mean()),3) for i in range(0, len(b), k)][:10], " heading", [round(r['ghost'][2],1) for r in b[::max(1,len(b)//6)]][:6])
