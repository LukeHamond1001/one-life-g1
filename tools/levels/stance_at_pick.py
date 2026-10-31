"""WHERE IT STOOD AS ITS PICK-UPS BEGAN (the first stage, 10-30): where the body stood when its pick-up began (pelvis, feet, heading, in its own heading frame, from the thing), against the outcome"""
import json, glob, os, math, sys
pat = sys.argv[1] if len(sys.argv) > 1 else 'data/workshops/workshop13/chain1s_*'
out = []
for R in sorted(glob.glob(pat)):
    P = R + '/workshop13'
    if not os.path.exists(P + '/passes.jsonl'): continue
    try: d = json.loads(open(P + '/passes.jsonl').readline())
    except Exception: continue
    pu = d.get('put') or {}
    o = 'IN' if pu.get('in_it') else ('dropped' if pu.get('dropped') is not None else ('not-in' if pu else ('picked?' if d.get('picked') else 'not-picked')))
    tk = [json.loads(l) for l in open(P + '/pass_000/ticks.jsonl')]
    pk = [r for r in tk if r['phase'] == 'pick' and (r.get('pick') or {}).get('block')]
    if not pk: continue
    k0 = pk[0]['k']; th = pk[0]['pick']['block']
    rows = [json.loads(l) for l in open(P + '/pass_000/rows.jsonl')]
    r0 = [r for r in rows if r['k'] == k0][0]
    yaw = math.radians(r0['yaw']); c, s = math.cos(yaw), math.sin(yaw)
    def loc(xy):   # the thing's place in the body's heading frame from this point: ahead, left
        dx, dy = th[0] - xy[0], th[1] - xy[1]
        return (dx * c + dy * s, -dx * s + dy * c)
    pv = loc(r0['pelvis']); f = [loc(ft['xy']) for ft in r0['feet']]
    fm = loc([(r0['feet'][0]['xy'][0] + r0['feet'][1]['xy'][0]) / 2, (r0['feet'][0]['xy'][1] + r0['feet'][1]['xy'][1]) / 2])
    bear = math.degrees(math.atan2(pv[1], pv[0]))
    out.append((fm[0], R.split('chain1s_')[1], o, pv, fm, f, bear, d.get('stop_d_pelvis')))
out.sort()
print('ahead-of-feet-mid  pass                outcome     pelvis->thing(ahead,left)  feetmid->thing  feet split(ahead L-R)  bearing  stop_d')
for fm0, n, o, pv, fm, f, b, sd in out:
    print(f"{fm0:+.3f}  {n:22s} {o:10s}  ({pv[0]:+.3f},{pv[1]:+.3f})  ({fm[0]:+.3f},{fm[1]:+.3f})  {f[1][0]-f[0][0]:+.3f}  {b:+6.1f}  {sd}")
