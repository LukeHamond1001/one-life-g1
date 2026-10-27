"""THE CURRICULUM (P8, 2026-10-22; the owner's protocol: "show a task, push and pay until it does it with no push and no pay; then a new task;
a new task shown first as the test: if it cannot, push and pay until it can; keep going until it figures out that what is on the iPad
is what we want"). A stage is the taught acts (interleaved: the clip alone tells which), one reserve act shown on probes only (never
pushed, never paid: the standing one-shot test), and its NEWEST act, whose first three tasks are probes (the zero-shot test; the
harness's --zero-shot-probes). MASTERY: the newest act's shown act done on CRIT_K of its last CRIT_N probes (no push, no pay). At mastery
(or after CAP_H hours on the stage) the next stage begins at P8's next fresh save: the next act enters, the mastered acts' pushes are
faded (P8_STUCK_FAMS act:FADE, ticks of its own before the push), the reserve moves on. Every decision is written to STATE (json) and
LOG. Run once in the background; it resumes from STATE.
Usage: python3 tools/live/curriculum_p8.py  (the live P8 run must be the learning set of 19:02, stage 0)"""
import datetime as dt
import json
import os
import subprocess
import time

D = "/Users/lukehamond/Projects/project/data/demo"; O = D + "/p8"; C = D + "/clips_box"
N = "/Users/lukehamond/Projects/project-worktrees/wt_next"
S = "/private/tmp/claude-501/-Users-lukehamond-Projects-project/42bb6702-f260-4f7f-9ccb-b32917f5d9be/scratchpad"
STATE = S + "/curriculum_state.json"; LOG = S + "/curriculum.log"; LIVE_ENV = S + "/p8_live_env.json"
CRIT_K, CRIT_N, CAP_H, FADE = 2, 3, 4.0, 8
CLIPS = {"box": C + "/front_in_s2", "shake": C + "/front_shake_s2", "lift": C + "/front_lift_s1", "wave": C + "/front_wave_s1",
         "out": C + "/front_out_s1", "cross": C + "/front_cross_s1", "low": C + "/front_low_s1", "side": C + "/front_side_s1",
         "poke": C + "/front_poke_s1"}
STAGES = [dict(taught=["box", "shake", "lift"], reserve="wave", newest="lift"),           # (22:55 10-22: the wave the held-out act, the
          dict(taught=["box", "shake", "lift", "out"], reserve="wave", newest="out"),     # final new-act test: from the front the hold-out
          dict(taught=["box", "shake", "lift", "out", "low"], reserve="wave", newest="low"),           # reads as the put-in's reach; the cross's
          dict(taught=["box", "shake", "lift", "out", "low", "side"], reserve="wave", newest="side"),  # push cannot reach). 02:5x 10-23: THE
          dict(taught=["box", "shake", "lift", "out", "low", "side", "wave"], reserve="poke", newest="wave")]   # LEARNING-TO-LEARN ACTS: the
                                                                                          # low and the side taught one at a time under one
                                                                                          # teacher (the fade, the pay by the help) before the
                                                                                          # wave: each new act's lessons to its first own test
                                                                                          # (23:2x 10-23: the poke, forward and back, stage 4's
                                                                                          # never-taught reserve: the owner's final new-act test)


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return dict(stage=0, since="2026-10-22 19:02", mastered=["box"])   # stage 0 began 19:02 (the put-in faded since 19:19)


def probes_of(fam, since):
    out = []
    for l_ in open(O + "/tasks_live.jsonl"):
        try:
            r = json.loads(l_)
        except ValueError:
            continue
        if r.get("fam") == fam and r.get("unasked") and r.get("at", "") >= since:
            out.append(bool((r.get("box") or {}).get("own_act")))
    return out


def deploy(st):
    g = STAGES[st["stage"]]
    fams = g["taught"] + ([g["reserve"]] if g["reserve"] else [])
    cls = [CLIPS[f] for f in fams]
    env = dict(os.environ, P8_LOG_STATE="1", P8_TAKE_BACK="1", P8_STUCK_BOX="0", P8_PROFILE_NIGHT="1", P8_ONE_FRAME="1",
               P8_STUCK_FAMS=",".join(f"{a}:{FADE}" for a in st["mastered"]), FAMS=",".join(fams), PROBE_ONLY=g["reserve"] or "",
               PROBE_ONLY_EVERY="3", ZERO_SHOT="3", PROBE_TOYS="all", TOYS="duck,cup")
    for i, cl in enumerate(cls[1:], start=2):
        env[f"CL{i}"] = cl
    if os.path.exists(LIVE_ENV):                                         # the live run's other settings (the push's gentleness, the
        env.update({k: str(v) for k, v in json.load(open(LIVE_ENV)).items()})   # lift's reach: whatever the last deploy chose) carried over
    while time.time() - os.path.getmtime(O + "/life.pt") >= 15:          # the next fresh save
        time.sleep(2)
    time.sleep(20)                                                       # (09:3x 10-23: a dawn's save comes before its night's record and
                                                                         # the morning's stand; stopped at once, night 24's record was lost)
    subprocess.run(["pkill", "-f", f"demo_test.py --student .* --out {O} "])
    for _ in range(90):
        if subprocess.run(["pgrep", "-f", f"demo_test.py --student .* --out {O} "], capture_output=True).returncode != 0:
            break
        time.sleep(1)
    r = subprocess.run(["bash", "tools/live/resume_p8.sh", O, cls[0]], cwd=N, env=env, capture_output=True, text=True)
    log(f"DEPLOYED stage {st['stage']}: taught {g['taught']}, reserve {g['reserve']}, newest {g['newest']}, faded {st['mastered']} "
        f"| {r.stdout.strip()[-120:]}")


def main():
    st = load()
    log(f"the curriculum watches: stage {st['stage']} {STAGES[st['stage']]} since {st['since']}, mastered {st['mastered']}")
    while True:
        g = STAGES[st["stage"]]
        pr = probes_of(g["newest"], st["since"])
        met = len(pr) >= CRIT_N and sum(pr[-CRIT_N:]) >= CRIT_K
        hours = (dt.datetime.now() - dt.datetime.strptime(st["since"], "%Y-%m-%d %H:%M")).total_seconds() / 3600.0
        if (met or hours >= CAP_H) and st["stage"] + 1 < len(STAGES):
            why = f"MASTERED ({sum(pr[-CRIT_N:])} of its last {CRIT_N} probes)" if met else f"the cap ({CAP_H} h) with {sum(pr)} of {len(pr)} probes"
            log(f"stage {st['stage']}'s newest act {g['newest']}: {why}; the next stage at the next save")
            if met:
                st["mastered"] = sorted(set(st["mastered"]) | {g["newest"]})
            st["stage"] += 1; st["since"] = f"{dt.datetime.now():%Y-%m-%d %H:%M}"
            json.dump(st, open(STATE, "w"))
            deploy(st)
        time.sleep(300)


if __name__ == "__main__":
    main()
