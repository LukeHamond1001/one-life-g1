> **ARCHIVED 2026-10-21. The recordings' ledger (tasks2-10, heldout9): every recording was deleted at 18:00 by the owner's word; the clips of its own body (data/demo/clips_body) are what remains.**

# The recordings (the ledger; 2026-10-21 16:10)

Every set lives in data/demo/<set>_s<seed>/ (chunk_*.npz: the senses obs__*, the acts act__*, the labels lab__* where the act done
was not the act meant, her face, the lever where recorded; labels_*.jsonl; tasks.jsonl, the judge's line per task; meta.json, the
generator's flags). The trainer relives a set through `--demo-glob data/demo/<set>_s*` (tools/harness/demo_train.py), the first glob's first
recording first, then the newest; failed tasks' ticks are dropped at relive time (D1w). Nothing under data/ is deleted, moved or
rewritten without the owner's word.

| set | made | recordings / ticks | what it holds | use |
|---|---|---|---|---|
| tasks2 to tasks7 | 2026-10-19 to 10-20 | ~700k | earlier generator versions; **no pretrained-eye channel (eye_d)** | not for the brain as it is (blind to its eye) |
| tasks8 (from s8140) | 10-20 to 10-21 | 60 / 144k (100 / 260k with s8100 to s8139, an older generator) | all seven families + take, name-held; the curriculum; the pretrained eye; one-tick noise labelled (7% of ticks) | P2, P3, P4's diet from s8140 |
| tasks9 | 10-21 | 24 / 64k | tasks8's families with wanders (R2, 0.0025: 5 to 6 a recording per effector, 8 to 22 ticks, labelled with the intended act) | P3's and P4's diet |
| heldout9 | 10-21 | 2 / 5.5k | the held-out ruler's recordings, never lived | measurement only |
| **tasks10 (the clean set)** | from 10-21 15:57 | growing (tools/pretrain/gen_clean.sh) | every family but `where` (success 0.53 to 0.71; the rest 0.72 to 1.00), 12 scanned objects a recording (data/objects/gso, 142 fetched), the judge's lever at every task met, wanders 0.0025, the pretrained eye | the next pretraining round, continued from P4's student7; the live-teaching objects (20 held out of it later) |
| | | | from 16:52 each recording is also filmed from the room camera (film.mp4 at 3 fps with film_captions.jsonl: the task and phase at each change; `demo_tasks --film`): THE SCREEN's clip library of its own body doing the tasks (docs/archive/PREREGISTRATION_LIVE.md 4d) | |
| | | | 17:30: the take asks of scanned objects had all been refused (0 of 4: her percept named one `cup_7`, which her line check's form refuses, and the reach list had taken every scanned word, shoes and bowls she has no word for). Fixed: the percept names a scanned object by its stem (`cup`), and only scanned objects with a birth word (cup: 22 of 142, car: 2) join the reach and naming lists; the rest stay the room's clutter. From the recordings after s10109. | |

Success rates by family (tasks8 from s8140 and tasks9): what 1.00, greet 1.00, call 1.00, name-held 0.99, peekaboo 1.00, bye 1.00, echo
0.94 to 0.97, take 0.86 to 0.89, come 0.85, find 0.81 to 0.90, look 0.72 to 0.76, where 0.53 to 0.71. The clean set asks none of
`where`; its expert's stalls were the one family whose failures needed reading.

## Generation

`tools/pretrain/gen_clean.sh` (start after any stop or restart of the Mac: `nohup tools/pretrain/gen_clean.sh > data/demo/gen_clean.log 2>&1 &`): two
slots from wt_next, none while a student sleeps, one while a home test runs; each recording 3,000 ticks in a fresh process. Its
settings are its environment (PREFIX, SEED0, FAMS, OBJECTS, WANDER, STUDENTS). The objects: `tools/harness/gso_fetch.py` (the owner's yes to
Google Scanned Objects, 10-21 15:12), placed per seed by body/sim/extras.add_gso through `demo_tasks.py --objects N`.

## The model the next round starts from

Not from scratch: P4's student7 (seed 7; its night snapshots in /private/tmp/claude-501/p4_snaps, its live save data/demo/student7) —
the brain whose gaze was on the expert's side (0.57 and 0.63) after one night. When the clean set is large enough (about 200k ticks),
P4 stops and the student continues (`demo_train.py --resume` with the clean set added to its globs) as the next pre-registered run;
the living phase (docs/archive/PREREGISTRATION_LIVE.md) wakes the same brain from its latest tested snapshot.
