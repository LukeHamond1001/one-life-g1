# tools/

The scripts around the body (`body/` is the architecture; nothing here changes it). Each folder has one job.

| folder | what it holds |
|---|---|
| `harness/` | **The harness** shared by the living run and the pretraining relive: the demo sim (`demo_sim.py`), the task generator and expert (`demo_tasks.py`), the relive trainer (`demo_train.py`), the student's test and living run (`demo_test.py`), the learning curve and held-out ruler (`demo_curve.py`, `demo_holdout.py`), the generator slots and watchdog, the film (`demo_film.py`, `film_pair.py`), the screen's clips and show (`demo_clips.py`, `demo_screen.py`), the shown tasks, the scanned objects' fetch. The modules import each other by name, so they stay together. |
| `live/` | **The living run** (P6, the iPad; docs/PREREGISTRATION_P6.md): `start_p6.sh`, `resume_p6.sh`, `screen_rung1.sh`; the earlier living test's start/resume. |
| `pretrain/` | **The relive** (P1-P5, finished 2026-10-21): the clean set's daemon (`gen_clean.sh`), the runs' start/resume scripts, the stop. Kept as the record of the method. |
| `compare/` | **The withdrawn comparison**: the behaviour-cloning, diffusion and Dreamer-style baselines at the student's size, the frontier follower and table, P1's conductor and report. |
| `measure/` | **The instruments**: the behaviour guard (`determinism_check.py`, its pins in `pins/`), the sim profile, the day's rulers and growth chart, the sim's measurements (eyes, face, friction, grasp, pain, the parent's ear and motion, the sink rates, the voices), the P1 probes. |
| `life/` | **The diary life's runner** (`sim_life.py`), its page, the world's migration, the film's cut, the stills, the forecast repair, the replay; `probes/` holds the life's probe and watch scripts (the sentinel, alert_wake, the day's diagnostics). |

Run everything from the repository root (`python3 tools/harness/demo_test.py ...`, `tools/live/resume_p6.sh`); each script finds the root itself.
