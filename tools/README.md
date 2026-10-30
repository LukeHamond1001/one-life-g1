# tools/

The scripts around the body (`body/` is the architecture; nothing here changes it). Each folder has one job.

| folder | what it holds |
|---|---|
| `harness/` | **The harness** of the living run: the demo sim and its record (`demo_sim.py`), the task generator and expert (`demo_tasks.py`), the box task (`demo_box.py`), the student's test and living run (`demo_test.py`), the film (`demo_film.py`), the screen's clips and show (`demo_clips.py`, `demo_screen.py`), the shown tasks (`demo_shown.py`), the scanned objects' fetch (`gso_fetch.py`). The modules import each other by name, so they stay together. |
| `live/` | **The living runs' scripts**: `start_nb2.sh` (NB2, the second newborn), `nb2_config.sh` (NB2's brain and body switches, shared with the levels), `start_nb.sh` (NB1, the recipe NB2 began from), `watch_page.py` (the live page). |
| `levels/` | **The levels** (`docs/LEVELS.md`, from 2026-10-27): one skill at a time from NB2's newest save, each passed only by doing the teacher's demonstration alone. `level1_walk.py`, `start_level1.sh` (one run) and `loop_level1.sh` (runs again and again until it passes) (Level 1: walk forward, eyes ahead), `sheet.py` (a run's pictures tiled on one sheet, to check it frame by frame), `saves.py` (two saves only: the current level's brain and the previous level's), `viewer.py` (the live view: no pictures, every tick read and checked as the run writes it). |
| `compare/` | **The baselines** a student is measured against in the same harness (`demo_test.py --student baseline:<checkpoint>`): behavior cloning (`baseline_bc.py`), Dreamer-style (`baseline_dreamer.py`), a diffusion policy (`baseline_frontier.py`). |
| `measure/` | **The instruments**: the behaviour guard (`determinism_check.py`, its pins in `pins/`), the sim profile, the day's rulers and growth chart, the sim's measurements (eyes, face, friction, grasp, pain, the parent's ear and motion, the voices). |
| `life/` | **The diary life's runner** (`sim_life.py`), its page, the world's migration, the stills, the replay (`replay.sh`). |

The pretraining runs' scripts (P1-P5), the P1/P2/P8 probes and conductors, the earlier living runs' scripts (P6-P8) and the diary
life's probes left the tree on 2026-10-27; they stay in the git history.

Run everything from the repository root (`python3 tools/harness/demo_test.py ...`, `tools/live/start_nb2.sh`); each script finds the root itself.
