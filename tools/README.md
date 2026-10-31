# tools/

The scripts around the body (`body/` is the architecture; nothing here changes it). Each folder has one job.

| folder | what it holds |
|---|---|
| `harness/` | **The harness** of the living run: the demo sim and its record (`demo_sim.py`), the task generator and expert (`demo_tasks.py`), the box task (`demo_box.py`), the student's test and living run (`demo_test.py`), the film (`demo_film.py`), the screen's clips and show (`demo_clips.py`, `demo_screen.py`), the shown tasks (`demo_shown.py`), the scanned objects' fetch (`gso_fetch.py`). The modules import each other by name, so they stay together. |
| `live/` | **The living runs' scripts**: `start_nb2.sh` (NB2, the second newborn), `nb2_config.sh` (NB2's brain and body switches, shared with the levels), `start_nb.sh` (NB1, the recipe NB2 began from), `watch_page.py` (the live page). |
| `levels/` | **The workshops** (`docs/LEVELS.md`, from 2026-10-27; Level 1 before them): the parent moves its body, never writes into its brain. `workshop1.py` is the shared workshop (the walk, name, find, pick and put tasks, the chain, her hands, the pushes, the frames and the record), `workshop2.py` to `workshop28.py` the workshops, `workshop13.py` the test; `test_chain.sh` the chain test as independent single passes (the first stage's bar), `rounds.sh` whole-chain practice rounds (every toy in turn, never one alone); `workshop_view.py` the live viewer, `sheet.py` and `workshop_sheet.py` frame sheets; `lines_check.py`, `words_check.py`, `eyes_check.py`, `stance_at_pick.py`, `bumps_summary.py` readings; `saves.py` and `workshop_name.py` the two saves. **The language workshops**: `lessons.py` the picture lessons and `lessons_sheet.py` what it calls things, on pictures from `photos_fetch.py` (COCO), `sim_pictures.py` (the scanned objects) and `rooms_fetch.py` (the rooms' floors). Level 1: `level1_walk.py`, `start_level1.sh`, `loop_level1.sh`, `viewer.py`. |
| `compare/` | **The baselines** a student is measured against in the same harness (`demo_test.py --student baseline:<checkpoint>`): behavior cloning (`baseline_bc.py`), Dreamer-style (`baseline_dreamer.py`), a diffusion policy (`baseline_frontier.py`). |
| `measure/` | **The instruments**: the behaviour guard (`determinism_check.py`, its pins in `pins/`), the sim profile, the day's rulers and growth chart, the sim's measurements (eyes, face, friction, grasp, pain, the parent's ear and motion, the voices). |
| `life/` | **The diary life's runner** (`sim_life.py`), its page, the world's migration, the stills, the replay (`replay.sh`). |

The pretraining runs' scripts (P1-P5), the P1/P2/P8 probes and conductors, the earlier living runs' scripts (P6-P8) and the diary
life's probes left the tree on 2026-10-27; they stay in the git history.

Run everything from the repository root (`python3 tools/harness/demo_test.py ...`, `tools/live/start_nb2.sh`); each script finds the root itself.
