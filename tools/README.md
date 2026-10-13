# tools

The life, its rulers and its guards. Run everything from the repository root.

- `sim_life.py` — the life. A new one: `python3 tools/sim_life.py --out data/mylife --days 1 --d 512 --seed 1 --voice fake --threads 4 --room door --extra book_box_bucket`.
  Resume a saved pair with `--resume`. `--voice real` uses the macOS speech engine through the pair's voice cache; `--voice cached`
  uses the cache alone (a replay on any machine); `--voice fake` is tones. `--page` serves a live view on port 8030.
- `replay.sh` — downloads a released pair, replays it for `TICKS` (1,500 by default: about ten minutes) and prints its rulers.
- `sim_rulers.py` — the day's rulers from a life's record: `python3 tools/sim_rulers.py data/mylife --day 3`.
- `determinism_check.py` — the behaviour guard: a tiny body lives a fixed script and every weight, memory and feeling is hashed.
  `--profile sim` is the G1's. The pinned digests and how each was taken are in `pins/digests.txt`; two machines with the same
  thread count must print the same digest.
- `sim_page.py` — the live page the runner serves with `--page`, and the studio view it draws (also `--film` frames): the robot and
  toys alone, her voice where it comes from, its words, and its reward, mood and stress bars.
- `sim_migrate_world.py` — carries a saved life into a changed room or body (new toys, the door).
- The probes (`sim_*.py`, `cereb_day.py`, `motor_units.py`) measure one organ or one sense on a saved pair; each file's first lines say
  what it reads and how to run it.

Tests are plain functions under `body/tests/`. Run one with the room set:

```bash
ROOM=door python3 -c "import sys; sys.path.insert(0, '.'); from body.tests.test_sim_world import test_the_scene as t; t()"
```
