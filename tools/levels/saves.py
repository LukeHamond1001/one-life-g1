"""THE LEVELS' SAVES (2026-10-27, the owner's word: "two saves, current level and one save for previous level going forward. only two
versions"). The brain going through the levels is kept in two versions and no more:
  - THE CURRENT LEVEL'S: its folder's brain (BRAIN: life.pt and legs_state.pt), written over in place at each night and at a run's end;
  - THE PREVIOUS LEVEL'S: the save the current level began from, in that level's folder, never written again (a level begins only
    from a level passed; Level 1 began from NB2's newest save, data/demo/nb2).
A level's first run copies the previous level's brain into its folder and writes where it began (FROM_FILE: `begin_level`); every run
of it then moves to the Trash the brain the previous level itself began from, two levels back (`keep_two`: kept at each run's start, so
a first run killed between its copy and the move still ends with two). A dry run (--no-save) reads the previous level's brain where it
lies and moves nothing."""
import json
import os
import shutil
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEVELS = os.path.join(ROOT, "data", "levels")   # the levels' folders: the rule moves a brain only for a level here (a scratch run's
                                                # --out elsewhere, begun from a level, is no level: it moves nothing)
BRAIN = ("life.pt", "legs_state.pt")        # the brain: its life (every organ, the store, the day) and the leg layer below it
FROM_FILE = "level_from.json"               # where a level's brain began: the previous level's folder
TRASH = os.path.expanduser("~/.Trash")


def _rel(path):
    """a folder as kept in FROM_FILE: from the project's root when inside it (the record moves with the project)"""
    p_ = os.path.realpath(path)
    return os.path.relpath(p_, ROOT) if p_.startswith(ROOT + os.sep) else p_


def began_from(folder):
    """the folder a level's brain was copied from (its FROM_FILE), or None (NB2, or a level from before the file was kept)"""
    p_ = os.path.join(folder, FROM_FILE)
    if not os.path.exists(p_):
        return None
    f_ = json.load(open(p_))["from"]
    return f_ if os.path.isabs(f_) else os.path.join(ROOT, f_)


def record_from(out, src, at=None):
    """write where the level's brain began (out/FROM_FILE)"""
    with open(os.path.join(out, FROM_FILE + ".tmp"), "w") as f_:
        json.dump({"from": _rel(src), "at": at or time.strftime("%Y-%m-%d %H:%M:%S")}, f_, indent=1)
    os.replace(os.path.join(out, FROM_FILE + ".tmp"), os.path.join(out, FROM_FILE))


def to_trash(folder, trash=TRASH):
    """the brain of a folder moved to the Trash (in a folder of its own, named for where it lay) -> the paths it now has"""
    have = [f_ for f_ in BRAIN if os.path.exists(os.path.join(folder, f_))]
    if not have:
        return []
    dst0 = os.path.join(trash, f"brain-{os.path.basename(os.path.normpath(folder))}-{time.strftime('%Y%m%d-%H%M%S')}")
    dst, n_ = dst0, 1
    while os.path.exists(dst):
        n_ += 1; dst = f"{dst0}-{n_}"
    os.makedirs(dst)
    out = []
    for f_ in have:
        shutil.move(os.path.join(folder, f_), os.path.join(dst, f_)); out.append(os.path.join(dst, f_))
    return out


def begin_level(src, out, dry=False):
    """THE LEVEL'S BRAIN AT ITS FIRST RUN (out holds no life.pt yet): the previous level's brain (src) copied into out and where it
    began recorded (dry: read where it lies, nothing recorded). The voice (her voice and ear, src/voice) is copied in either case
    -> (the life to load, the legs to load)"""
    if os.path.realpath(src) == os.path.realpath(out):
        raise SystemExit("--from and --out are the same folder")
    for f_ in BRAIN:
        if not os.path.exists(os.path.join(src, f_)):
            raise SystemExit(f"--from {src} has no {f_}")
    os.makedirs(out, exist_ok=True)
    if os.path.isdir(os.path.join(src, "voice")) and not os.path.exists(os.path.join(out, "voice")):
        shutil.copytree(os.path.join(src, "voice"), os.path.join(out, "voice"))
    if dry:
        return os.path.join(src, "life.pt"), os.path.join(src, "legs_state.pt")
    life_p, legs_p = os.path.join(out, "life.pt"), os.path.join(out, "legs_state.pt")
    shutil.copy2(os.path.join(src, "legs_state.pt"), legs_p)
    shutil.copy2(os.path.join(src, "life.pt"), life_p + ".tmp")
    record_from(out, src)
    os.replace(life_p + ".tmp", life_p)                               # (life.pt last: its presence is the copy done and recorded)
    return life_p, legs_p


def keep_two(out, trash=TRASH, levels=LEVELS):
    """THE RULE AT EVERY RUN'S START: of the brain going through the levels only the current level's (out) and the previous level's
    (the one out began from) kept; the one before (the brain the previous level began from) moved to the Trash -> the files moved.
    Only for a level's own folder (under `levels`)"""
    if not os.path.realpath(out).startswith(os.path.realpath(levels) + os.sep):
        return []
    src = began_from(out)
    older = began_from(src) if src is not None else None
    if older is None or os.path.realpath(older) in (os.path.realpath(src), os.path.realpath(out)):
        return []
    return to_trash(older, trash)
