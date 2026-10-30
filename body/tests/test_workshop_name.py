"""THE NAMING WORKSHOPS (tools/levels/workshop_name.py, workshop1's name task): her graded reaction to its try, the workshops' words
in the room and in her ear, the options handed to the passes, and the owner's two saves kept as the workshops go on"""
import math
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "levels")); sys.path.insert(0, os.path.join(ROOT, "tools", "harness"))
import workshop1 as WS1  # noqa: E402
import workshop_name as WN  # noqa: E402


class _H:
    def __init__(self, word, exact, d, d_bank):
        self.word, self.exact, self.d, self.d_bank = word, exact, d, d_bank


def test_her_reaction_grows_as_its_try_sounds_more_like_her_word():
    far, near = WS1.her_reaction(_H(None, False, 20.0, 12.0), "ball"), WS1.her_reaction(_H(None, False, 15.0, 12.0), "ball")
    heard, right = WS1.her_reaction(_H("ball", False, 15.0, 12.0), "ball"), WS1.her_reaction(_H("ball", True, 15.0, 12.0), "ball")
    assert far < near < heard < right and heard - near == 5.0 and right - heard == 5.0
    unsure = WS1.her_reaction(type("H", (), dict(word=None, exact=False, d=15.0, d_bank=12.0, d_best=13.0))(), "ball")
    assert unsure == near - WS1.DISTINCT * 2.0                                                # (another of her words 2 nearer: less)
    assert WS1.her_reaction(_H(None, False, math.inf, 12.0), "ball") == -10.0 and WS1.her_reaction(None, "ball") == -10.0


def test_the_words_are_the_rooms_toys_her_ear_knows():
    from body.sim import make_g1room as MR
    assert len(WN.WORDS) == 8 and sorted(WN.WORDS) == list(range(3, 11))
    assert set(WN.WORDS.values()) <= set(MR.TOYS)
    import json, numpy as np
    z = np.load(os.path.join(ROOT, "data", "workshops", "brain", "voice", "parent_ear.npz"), allow_pickle=True)
    words = json.loads(bytes(z["manifest"].astype(np.uint8)).decode())["words"]
    assert set(WN.WORDS.values()) <= set(words)


def test_its_options_reach_the_passes(monkeypatch):
    seen = {}
    monkeypatch.setattr(WS1, "main", lambda argv=None: seen.setdefault("argv", list(argv)))
    WN.main(["4", "--out", "/x", "--no-save", "--passes", "2"])
    a = seen["argv"]
    assert a[a.index("--task") + 1] == "name" and a[a.index("--word") + 1] == "cup" and a[a.index("--name") + 1] == "workshop4"


def test_a_workshop_run_again_moves_no_save(monkeypatch):
    seen = {}; called = []
    monkeypatch.setattr(WS1, "main", lambda argv=None: seen.setdefault("argv", list(argv)))
    monkeypatch.setattr(WN, "keep_previous", lambda *a, **k: called.append(a) or (False, []))
    WN.main(["3", "--out", "/x", "--again", "--passes", "2"])
    assert not called and "--again" not in seen["argv"] and seen["argv"][seen["argv"].index("--word") + 1] == "ball"


def test_two_saves_kept_as_the_workshops_go_on(tmp_path):
    root, trash, brain = tmp_path / "w", tmp_path / "trash", tmp_path / "brain"
    for d_ in (root, trash, brain, root / "brain_after_w1"):
        d_.mkdir(parents=True)
    (brain / "life.pt").write_bytes(b"w2"); (brain / "legs_state.pt").write_bytes(b"legs2")
    (root / "brain_after_w1" / "life.pt").write_bytes(b"w1"); (root / "brain_after_w1" / "legs_state.pt").write_bytes(b"legs1")
    copied, moved = WN.keep_previous(3, str(brain), str(root), str(trash))
    assert copied and (root / "brain_after_w2" / "life.pt").read_bytes() == b"w2"
    assert len(moved) == 2 and not (root / "brain_after_w1").exists() and all(str(trash) in m for m in moved)
    assert WN.keep_previous(3, str(brain), str(root), str(trash)) == (False, [])                # (once)
