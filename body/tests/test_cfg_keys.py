"""I2: every constant the body's code reads by name (`cfg.get(k, ...)`, `_const(k)`) is declared in a constant table or listed in
body/life.py's INLINE_KEYS, so the unknown-key warning names only keys nothing reads; the sim's constants are all read"""
import glob
import re


def _reads():
    keys = set()
    for f in glob.glob("body/**/*.py", recursive=True):
        if "/tests/" in f:
            continue
        for line in open(f, errors="replace"):
            keys.update(re.findall(r"cfg\.get\(\s*[\"']([a-z_0-9]+)[\"']", line))
            keys.update(re.findall(r"_const\(\s*[\"']([a-z_0-9]+)[\"']", line))
    return keys


def test_every_key_read_is_known():
    from body.core.physiology import PHYSIOLOGY, SWITCHES, MOTOR, CEREB, REFLEX, FRAMES, AMYG, SLEEP
    from body.life import INLINE_KEYS, RETIRED_KEYS
    from body.sim.anatomy import SIM_CFG
    known = set(PHYSIOLOGY) | set(SWITCHES) | set(MOTOR) | set(CEREB) | set(REFLEX) | set(FRAMES) | set(AMYG) | set(SLEEP)
    reads = _reads()
    assert not sorted(k for k in reads if k not in known and k not in INLINE_KEYS), "read but declared nowhere: add to a table or INLINE_KEYS"
    assert not sorted(k for k in INLINE_KEYS if k not in reads or k in known)          # the list holds only keys read inline
    assert not sorted(k for k in RETIRED_KEYS if k in reads or k in known)             # a retired key is read by nothing
    assert not sorted(k for k in SIM_CFG if k not in reads and k not in known)         # every sim constant is read
    print("I2: keys read inline", sorted(INLINE_KEYS), "| retired", RETIRED_KEYS)
