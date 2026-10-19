"""I5 (2026-10-17, the owner's word: 'bug check absolutely everything'): NO ERROR SWALLOWED UNSEEN. A handler that catches an error
and carries on (a best-effort read, a fallback, a lesson skipped this tick) calls `swallowed(site, err)`: the first failure at a site,
and every EVERY-th after it, is printed to the run log as 'SWALLOWED (I5) at <site> (#n): <error>' (the bug stalker reads it), the
count kept per site (COUNT, an instrument; not saved). The handler's fallback is unchanged: a site that never fails prints nothing and
changes nothing, so no digest moves. Until I5, 32 such handlers in the life's path (the come window's, the waking cortex's lesson's,
each effector's gate lesson's among them) failed without a word"""
import collections

EVERY = 1000
COUNT = collections.Counter()


def swallowed(site, err):
    COUNT[site] += 1
    n = COUNT[site]
    if n == 1 or n % EVERY == 0:
        print(f"SWALLOWED (I5) at {site} (#{n}): {type(err).__name__}: {str(err)[:160]}", flush=True)
