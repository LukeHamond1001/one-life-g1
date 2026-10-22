> **ARCHIVED 2026-10-21. P1's claim two (memory to act: the hidden toy's ask): read in P1; not pursued further.**

# Pre-registration of claim two (P1, section 6): memory beyond the window

Written and committed 2026-10-20, before any hide is recorded (no smoke test, no generator trial, no student's test). It details
docs/archive/PREREGISTRATION.md section 6 and changes nothing there. Sections 1 to 7 are not edited after the first hide is recorded; amendments
are appended in section 8, dated, with their reason, and a change of what a trial is or how it is scored ends claim two's run.

## 1. The question

On a task whose answer was seen and is now out of view for longer than any 64-tick window holds, does student3 (the pre-registered
student, after claim one's door test) answer from memory, by a clear margin over two behavior-cloning transformers trained on the same
expert recordings: B-64 (window 64 ticks, 9.6 s) and B-long (window 1,024 ticks, 154 s)?

## 2. The task: hide-later (the delayed search)

- **The room**: the home room with a second bucket (extras.add_book_box_buckets: the first on the mat at (-0.3, -0.45), the second
  terracotta at (1.55, 1.05), 2.3 m apart), brought in only when claim two's rung opens. Two places, so chance is 50%.
- **The hide** (tools/harness/demo_tasks.py HideLater, an environment's act, disclosed, as her hide by placement is, C340): the child set down
  1.7 m back from the two buckets' midpoint, facing it (both in its view); a toy it can name, held by no one, shown before its eyes
  (0.45 m out along its head's line, 0.25 m down) for 10 ticks, then carried in an arc over the drawn bucket's rim for 24 ticks and let
  go into it; her line names it as it is shown. A toy that does not land inside is a lost hide, logged, no trial.
- **Which bucket**: the first hide's at random; after it, the same bucket as the last hide or the other, drawn in shuffled blocks of
  two and two (balanced, neither foretold).
- **The delay**, from the drop to her ask, drawn each hide in its band, the bands in shuffled blocks of two and two: *short* 300 to 600
  ticks (45 to 90 s: past B-64's window, inside B-long's), *long* 1,200 to 1,800 ticks (3 to 4.5 minutes: past both). A come runs first
  (she calls the child to her from where she stands), then the day's other tasks (come, look, where, call), none of them about the
  hidden toy, no scatter of it.
- **Her ask**: "where's the X?" (ask_where: no point, no look to it), her name glance off (lane.glance_off: her eyes stay on the
  child as she asks); she names it from her memory of the hide where she cannot see into the bucket from where she stands.
- **The answer is the first choice**: the first bucket the child's pelvis comes within 0.65 m of, or whose hidden toy its eyes hold
  (her reading of its target) 3 ticks running, within 300 ticks of the ask being heard. Met when that bucket is the hidden toy's. No
  choice by 300 ticks: not met.
- **Logged per trial**: the toy, the bucket, same or switched, the band and the delay in ticks, whether the drop's last 7 ticks were
  in the child's view (her percept's child_sees) and in its fovea (her reading of its target), the choice, its tick.

## 3. Training

When claim two's rung opens (after the door test, by a commit that names this document), the generator's slots run with `--claim2`
(the two-bucket room; a hide begun at 30% of the task starts when none is waiting), the expert answering each ask by the walk to the
right bucket and its look into it. The curve's DAgger collections include the hides and the asks (`--claim2`). B-64 and B-long train on
exactly the expert recordings student3 lives, each with its own DAgger (`--claim2`).

## 4. The expert's script

The expert's walk circles on half its where tasks in the claim-one harness (P1 section 9: 10 of 21). Its repair (in the development
tree, measured on the same seeds) lands before claim two's rung opens and is named in section 8; claim two's data and its ceiling are
the repaired expert's. Claim two runs only if the repaired expert meets at least 90% of 50 hide-later trials (generator trials, seeds
named in section 8).

## 5. The tests and the bars

- **The test**: `tools/harness/demo_test.py --claim2`, student3 frozen, seeds 61 and up, run until each band has at least 30 trials whose drop
  was in the child's view (at least half of the drop's last ticks): about 30 short and 30 long trials. B-64 and B-long take the same test
  with the same seeds, each with its own save at the same night.
- **Primary analysis**: trials whose drop was in the child's view, each band apart. **Pass**, in each band: student3's first choice
  right on at least 80% (chance is 50%) and at least 30 percentage points above the better of B-64 and B-long.
- **Reported beside it**: all landed trials (in view or not); the fovea's share; same and switched apart (a child that goes back to
  the last bucket it was rewarded at is right on the same trials and wrong on the switched: A-not-B); the delay's ticks against right and
  wrong. B-long is expected near chance in the long band and may be well above it in the short: the long band is where a window cannot
  explain a win.
- **The bring family** (section 6 of P1) is used only if the repaired expert meets it on 90% of 50 trials before claim two's data
  begins; until then claim two is read on hide-later alone.

## 6. What may change

Instruments (logs, reports). Bugs that leave the task, its scoring and its data as written here, each logged in section 8 with a
test that shows it. Nothing else without an amendment that ends the run.

## 7. Who and when

The lead, 2026-10-20, after Fable's three reviews of the day (the delay past the window and varied; her gaze on the child at the ask;
the hide in view, logged; the ceiling's noise; the bar against two-bucket chance; same and switched; a band past B-long's window).

## 8. Amendments and records

(none)
