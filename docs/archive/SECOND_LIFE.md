> **ARCHIVED 2026-10-21. The second life's design notes.**

# The second life (branch alive, born 2026-10-19 02:52)

A newborn of the same architecture as Pip, at the same size (d 512, seed 1: Pip's twin at birth), raised from birth on generated
experience and then living, one timeline, never reset. Pip is paused at dawn 207 and stays the yardstick: every ruler of the second life
is read day for day against Pip's same day (`p1_tools/life/probes/alive_rulers.py D`).

**The owner's goal:** train it until it understands enough of its world that reinforcement learning on it comes as easily as it does
for a person, who knows the next thing to do in an environment; and make sure the reinforcement-learning architecture is what the data
imprints, so the model can always go on learning from reward.

## The body at birth (the decision log's entries)

| entry | what | why |
|---|---|---|
| A238 | Breath retired: the tract's reservoir never empties; the born breath is a steady supply (the lungs driven always, never run dry) | the owner's word: no breath restraint, now or later |
| A239 | The scaffolds off: the born approach (the guide), the gait's catch on a fall, her lures | the owner's word: the experiences supply what they forced |
| A239b | In pretraining Unitree's gait keeps the legs (its share does not fade) | the first command block faded it to 0 in 3,000 ticks; the newborn's layer then fell 10 times |
| A185b | The wrists in the resting tone (their middle, as the shoulders and elbows already were) | day 0's floor play: wrist pain on 12 to 14% of ticks, the child's own acts stepping a wrist into the stop it stood at |
| A238b | The voice's power on demand: the lungs driven on a tick the voice acts, drawn back when it rests (day 2) | A238's steady air hissed through the open glottis on 86% of ticks; her ear could not find the child's turns |
| A229z | The inverse models read their input at its own scale, once, over their first 2,048 own acts (day 2) | the walk command's inverse read the raw numbers (A229y) and never learned (kappa 0.01): the shown walks were never labelled as the acts they equal |

## The experience generator (the data's side, never the body's)

Each hand pushes an effector in the brain's own act vocabulary, summed with the brain's own act, so the brain keeps choosing (on-policy)
and its inverse model reads the move as a demonstration (A228 to A235). Her approval and its pain come from the world as in life. When
pretraining ends the hands are off and nothing inside the brain changes.

| entry | hand | shown and tried |
|---|---|---|
| E1, E1b | The walk to her in the come lesson, once she has called, through the walk command's own settings (Unitree's gait carries it) | the first command block of every two shown, the next tried |
| E2 | The reach to a toy within an arm's reach, from above, the hand opened on the way; the palm's touch fires the born grasp | one reach of every two shown |
| E3 | Talk: the sibling asks her about a toy she can name, or names the one in its hand; she answers in her turn, her look on it | (overheard: no act of the child's is shown) |

**Shown and tried** is the safeguard the owner asked for: a tried block is the life without the generator, so its comings and reaches
are the brain's own acts paid by her face alone, read every day. If pretraining were imprinting anything the reward's road could not
build on, the tried blocks would show it first.

## The rulers

- `p1_tools/life/probes/alive_rulers.py D`: her judgments by kind, pain, crying and stress, E1's shown and tried blocks with their comings, E2's
  reaches and grasps and the child's own, E3's talk; against Pip's day D.
- `p1_tools/life/probes/rl_ease.py`: how fast a copy learns a contingency nothing in its life has paid (its right hand raised, calibrated to its own
  arm; her smile at each onset) against a control copy. The owner's goal as a number: the paid trials a copy needs to double the act.
  Retired as the daily ruler on day 2 (`p1_tools/life/probes/rl_diag.py`, the tick-105,000 pair): the raise is no choice of the newborn's. Its
  shoulder pitch's commonest setting (two thirds of its draws, a small step the resting tone cancels at its equilibrium) moved the palm
  0.1 mm a tick; the raises came from posture and the other joints, and her smile's credit went to that setting (178 of the 206 acts
  in the ten ticks before a paid raise). Paid 21 raises against 19 unpaid.
- `p1_tools/life/probes/rl_ease2.py` (the daily ruler from day 2): a contingency one choice of the child's makes in a tick, paid by her smile, on a
  copy in the quiet room (MUTE: her other judgments smile no more after the calibration, the lab of the infant studies), against the
  same copy unpaid. KIND=voice: the child's voice acting after six ticks of silence (Rheingold, Gewirtz and Ross 1959: 3-month-olds'
  vocalizing rose under an adult's smile given at each one). KIND=act: one setting of one joint drawn by the effector's own choice
  (the right shoulder pitch's big step down, about 1% of its draws: the mobile's kick, Rovee-Collier). The number: the voice's acting
  time in the last thousand ticks over the first, paid against unpaid. Day 2 (tick-105,000 pair, 3,000 ticks): paid x1.46 (192 to
  281 ticks of a thousand), unpaid x1.20 (163 to 195): net x1.22, the voice acting 706 ticks against 546. The reward's road to the
  policy works on the newborn; slowly: her face held the last smile at 93 of 138 paid bursts (one smile in some 35 ticks at most).
  A paid burst is answered later than a person's approval would be, and in the life her other judgments' smiles (one every 35 ticks
  on day 2) and the curiosity rewards (12% of ticks) dilute a contingency's credit further.

## The days

Read by `p1_tools/life/probes/alive_rulers.py D` over the day's 24,000 waking ticks (Pip's same day in brackets where it had the measure).

| day | pain | her judgments (most) | E1 shown / tried comings | E2 reaches / grasps / own grasps | E3 talk | changes landed |
|---|---|---|---|---|---|---|
| 0 | 5.09% (Pip 0.00) | echo 149, right_name 146 (Pip 42), reach_nearer 109, sat 87, stood 82, stepped 67 | 11 in 2 blocks / 1 in 1 | (not yet) / own 18 | (not yet) | born 02:52 under A238, A239, E1 |
| 1 | 0.21% | word 463, lifted 83, reach_nearer 65, right_name 63, echo 49, got 45 | 6 in 1 / 0 in 1 | 30 / 9 / 33 | 19 lines | dawn 1: A185b, A239b, E1b, E2, E3 |
| 2 | 0.15% (Pip 0.00) | word 273, right_name 120 (Pip 74), got 44, reach_nearer 41, lifted 38, stepped 38 | 4 in 1 / 0 in 2 | 25 / 24 / 25 | 47 lines | dawn 2: E2b, E3b, R12; mid-day: C376, C377 (97,000), A238b (104,000), R12b (115,000) |
| 3 | 0.69% (Pip 4.24) | word 323, right_name 56 (Pip 18), lifted 40, stepped 37, hit 36, got 36 | 4 in 2 / 0 in 1 | 23 / 26 / 20 | 41 lines | dawn 3: A229z, A185b's tests; mid-day: C378 (162,000) |

Day 3's: the voice's contingency net x1.31 (paid 239 to 282 acting ticks a thousand, unpaid 241 to 216). Every inverse but the walk command's read its scale (A229z) by tick 153,000, the walk command's at 1,758 of its 2,048 pairs at day 3's first command block's end. The forecast's errors fell again (body 0.034 to 0.031, ears 0.069 to 0.046, face 0.026 to 0.019, touch 0.024 to 0.022, vestibular 6.7 to 5.0, the walk command 0.0030 to 0.0022; words 0.88 flat). The pain was the right wrist pitch's, nearly all in ticks 147,000 to 150,000 (A185c tried for it and not landed).

Day 2's other rulers. The voice's contingency in the quiet room (rl_ease2, above): net x1.22 in 3,000 ticks. The walk command's inverse
read at chance (kappa 0.01 to 0.02, against the arms' 0.02 to 0.5 by day 2's end): A229z, landing at dawn 3, is for it. Pain fell from
5% to 0.2% with A185b; the right names her ear judged rose 63 to 120; her ear heard the child's turns once the voice rested (A238b).
