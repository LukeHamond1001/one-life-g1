# The curriculum: the workshops in order

The owner (2026-10-30 night): "i just want to stay organized with workshops. nice so if train bigger model we can just run it though
these in order to speed train". This is the order to take a model through, each workshop's command, its bar, what it took the brain
raised so far, and what went wrong on the way. `docs/LEVELS.md` is the full record of every run; this page is the route through it.

## The rules every workshop keeps

- **The parent only moves its body** (her hands by the gap, her voice, her hand turning its head); nothing is written into the brain
  (no labels, no forced acts, no switch turned off). Architecture changes only to make it more biological (`docs/SIM_DESIGN.md`).
- **Run each workshop to its bar, not to a count.** The passes below are what this brain needed; a bigger or newer model will need
  its own number. Test with her hands off, nothing saved (`workshop13.py`, `test_chain.sh`).
- **Never one thing drilled alone** once there is a chain (from Workshop 26): every toy in turn inside the whole chain (`rounds.sh`).
  Three times a part or one toy practised alone broke what the others share.
- **The gate after every saving run** from Workshop 24 on: the chain test (`test_chain.sh`, 15 independent single passes; decide on
  30, two seed sets: a test of 15 swings about 2 either way). Below 12 of 15: send the brain back to the save before
  (`brain_after_w<N-1>`, the owner's two saves) and change the workshop, not the brain.
- **Read every failure** before changing anything: `pass_digest.py` (each pass's outcome and why, its moments' sheet), the frames, the
  tick records. Every fix so far came from a pass read whole.
- At most two sims at a time on the 16 GB machine. Records go to the Trash once written up.

## The start

The workshops began from NB2's newest save (data/demo/nb2, 2026-10-26: a newborn after its first lived days, `tools/live/start_nb2.sh`),
copied into `data/workshops/brain` by Workshop 1's first run (`--from`). A new or bigger model needs its own first days of life the
same way, or that save.

Every command below runs from the repository root after `. tools/live/nb2_config.sh` (it sets `$CFG`), and saves into
`data/workshops/brain` unless `--no-save`; `--again` for every saving run of a workshop after its first.

## Workshops 1 to 28 (done, 2026-10-27 to 10-30)

| # | workshop | command | its bar | what this brain took |
|---|---|---|---|---|
| 1 | walked into the red circle | `workshop1.py --out DIR --passes N` | her share 0.10 or less on every part | four runs (among them 60, 150 and 300 passes); the arm and stance layers and the eye layer built on the way |
| 2 | a longer way to the circle | `workshop2.py --out DIR --passes N` | her share 0.10 or less over 50 passes, 50 of 50 in | 7 batches of 40 to 60 passes |
| 3-10 | named like a flashcard: ball, cup, duck, car, bear, block, drum, ring | `workshop_name.py N --out DIR --passes 150` | says the word on 80% of the last 50, her share 0.10 or less | 150 passes each; block and others polished with `--again` |
| 11 | picked up off the floor (the block) | `workshop11.py --out DIR --passes N` | its own movement 17 of 20, shares 0.08 or less, no pain, no fall | many dry and saving runs; the movement layer, the muscle lag, the clean handover (A240b) built on the way |
| 12 | a second thing (the cup) | `workshop12.py --out DIR --passes N` | lifted on its own | her plan for the cup's round grasp first |
| 12b | found by its name | `workshop12b.py --out DIR --passes N` | its eyes to the thing named, 14 of 15 | |
| 13 | **the test**: `go pick up the <thing>` | `workshop13.py --out DIR --passes 5 --ask cup` | lifted 5 of 5 a toy, no hands | (a test: never saves) |
| 14 | what things are like ("the cup is round and small") | `workshop14.py --out DIR --passes 70` | its eyes to the toy named on their own | 70 passes, 10 rounds of seven toys |
| 15 | walked to the one she names | `workshop15.py --out DIR --ask TOY` | | stopped, not kept (sent back) |
| 16 | a new thing's name (the can) | `workshop16.py --out DIR --passes 48` | "go pick up the can" 5 of 5, never picked before | **its name alone first** (`workshop1.py --task name --word can`, 8 passes), then described; the other order never took |
| 17 | picked up and looked at | `workshop17.py --out DIR --ask TOY --passes 1` (each toy in turn) | the pick-up ends with the toy held before its eyes | |
| 18 | its own eyes (its seeing from its camera, A265) | `workshop18.py --out DIR --passes N` | finds the named toy with its own eyes (11 of 16 by themselves at first; 16 of 16 after the looks were taught again) | |
| 19 | put it down | `workshop19.py --out DIR --passes 28` | 11 of 11 put down, no hands (`--no-hands`) | |
| 20 | a new thing's name, seen | `workshop20.py --out DIR --passes N` | | |
| 21 | put it in the bucket | `workshop21.py --out DIR --passes N` | | the looks then drifted: taught again (A271, A273) |
| 22 | carried to the bucket | `workshop22.py --out DIR --passes N` | 20 of 20 with her hands | broke its own start of the pick-up: Workshop 23 |
| 23 | the pick-up practised again | `workshop23.py --out DIR --ask cup` | | sent back; the bucket's name taught again instead |
| 24 | pick it up, then put it in the bucket | `workshop24.py --out DIR --ask cup --passes 1` (rounds of the toys) | the chain by itself | sent back once; then rounds of 4 and 5 toys |
| 24a | the requests shown | `workshop24a.py --out DIR --passes 16` | | before Workshop 24 again |
| 25 | carry it to the bucket | `workshop25.py --out DIR --passes 24` | 22 of 24 in | four times (A278 steadiness, A279 her grip teaching its held grip) |
| 26 | hold it while you walk (the bucket 1-2 m on, turned) | `rounds.sh 26 [ROUNDS] [keep\|again] 2` | **the first stage: the chain by itself 12 of 15** | the cerebellum's held-thing module (A283) and fall rule (A284) first; then every toy taught again from her plan in turn (`EXTRA="--her-plan --stop-at 0.355" rounds.sh 26 1 again 2`), a round of its own: **14 of 15, then 12 of 15 on new seeds** |
| 27 | stop where your hand reaches | `workshop27.py` | | both forms tried and not kept (they drifted the stops; drilling the carry alone broke the picks) |
| 28 | stand steady, take a bump | `rounds.sh 28` (`--bumps 20,40`) | no falls, sway under 5 cm | built, not yet run: moved to Workshop 45 |

Between Workshops 26 and 29: A288 (its own arm in front of a thing is no whole sighting) took the car 0 of 3 to 3 of 3; the chain
13 of 15 after it.

## Workshops 29 to 50 (the plan from 2026-10-30 night)

| # | workshop | command | its bar |
|---|---|---|---|
| 29 | look where I point (pictures, its body held still) | `lessons.py --lesson 1 --days 20 --to-bar` | re-finding a thing among four other kinds, 9 of 10 (NB2: at its bar after 24 days, 0.93) |
| 30 | what is it: about 70 words from real photos and the sim's | `lessons.py --lesson 2 --days 40 --to-bar` | "look at the X" among four 8 of 10, naming 7 of 10, real and sim; crossed reported (NB2 after 40 days on 150 photos a word: sim met, 0.96 and 0.90; real not, 0.76 and 0.34; crossed at chance; going on with 600 a word) |
| 31 | its new eyes in the room: the what pathway names what its seeing finds (A290) | `WHAT_EYES=1 tools/levels/test_chain.sh TAG` (seeds 2-4, then 5-7) | the chain 12 of 15 or more, 30 passes |
| 32 | same kind, different looks | (to build) | a never-seen cup, dog, car named, 3 of 4 |
| 33 | what it's like: colours, sizes, shapes | (to build) | "the red one", 8 of 10 |
| 34 | one, two, the other | (to build) | 3 of 4 |
| 35 | where it is: in, on, under, next to, behind | (to build) | 8 of 10 |
| 36 | kinds of kinds | (to build) | a never-seen animal, 3 of 4 |
| 37 | what happens (videos) | (to build) | the event named, 7 of 10 |
| 38 | who did what to what | (to build) | a sentence picks the right video of three, 3 of 4 |
| 39 | say what you see | (to build) | its own descriptions right, 7 of 10 |
| 40 | known tasks in new words | (to build) | 8 of 10 |
| 41 | new combinations of thing, action and place | (to build) | 8 of 10 |
| 42 | watch her do it once, then do it | (to build) | 8 of 10 |
| 43 | watch a video once, then do it | (to build) | 7 of 10 |
| 44 | a new action by watching and her hands | (to build) | then used in told tasks |
| 45 | take a bump, floors and weights varied | `rounds.sh 28` and more | no falls, the chain 12 of 15 or more |
| 46 | realistic rooms and scanned toys inside the chain | (to build: `body/sim/looks.py` is ready) | 4 of 5 unseen rooms |
| 47 | stop, no, wait | (to build) | frozen within 0.5 s, 10 of 10 |
| 48 | look around for it; around, not through | (to build) | 9 of 10 |
| 49 | get up by itself | (to build) | 9 of 10 |
| 50 | tidy up | (to build) | 4 of 5 rooms |

The picture lessons need their pictures first: `photos_fetch.py` (COCO), `sim_pictures.py` (then `--reindex`), `rooms_fetch.py`.
