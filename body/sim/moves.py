"""THE MOVEMENT LAYER (2026-10-28, the owner: "doesnt human have layer like that for every phyiscal movment" (02:xx), then, Workshop 11's
pick-up still all her hands: "i thought we already did this for every physical movment", "build it then get workshop 11 done"): A
LEARNED MOVEMENT BELOW THE BRAIN, ANY MOVEMENT A HAND REPEATS. The general form of the arm and stance layers (body/sim/arms.py), which
each hold one movement on one clock (the walk's cycle, a still stance).

What it is. A child's new movements (to squat and pick a toy off the floor, to climb a step) are learned by repetition into motor
programs below its deliberation: sequences of postures run off as a whole once started, the cerebellum and the motor cortex storing
the sequence and the spinal circuits running it (Wolpert, Diedrichsen and Flanagan 2011; Schmidt's motor programs, 1975; a toddler's
squat learned by practice, Adolph and Berger 2006). Here each movement is a PROGRAM: a target angle per joint over the movement's
phase (0 at its start, 1 at its end), held as weights on N_B smooth bumps over the phase (normalized Gaussians: a sequence of
postures, each bump a moment of it). While a movement runs (`start`, then `advance` a tick), the layer carries every limb the program
holds: the brain's acts on them out of its hands (A231), the cord's reflexes still added (the grasp closing on a pressed palm, the
tendon organ letting a loaded joint go), the resting tone and the standing reflexes giving way, as under the stance and arm layers.
Its phase advances by itself, one tick of its length at a time: once started, the movement runs off.

How it learns. From a hand moving the body through the movement (`lesson`: her hands in a workshop, the pose they push toward at the
phase, at the share they push with): the program's bumps at that phase step toward that pose (least mean squares normalized by the
bumps' power, RATE x the share). A movement never shown is born at the pose the body stands in when it first starts: until taught it
holds still. Each movement its own program (by name): learning one leaves the others as they were.

ITS POSE AND ITS DRIVE (10-28, Workshop 11). A program keeps two things: the pose a hand showed it (`pose`: the movement's shape, its
kinematics) and its command (`target`: that pose and the extra drive its joints need to reach it under the body's weight, the dynamics;
the cerebellum's internal model of a movement's forces beside its plan: Wolpert, Miall and Kawato 1998; Shadmehr and Mussa-Ivaldi 1994).
A hand placing the limbs teaches both; a hand pushing a limb back up where it sags (`lesson(..., drive=True)`) teaches the command alone,
so the pose stays the one shown (10-28: one set of weights for both, the pose its own squat was held to moved as it learned to lead its
sag: its hips 0.5 rad further in 16 passes, its weight 10 cm off its feet); the drive stays within DRIVE_MAX of the pose.

The numbers. N_B, the bumps' width and RATE are ours (a movement a whole hand repeats is learned to 1/e in about 4 passes).
Saved with the body (beside the leg layer's state)."""
import math

import numpy as np

N_B = 30                                      # the bumps over a program's phase (ours: a 60-tick movement, a bump every 2 ticks)
WIDTH = 0.8                                   # a bump's width, in the spacing of their centres (ours: neighbours overlapping)
RATE = 0.12                                   # a lesson's step, normalized least mean squares over the bumps (ours)
DRIVE_MAX = 0.15                              # rad: its drive (its command off its pose) at most, a joint (ours; 10-28, a squat failing for
                                              # its balance, not its joints' sag, wound the drive on pass after pass, 0.03 to 0.27 RMS)
CENTRES = np.linspace(0.0, 1.0, N_B)


def features(phase):
    """the bumps at a phase (0 to 1): normalized Gaussians, summing to 1"""
    s_ = WIDTH / float(N_B - 1)
    g_ = np.exp(-0.5 * ((float(min(1.0, max(0.0, phase))) - CENTRES) / s_) ** 2)
    return g_ / float(g_.sum())


class MoveLayer:
    def __init__(self, joints):
        """joints: {limb: [joint names in the limb's order]}: the limbs a program may hold"""
        self.joints = {k_: list(v_) for k_, v_ in joints.items()}
        self.programs = {}                    # name -> dict(w={limb: [n_joints, N_B]}, T=ticks, lessons=n, runs=n)
        self.running = None                   # a movement under way: dict(name, k, T)

    def known(self, name):
        return name in self.programs

    def start(self, name, T, pose):
        """the movement `name` begins this tick, T ticks long; a movement never shown is born holding `pose` ({limb: its joints'
        angles}, the pose it starts from) at every phase -> its program"""
        if name not in self.programs:
            w_ = {l_: np.repeat(np.asarray(pose[l_], float)[:, None], N_B, axis=1) for l_ in self.joints if l_ in pose}
            self.programs[name] = dict(w=w_, pose={l_: v_.copy() for l_, v_ in w_.items()}, T=int(T), lessons=0, runs=0)
        pr_ = self.programs[name]
        pr_["runs"] += 1
        self.running = dict(name=name, k=0, T=int(T))
        return pr_

    def variant(self, name, thing, like=None):
        """A249 (10-29): THE MOVEMENT FOR THIS THING: the program `name` as its own for the thing `thing` (`name:thing`), born the first
        time it is wanted as a copy of `name` (its pose, its command, its runs), its own after: a grasp learned on a cube does not hold a
        cup, and the same movement shaped by the cup's lessons missed the cube (motor memories keyed to the object, its identity recalling
        how to grip and lift it: Gordon, Westling, Cole and Johansson 1993) -> the variant's name (`name` itself for no thing, or no
        program `name` yet). A256: born from the movement for the thing `like` it is like (`name:like`), when there is one"""
        if thing is None or name not in self.programs or ":" in name:
            return name
        v_ = f"{name}:{thing}"
        if v_ not in self.programs:
            pr_ = self.programs.get(f"{name}:{like}", self.programs[name]) if like is not None else self.programs[name]
            self.programs[v_] = dict(w={l_: a_.copy() for l_, a_ in pr_["w"].items()}, pose={l_: a_.copy() for l_, a_ in pr_["pose"].items()},
                                     T=int(pr_["T"]), lessons=int(pr_["lessons"]), runs=int(pr_["runs"]))
        return v_

    def phase(self):
        """the running movement's phase (0 to 1), or None"""
        r_ = self.running
        return None if r_ is None else min(1.0, r_["k"] / float(max(1, r_["T"])))

    def limbs(self):
        """the limbs the running movement holds (none when none runs)"""
        r_ = self.running
        return () if r_ is None else tuple(self.programs[r_["name"]]["w"].keys())

    def target(self, limb, phase=None, name=None):
        """the program's target angles for a limb's joints at a phase (the running movement's, its phase now, unless given)"""
        nm_ = name if name is not None else self.running["name"]
        ph_ = self.phase() if phase is None else phase
        return self.programs[nm_]["w"][limb] @ features(ph_)

    def pose(self, limb, phase=None, name=None):
        """the pose a hand showed for a limb's joints at a phase (the running movement's, its phase now, unless given): its command
        less the drive it learned"""
        nm_ = name if name is not None else self.running["name"]
        ph_ = self.phase() if phase is None else phase
        return self.programs[nm_]["pose"][limb] @ features(ph_)

    def advance(self):
        """a tick of the running movement gone by; at its end it stops -> whether it still runs"""
        r_ = self.running
        if r_ is None:
            return False
        r_["k"] += 1
        if r_["k"] > r_["T"]:
            self.running = None
            return False
        return True

    def stop(self):
        self.running = None

    def lesson(self, limb, target, share=1.0, phase=None, name=None, drive=False):
        """a hand moved the limb toward `target` (its joints' angles) at the movement's phase with `share` of its push (0 to 1): the
        program's command there steps toward it, normalized least mean squares at RATE x share, and its pose too unless the hand only
        pushed against its sag (`drive`) -> the command's error before the step (RMS, rad)"""
        nm_ = name if name is not None else self.running["name"]
        ph_ = self.phase() if phase is None else phase
        pr_ = self.programs[nm_]
        f = features(ph_)
        k_ = RATE * float(max(0.0, min(1.0, share))) / float(f @ f)
        tgt_ = np.asarray(target, float)
        if drive:                                                       # (a push on its sag: its command no further than DRIVE_MAX off its
            po_ = pr_["pose"][limb] @ f                                 # pose)
            tgt_ = np.clip(tgt_, po_ - DRIVE_MAX, po_ + DRIVE_MAX)
        err = tgt_ - pr_["w"][limb] @ f
        pr_["w"][limb] += k_ * np.outer(err, f)
        if not drive:
            pr_["pose"][limb] += k_ * np.outer(np.asarray(target, float) - pr_["pose"][limb] @ f, f)
        pr_["lessons"] += 1
        return float(np.sqrt(np.mean(err ** 2)))

    def state(self):
        return {"programs": {n_: dict(w={l_: np.asarray(v_).tolist() for l_, v_ in p_["w"].items()},
                                      pose={l_: np.asarray(v_).tolist() for l_, v_ in p_["pose"].items()}, T=int(p_["T"]),
                                      lessons=int(p_["lessons"]), runs=int(p_["runs"])) for n_, p_ in self.programs.items()}}

    def load_state(self, st):
        self.programs = {}
        for n_, p_ in ((st or {}).get("programs") or {}).items():
            w_ = {l_: np.asarray(v_, float) for l_, v_ in (p_.get("w") or {}).items() if l_ in self.joints}
            po_ = {l_: np.asarray(v_, float) for l_, v_ in (p_.get("pose") or {}).items() if l_ in self.joints}
            po_ = {l_: po_.get(l_, v_.copy()) for l_, v_ in w_.items()}      # (a program saved before its pose was kept: its command)
            if all(v_.shape == (len(self.joints[l_]), N_B) for d_ in (w_, po_) for l_, v_ in d_.items()):
                self.programs[n_] = dict(w=w_, pose=po_, T=int(p_.get("T", 1)), lessons=int(p_.get("lessons", 0)), runs=int(p_.get("runs", 0)))
        self.running = None
