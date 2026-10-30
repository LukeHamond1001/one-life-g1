"""THE ARM LAYER (2026-10-28, the owner: "doesnt human have layer like that for every phyiscal movment", then "build whatever biology
got"): A LEARNED CARRIAGE OF THE ARMS AND THE HANDS IN A WALK, BELOW THE BRAIN, the arms' counterpart of the leg layer (body/sim/legs.py).

What it is. In a walk (the gait walking the body: the ghost's walk mode, the pose not held) the arms and the hands are carried by this
layer, not by the brain's acts, as the legs' stepping is the leg layer's: a walking child's arm carriage and swing come from the
spinal and brainstem circuits that pattern the walk, coupled to the legs' cycle (interlimb coordination: Dietz 2002, Zehr and Duysens
2004; the new walker's arms held in high guard, Ledebt 2000), and the brain's acts on them are out of its hands for the walk (A231:
their `_engaged` flag shows 0, so its own act is its rest and its gate draws nothing). What it holds is a target angle per joint over
the gait's cycle: a posture and two harmonics of the cycle's phase (the swing a gait may carry), target_j = w_j . [1, sin p, cos p,
sin 2p, cos 2p]. The cord's reflexes (the grasp, the traction response) still add their steps; the resting tone does not (in a walk the
layer's posture is the arms' set point). Outside a walk the arms are the brain's as before.

How it learns. From the posture the arms are held toward in a walk: each tick a hand moves them (`lesson`: her hands in a workshop, the
target they push toward, at the share they push with), the layer steps toward that target on the cycle's phase (least mean squares,
normalized by the features' power, RATE a tick). Held in a pose, it comes to hold that pose; swung, it would swing. Nothing else
teaches it: in a life no hand carries the arms in a walk and the layer keeps what it has. Born at the resting tone's angles
(reflexes.TONE_REST; a joint without one, the hands' fingers, at 0), no swing.

THE STANCE LAYER (10-28, the same law, STANCE_LIMBS): while it is held standing (the pose held, `ghost.holds_pose`) the legs and the
waist are carried by a second layer of this kind, at the cycle's phase 0 (a stance, no cycle): born at its standing reflex's stance
angles (reflexes.STANCE: the knees 0.10, against the gait's crouch's 0.30), taught by the stance a hand holds it in; while it carries
them its standing reflex and its postural tone give way (the layer's stance its set point), and the brain's acts on them are out of its
hands as under the pose (A231). Let go of her lift, lead and righting but her pose kept on its legs (`--let-go lift`, 10-28 02:5x) it
stood (the trunk within 2.6 deg); its own standing reflexes at their own stance, let go of all of it (`--let-go stand`), it folded
12 to 26 deg and hurt its waist: a stance, not strength, was what it lacked.

The numbers. RATE is ours (a posture held still taught in about 50 ticks, two or three passes of Workshop 1's walk); the two
harmonics are ours (a walk's arm swing is mostly its first). Saved with the body (beside the leg layer's state)."""
import math

import numpy as np

LIMBS = ("arm_l", "arm_r", "hand_l", "hand_r")   # the effectors the layer carries in a walk
STANCE_LIMBS = ("leg_l", "leg_r", "waist")    # THE STANCE LAYER (10-28, the same law): the legs and the waist while it is held standing
N_F = 5                                       # its features: 1, sin p, cos p, sin 2p, cos 2p
RATE = 0.06                                   # a lesson's step, normalized least mean squares over the five features (ours: the
                                              # posture's own share of it a third, so a held pose is learned to 1/e in about 50 lessons)


def features(phase):
    """the cycle's features at a phase (0 to 1 over the gait's period)"""
    p = 2.0 * math.pi * float(phase)
    return np.array([1.0, math.sin(p), math.cos(p), math.sin(2.0 * p), math.cos(2.0 * p)])


class ArmLayer:
    def __init__(self, joints, born, limbs=LIMBS):
        """joints: {limb: [joint names in the limb's order]}; born: {joint name: its born angle} (a joint absent: 0); limbs: the limbs it
        carries (the arm layer's, LIMBS; the stance layer's, STANCE_LIMBS)"""
        self.joints = {k_: list(v_) for k_, v_ in joints.items() if k_ in limbs}
        self.w = {}
        for limb, js in self.joints.items():
            w_ = np.zeros((len(js), N_F))
            w_[:, 0] = [float(born.get(j_, 0.0)) for j_ in js]
            self.w[limb] = w_
        self.lessons = 0

    def target(self, limb, phase):
        """the layer's target angles for a limb's joints at the cycle's phase"""
        return self.w[limb] @ features(phase)

    def lesson(self, limb, phase, target, share=1.0):
        """a hand moved the limb toward `target` (its joints' angles) at the cycle's phase with `share` of its push (0 to 1): the layer
        steps toward it, normalized least mean squares at RATE x share -> the error before the step (RMS, rad)"""
        f = features(phase)
        err = np.asarray(target, float) - self.w[limb] @ f
        self.w[limb] += (RATE * float(max(0.0, min(1.0, share))) / float(f @ f)) * np.outer(err, f)
        self.lessons += 1
        return float(np.sqrt(np.mean(err ** 2)))

    def state(self):
        return {"w": {k_: v_.tolist() for k_, v_ in self.w.items()}, "lessons": int(self.lessons)}

    def load_state(self, st):
        for k_, v_ in (st.get("w") or {}).items():
            if k_ in self.w and np.asarray(v_).shape == self.w[k_].shape:
                self.w[k_] = np.asarray(v_, float).copy()
        self.lessons = int(st.get("lessons", 0))


def stance_born(joints, stance):
    """the stance layer's born angles: each leg joint's at the standing reflex's stance angle (reflexes.STANCE, by the joint's kind:
    hip_pitch, knee, ankle_pitch), the rest 0 -> {joint name: angle}"""
    return {j_: next((float(v_) for k_, v_ in stance.items() if j_.endswith(k_ + "_joint")), 0.0) for js in joints.values() for j_ in js}

