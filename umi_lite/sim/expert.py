"""Scripted demonstrator (stands in for the human with the handheld gripper)."""
import numpy as np

from umi_lite.geometry import make_pose
from umi_lite.sim.env import HOME_Z, OPEN, hand_rot

CLOSED = 0.0
GRASP_Z, CARRY_Z, PLACE_Z = 0.02, 0.12, 0.035
HOVER_Z = 0.10
EMPTY_W = 0.015       # measured width below this while "holding" = nothing between the fingers (missed / dropped)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ScriptedExpert:
    def __init__(self, env, rng, miss_offset=None, max_retries=3):
        """miss_offset: optional (dx, dy, dz) [m] added to the first grasp attempt (to create a miss on purpose,
        used for re-grasp demos). The expert re-grasps after a missed or dropped cube (width < EMPTY_W)."""
        self.env = env
        self.v = rng.uniform(0.015, 0.025)       # m per control step (0.15-0.25 m/s)
        self.w = 0.12                             # rad per control step
        T = env.hand_pose()
        self.pos = T[:3, 3].copy()
        self.yaw = float(np.arctan2(T[1, 0], T[0, 0]))
        self.grip = OPEN
        self.phase, self.hold, self.done = 0, 0, False
        self.grasp_yaw = None
        self.miss_offset = None if miss_offset is None else np.asarray(miss_offset, float)
        self.retreat = False          # lifting straight up with open fingers after a miss
        self.retries, self.max_retries = 0, max_retries
        self.missed = False           # becomes True at the step a miss/drop is detected (for recording)

    def perturb(self, dxyz, dyaw):
        """shift the commanded pose (DART-style disturbance); the expert then corrects back"""
        self.pos = self.pos + np.asarray(dxyz, float)
        self.pos[2] = max(self.pos[2], GRASP_Z)
        self.yaw += dyaw
        return make_pose(self.pos, hand_rot(self.yaw)), self.grip

    def _goal(self):
        c, t = self.env.cube_pos(), self.env.target_pos()
        if self.grasp_yaw is None:  # cube has 90 deg symmetry -> closest equivalent yaw
            cands = self.env.cube_yaw() + np.arange(-2, 3) * np.pi / 2
            self.grasp_yaw = float(cands[np.argmin(np.abs(wrap(cands - self.yaw)))])
        g = self.grasp_yaw
        gxy, gz = c[:2], GRASP_Z
        if self.miss_offset is not None:
            gxy, gz = c[:2] + self.miss_offset[:2], GRASP_Z + self.miss_offset[2]
        return [  # (xy, z, yaw, gripper, hold_steps)
            (gxy, HOVER_Z, g, OPEN, 1),           # align xy + yaw above the cube, then descend vertically
            (gxy, gz, g, OPEN, 0),
            (gxy, gz, g, CLOSED, 6),
            (c[:2], CARRY_Z, g, CLOSED, 0),
            (t[:2], CARRY_Z, g, CLOSED, 0),
            (t[:2], PLACE_Z, g, CLOSED, 0),
            (t[:2], PLACE_Z, g, OPEN, 5),
            (t[:2], HOME_Z - 0.06, g, OPEN, 0),
        ]

    def act(self):
        self.missed = False
        if 3 <= self.phase <= 5 and self.env.width() < EMPTY_W and self.retries < self.max_retries:
            # fingers closed on nothing: missed grasp or dropped cube -> open, lift, re-grasp
            self.missed, self.retreat = True, True
            self.retries += 1
            self.phase, self.hold, self.grasp_yaw, self.miss_offset = 0, 0, None, None
        if self.retreat:
            xy, z, yaw, grip, hold = self.pos[:2].copy(), HOVER_Z, self.yaw, OPEN, 0
        else:
            xy, z, yaw, grip, hold = self._goal()[self.phase]
        goal = np.array([xy[0], xy[1], z])
        d = goal - self.pos
        n = np.linalg.norm(d)
        self.pos = goal.copy() if n <= self.v else self.pos + d / n * self.v
        dy = wrap(yaw - self.yaw)
        self.yaw += float(np.clip(dy, -self.w, self.w))
        self.grip = grip
        reached = n <= self.v and abs(dy) <= self.w
        if reached and self.retreat:
            self.retreat = False
        elif reached:
            goals = self._goal()
            self.hold += 1
            if self.hold > hold:
                self.hold = 0
                self.phase += 1
                if self.phase == len(goals):
                    self.done = True
                    self.phase -= 1
        return make_pose(self.pos, hand_rot(self.yaw)), self.grip
