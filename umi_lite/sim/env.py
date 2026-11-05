"""MuJoCo pick-and-place with a floating 'UMI gripper' (2 fingers + wrist camera).

The floating gripper is exactly the UMI abstraction: the policy outputs end-effector poses,
whatever robot carries the gripper only needs to track them. Here a mocap body drags the
gripper through a soft weld constraint, so contacts/grasping are still physical.
"""
import os
import sys

# Headless Linux (Colab, servers): render offscreen through EGL on the GPU (fallback: osmesa,
# set MUJOCO_GL=osmesa yourself). Must happen before mujoco is imported.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from umi_lite.geometry import make_pose, rot_x, rot_z

XML = """
<mujoco model="umi_sim">
  <option timestep="0.002" impratio="10" cone="elliptic"/>
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="2048"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".62 .58 .52" rgb2=".52 .48 .42"
             width="512" height="512" mark="edge" markrgb=".4 .4 .4"/>
    <material name="grid" texture="grid" texrepeat="10 10" reflectance="0.05"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".6 .7 .8" rgb2=".2 .2 .25" width="64" height="64"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1" diffuse=".7 .7 .7" castshadow="true"/>
    <light pos="-0.3 0.3 1.2" dir="0.2 -0.2 -1" diffuse=".3 .3 .3" castshadow="false"/>
    <geom name="table" type="plane" size="0.6 0.6 0.01" material="grid"/>
    <camera name="overview" pos="0 -0.62 0.55" xyaxes="1 0 0 0 0.66 0.75" fovy="50"/>
    <body name="target" pos="0.1 0 0.0005">
      <geom type="cylinder" size="0.035 0.0005" rgba="0.1 0.75 0.25 1" contype="0" conaffinity="0"/>
    </body>
    <body name="cube" pos="0 0 0.02">
      <freejoint name="cube"/>
      <geom name="cube" type="box" size="0.02 0.02 0.02" mass="0.05" rgba="0.85 0.15 0.1 1"
            friction="1.5 0.01 0.0002" condim="4"/>
    </body>
    <body name="mocap" mocap="true" pos="0 0 0.2" quat="0 1 0 0"/>
    <!-- hand frame: origin = TCP (between fingertips), +z = approach direction -->
    <body name="hand" pos="0 0 0.2" quat="0 1 0 0" gravcomp="1">
      <freejoint name="hand"/>
      <geom name="palm" type="box" pos="0 0 -0.065" size="0.045 0.018 0.01" mass="0.3" rgba=".15 .15 .17 1"/>
      <geom name="cam_mount" type="box" pos="0 -0.045 -0.08" size="0.015 0.03 0.008" mass="0.05" rgba=".15 .15 .17 1"/>
      <camera name="wrist" pos="0 -0.06 -0.09" xyaxes="1 0 0 0 -1 0" fovy="100"/>
      <body name="finger_l" pos="0 0 -0.025" gravcomp="1">
        <joint name="fl" type="slide" axis="1 0 0" range="0 0.045" damping="5" armature="0.001"/>
        <geom type="box" pos="0.009 0 0" size="0.008 0.012 0.03" mass="0.02" rgba=".85 .85 .2 1"
              friction="1.5 0.01 0.0002" condim="4"/>
      </body>
      <body name="finger_r" pos="0 0 -0.025" gravcomp="1">
        <joint name="fr" type="slide" axis="-1 0 0" range="0 0.045" damping="5" armature="0.001"/>
        <geom type="box" pos="-0.009 0 0" size="0.008 0.012 0.03" mass="0.02" rgba=".2 .4 .85 1"
              friction="1.5 0.01 0.0002" condim="4"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <weld body1="mocap" body2="hand" relpose="0 0 0 1 0 0 0" solref="0.01 1"/>
  </equality>
  <actuator>
    <position name="fl" joint="fl" kp="200" ctrlrange="0 0.045" forcerange="-15 15"/>
    <position name="fr" joint="fr" kp="200" ctrlrange="0 0.045" forcerange="-15 15"/>
  </actuator>
</mujoco>
"""

HOME_Z = 0.20
OPEN = 0.08
WORKSPACE_LO = np.array([-0.3, -0.3, 0.012])
WORKSPACE_HI = np.array([0.3, 0.3, 0.35])


def hand_rot(yaw):
    """gripper pointing down (+z_hand = -z_world), rotated by yaw about world z"""
    return rot_z(yaw) @ rot_x(np.pi)


class PickPlaceEnv:
    def __init__(self, img_res=96, control_hz=10):
        self.m = mujoco.MjModel.from_xml_string(XML)
        self.d = mujoco.MjData(self.m)
        self.n_sub = int(round(1.0 / control_hz / self.m.opt.timestep))
        self.res = img_res
        self.renderer = mujoco.Renderer(self.m, img_res, img_res)
        self._overview = None
        self.hand = self.m.body("hand").id
        self.cube = self.m.body("cube").id
        self.target_body = self.m.body("target").id
        self.q_cube = self.m.jnt_qposadr[self.m.joint("cube").id]
        self.q_hand = self.m.jnt_qposadr[self.m.joint("hand").id]
        self.q_fl = self.m.jnt_qposadr[self.m.joint("fl").id]
        self.q_fr = self.m.jnt_qposadr[self.m.joint("fr").id]

    # ------------------------------------------------------------------ state access
    def hand_pose(self):
        return make_pose(self.d.xpos[self.hand].copy(), self.d.xmat[self.hand].reshape(3, 3).copy())

    def width(self):
        return float(self.d.qpos[self.q_fl] + self.d.qpos[self.q_fr])

    def cube_pos(self):
        return self.d.xpos[self.cube].copy()

    def cube_yaw(self):
        R = self.d.xmat[self.cube].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def target_pos(self):
        return self.m.body_pos[self.target_body].copy()

    def obs(self):
        self.renderer.update_scene(self.d, camera="wrist")
        return {"rgb": self.renderer.render().copy(), "pose": self.hand_pose(), "gripper": self.width()}

    # ------------------------------------------------------------------ control
    def reset(self, seed):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.m, self.d)
        cube = rng.uniform([-0.12, -0.10], [0.12, 0.10])
        while True:
            tgt = rng.uniform([-0.15, -0.12], [0.15, 0.12])
            if np.linalg.norm(tgt - cube) > 0.10:
                break
        self.m.body_pos[self.target_body][:2] = tgt
        yaw = rng.uniform(-np.pi / 4, np.pi / 4)
        self.d.qpos[self.q_cube:self.q_cube + 7] = [*cube, 0.02, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        hp = np.array([*rng.uniform(-0.04, 0.04, 2), HOME_Z])
        hq = np.zeros(4)
        mujoco.mju_mat2Quat(hq, hand_rot(rng.uniform(-0.3, 0.3)).reshape(-1))
        self.d.qpos[self.q_hand:self.q_hand + 7] = [*hp, *hq]
        self.d.mocap_pos[0], self.d.mocap_quat[0] = hp, hq
        self.d.qpos[self.q_fl] = self.d.qpos[self.q_fr] = OPEN / 2
        self.d.ctrl[:] = OPEN / 2
        mujoco.mj_forward(self.m, self.d)
        for _ in range(50):  # settle
            mujoco.mj_step(self.m, self.d)
        return self.obs()

    def step(self, target_pose, gripper_width):
        p = np.clip(target_pose[:3, 3], WORKSPACE_LO, WORKSPACE_HI)
        R = target_pose[:3, :3]
        u, _, vt = np.linalg.svd(R)  # re-orthonormalise
        R = u @ vt
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.reshape(-1))
        self.d.mocap_pos[0], self.d.mocap_quat[0] = p, q
        self.d.ctrl[:] = np.clip(gripper_width, 0, 0.09) / 2
        for _ in range(self.n_sub):
            mujoco.mj_step(self.m, self.d)
        return self.obs()

    def success(self):
        c = self.cube_pos()
        on_target = np.linalg.norm(c[:2] - self.target_pos()[:2]) < 0.035 and c[2] < 0.03
        released = self.d.xpos[self.hand][2] - c[2] > 0.05
        return bool(on_target and released)

    def render_overview(self, size=320):
        if self._overview is None:
            self._overview = mujoco.Renderer(self.m, size, size)
        self._overview.update_scene(self.d, camera="overview")
        return self._overview.render().copy()
