"""Pose helpers (numpy). Poses are 4x4 homogeneous matrices; the policy uses a 9-D
vector per pose: xyz + 6-D rotation (first two columns of R, Zhou et al. 2019)."""
import numpy as np


def make_pose(pos, R):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def inv_pose(T):
    """Inverse of one or many (…,4,4) rigid transforms."""
    R = T[..., :3, :3]
    p = T[..., :3, 3]
    Ti = np.zeros_like(T)
    Rt = np.swapaxes(R, -1, -2)
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = -(Rt @ p[..., None])[..., 0]
    Ti[..., 3, 3] = 1.0
    return Ti


def mat_to_rot6d(R):
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True).clip(1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # vectors are the columns


def pose_to_vec9(T):
    return np.concatenate([T[..., :3, 3], mat_to_rot6d(T[..., :3, :3])], axis=-1)


def vec9_to_pose(v):
    shape = v.shape[:-1]
    T = np.zeros(shape + (4, 4), dtype=np.float64)
    T[..., :3, :3] = rot6d_to_mat(v[..., 3:9])
    T[..., :3, 3] = v[..., :3]
    T[..., 3, 3] = 1.0
    return T


def relative_poses(T_ref, T_seq):
    """Express poses T_seq (N,4,4) in the frame of T_ref (4,4) -> (N,4,4).
    This is UMI's 'relative trajectory' representation: no robot/world calibration needed."""
    return inv_pose(T_ref)[None] @ T_seq


def rodrigues_to_mat(rvec):
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    th = np.linalg.norm(rvec)
    if th < 1e-12:
        return np.eye(3)
    k = rvec / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])
