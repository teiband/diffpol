"""umi_lite - a minimal UMI-style data-collection + diffusion-policy training framework.

Pipeline:
    webcam on handheld gripper  ->  capture/record.py
    ArUco table board + finger markers  ->  tracking/track_video.py  (6-DoF pose + gripper width)
    episodes (.npz)  ->  data/build_dataset.py
    diffusion policy  ->  train.py
    evaluation  ->  eval_offline.py (real data) / sim/eval_policy.py (MuJoCo, closed loop)
"""
