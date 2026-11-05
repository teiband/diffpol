# configs

Small, hand-made files that are expensive to recreate and therefore versioned in git:

- `camera.json` — camera intrinsics from `python -m umi_lite.capture.calibrate --out configs/camera.json`
  (one file per camera/resolution, e.g. `camera_c920_1280x720.json`)
- `gripper.json` — finger-marker distance → width mapping from
  `python -m umi_lite.tracking.track_video --calib-gripper ...` (default output path)

Large files (datasets, checkpoints, videos) go to the data store instead, see `umi_lite/paths.py`.
