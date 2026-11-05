# Hardware list (webcam-based UMI, no robot needed)

Prices are rough 2026 street prices and only meant for orientation.

## Required

| # | Item | Notes | ~Price |
|---|------|-------|--------|
| 1 | **Wide-angle USB webcam, fixed focus** | The most important part. Wanted: wide FOV (≥ 120°, ideally a 150–180° M12 fisheye lens), ≥ 60 fps at 720p (MJPG), manual exposure, **no autofocus** (or autofocus that can be locked). Ideally **global shutter** (fast hand motions + rolling shutter distort marker corners). Examples: Arducam/ELP UVC boards with M12 fisheye lens; global-shutter colour variants (OV9782/AR0234 sensors) are best. A Logitech C920-class webcam works for a first test but has a narrow FOV (~78°) and loses the board easily. | 30–90 € |
| 2 | **Handheld gripper** | Either the open-source UMI gripper (3D-printed; STL/BOM in the UMI repo) or a simple **pick-up / reacher grabber tool** with two flat jaws. | 15 € (grabber) / ~50 € (printed UMI) |
| 3 | **Camera mount** | 3D-printed bracket or a small clamp + ball head; camera rigidly fixed above the fingers, looking along the approach direction, fingertips visible at the bottom of the image. **Must not wobble** - any play between camera and gripper is tracking error. | 0–15 € |
| 4 | **Printed markers** | `python -m umi_lite.capture.make_markers`: world board (A4, laminate it and tape it flat to the table), two 15 mm finger markers (glue onto the jaw sides facing the camera), ChArUco calibration board. | ~5 € |
| 5 | **Rigid flat backing** for the ChArUco board | Foam board / acrylic / aluminium composite. A bent calibration board ruins the intrinsics. | 5 € |
| 6 | **USB extension cable** 2–3 m | Lets you move freely; active cable if longer than 3 m. | 10 € |
| 7 | **Computer** | Any laptop for recording. Training: a GPU with ≥ 2 GB VRAM works with the defaults (e.g. GeForce MX250 → slow but OK), 6–8 GB is comfortable. CPU-only works with `--encoder small`. Or use Google Colab / Kaggle free GPUs for training. | - |
| 8 | **Even lighting** | Desk lamp / LED panel. Bright light lets you use short exposures → less motion blur. | 0–25 € |
| 9 | **Ruler or caliper** | Measure printed marker sizes, update `umi_lite/markers.py`. | 5 € |

## Optional

| Item | Why |
|------|-----|
| **IMU** (BNO085 / ICM-20948 breakout + ESP32/RP2040, or a phone strapped to the gripper) | Only needed if you switch to markerless visual-inertial SLAM (large workspaces, no table board). Must be time-synchronised with the camera. See README → "Is an IMU needed?" |
| **GoPro Hero 9-13 + Max Lens Mod** | The original UMI camera: 155° FOV, built-in IMU, time-synced telemetry. Lets you use the official UMI pipeline (ORB-SLAM3) instead of markers. |
| Second ArUco board / more boards | Larger workspace: the camera must always see ≥ 2 board markers. |
| Soft finger pads (TPU print / silicone) | Better grasps, like UMI's compliant fingers. |
| Low-cost arm later (e.g. SO-100/SO-101, ~200-300 € DIY kit; or a used UR/Franka at a lab) | For real deployment. The relative-pose actions transfer to any arm that has IK + a camera mounted at the same place relative to the gripper. |
