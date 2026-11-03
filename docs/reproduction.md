# Reproduction and laptop migration

Application setup, external GraspNet setup, and physical robot qualification are separate steps. Saved calibration and poses describe Joel's hardware.

## Application environment

Target: Ubuntu 22.04, Python 3.10, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/joeltang17/piper-yolo-grasp-project.git
cd piper-yolo-grasp-project
./scripts/setup_env.sh
./scripts/run_doctor.sh
```

Rebuild the application `.venv` on the new machine. The distribution/package names remain `piper-vision-grasp-new` / `piper_app` for compatibility. Package constraints are not a fully locked environment; record actual dependency versions with published evaluations.

## Assets a clone does not supply

| Asset | Existing location/configuration | New-machine action |
| --- | --- | --- |
| YOLOE checkpoint | `models/yoloe-26s-seg.pt` | Obtain from its upstream or separate backup. |
| YOLO-World checkpoint | `models/yolov8s-worldv2.pt` | Needed only for that workflow. |
| YOLOE text encoder | `mobileclip2_b.ts` in the working directory | Restore the matching encoder to avoid first-use download. |
| YOLO-World encoder | Runtime/cache dependent | Obtain the encoder required by that detector. |
| GraspNet source | `source_root` in `configs/graspnet/default.yaml` | Obtain the upstream/local source under its own terms. |
| GraspNet checkpoint | `checkpoint` in the same YAML | Restore the checkpoint matching the configured SHA-256. |
| GraspNet runtime | `runtime_python` in the same YAML | Build compatible PyTorch/CUDA and native operators separately. |

Current GraspNet paths are `../Robot_arm/.venv/bin/python`, `../Robot_arm/graspnet`, and `../Robot_arm/graspnet/checkpoint-rs.tar`. They resolve relative to this repository. Update all three on the new computer; the app setup script does not create that environment.

Follow the [GraspNet baseline installation instructions](https://github.com/graspnet/graspnet-baseline#installation) for pointnet2/knn operators and [GraspNet API](https://github.com/graspnet/graspnetAPI). Match the CUDA/PyTorch/compiler combination to the actual GPU. Historical upstream requirements are not an automatically verified recipe for a new GPU.

Weights, encoders, environments, and scene caches are separate assets with their own terms. The application license does not relicense them.

## Calibration and poses

Retain and revalidate these local settings:

- `configs/calibration/camera_intrinsics.yaml`.
- `configs/calibration/handeye_active.yaml`, `handeye_park.yaml`, and `handeye_tsai.yaml`.
- `configs/calibration/tcp_offset_estimate.yaml` and `configs/robot/piper_default.yaml`.
- `configs/task/pick_demo_points.yaml` and `pick_demo_default.yaml`.
- `configs/graspnet/default.yaml`.
- `configs/calibration/local_board/charuco_board.yaml`.

Local YAML files were included in the migration snapshot despite some ignore rules. New ignored calibration sessions/raw captures are not tracked automatically.

Revalidate when mounting, intrinsics/image mode, TCP, gripper geometry, or workspace changes. A copied taught pose does not qualify a different physical scene.

## Preview and offline checks

```bash
./scripts/run_graspnet_target_pick_demo.sh
```

This preview opens the camera, may read pose telemetry, and refuses motion. Existing saved RGB-D scenes can instead be replayed without camera or CAN:

```bash
./scripts/run_graspnet_target_pick_demo.sh   --replay-scene /path/to/scene.npz --dry-run
```

Scenes are saved under `.cache/graspnet/requests/<scene_id>/`. The portfolio PNG is a perception example, not a replacement for an RGB-D `scene.npz`.

Existing regression suite, using the prepared application environment:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

Record the commit, environment, and logs for each test claim. Unit tests, replay, and physical trials are separate evidence categories.

## Physical setup

Set up camera USB access, CAN, GPU drivers/runtime, and measured gripper geometry. See [installation](installation.md), [hand-eye calibration](handeye.md), and [keypoints](keypoints.md).

Before motion, follow the [supervised trial guide](graspnet_pick_demo.md), including open-jaw alignment and fresh scene/pose checks. The cached `.cache/graspnet/frame-verification.json` is not portable proof of correct alignment. The dedicated execution demo starts with dry-run checked.

[LOCAL_SETUP.md](LOCAL_SETUP.md) is a historical log containing provisional values. Current YAML and the trial guide describe adopted settings.
