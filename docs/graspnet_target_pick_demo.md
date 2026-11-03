# GraspNet target preview

This first GraspNet integration preserves category selection: **type category → Pause / Freeze → Plan Grasp**. It draws a yellow virtual gripper over the detected object. No arm/gripper motion can be executed from this preview window. The original target-pick launcher retains its existing live centre-point pickup.

A separate [restricted GraspNet pick trial](graspnet_pick_demo.md) now connects checked grasps to the existing client. This preview launcher retains its motion refusal.

## Launch

Place the arm at its useful viewing pose with the existing tools, then close the previous camera window. From the current repository in Ubuntu:

```bash
./scripts/run_graspnet_target_pick_demo.sh
```

The new window connects to the camera and optionally reads robot pose telemetry. Its robot connection uses `configure_robot=False, init_gripper=False`. Home/Observe/gripper controls and motion keyboard shortcuts are disabled. `--no-dry-run` is rejected by this launcher.

1. Type a category and press Enter. Confirm that its green mask covers the object.
2. Pause / Freeze. The highest-confidence matching instance is selected, as in the existing category demo.
3. Press **Plan Grasp (preview)**. GraspNet runs in a background inference process, leaving the GUI responsive.
4. Inspect the yellow gripper and the opening/depth/score. **Next Grasp** cycles surviving candidates.
5. Resume/reframe if the mask/depth is poor or no candidate survives. Changing the category or snapshot discards stale results.

Grasp scores are ranking values, not probabilities of physical success. The yellow fingers show GraspNet's baseline virtual gripper, not a verified model of the actual Piper jaws.

## Data path

- Every camera bundle records its own color intrinsics, distortion model, frame number and timestamp. Frozen bundle point queries use that bundle's intrinsics.
- The selected RGB, metric depth and instance mask are copied into a read-only snapshot.
- The largest depth-connected surface inside the mask supplies clean object ownership points. Balanced samples from that surface and nearby scene context feed the neural network; all valid scene points remain available for collision checks. Invalid depth is not filled with table depth. The box centre is not used as a contact point.
- The worker rejects invalid rotations/dimensions, widths above the configured opening envelope, candidates away from the selected object, empty grasps and baseline scene collisions. Rejections are counted.
- GraspNet returns camera-frame candidates. No legacy object-height offset or guessed Euler correction is applied.

The pose sample saved with the image is diagnostic and is not hardware synchronized. These previews are not execution-qualified base-frame targets.

## Existing GPU environment

`configs/graspnet/default.yaml` explicitly points to the previously working environment and checkpoint:

- `../Robot_arm/.venv/bin/python`
- `../Robot_arm/graspnet`
- `../Robot_arm/graspnet/checkpoint-rs.tar`

Paths resolve relative to this repository. The checkpoint SHA-256 is checked. No packages or weights are installed automatically. The application remains in its current environment; only the inference worker uses the old CUDA environment. It imports no robot controller and receives only point-cloud arrays and inference settings.

The user measured a 100 mm fully open inner-pad gap and 70 mm longitudinal pad length on 2026-10-28. The configuration now uses a 100 mm opening cap, a 4 mm total opening allowance (96 mm maximum proposed width), 200 minimum valid object pixels, 20,000 sampled neural points, 5 mm collision voxels, and a 90-second worker timeout. Review these in the YAML as evidence accumulates. Sampling duplicates does not increase the amount of observed geometry.

The preview and simplified collision model now use the measured 70 mm longitudinal pad span. The old 64.4 mm opening cap was an incorrect provisional assumption and is superseded. Finger thickness remains the baseline 10 mm and height comes from the prediction; the complete finger body, palm and mount are unverified. These checks do not establish full physical gripper or arm-path clearance. The live legacy pickup settings are separate and are not changed by this preview update.

## Saved scenes and replay

Each attempt saves RGB-D/mask/intrinsics, the request, candidates, model logs and a summary under:

```text
.cache/graspnet/requests/<scene_id>/
```

Accepted previews also save `preview.png`. A geometry failure still leaves `scene.npz` for diagnosis. Offline replay opens neither camera nor CAN:

```bash
./scripts/run_graspnet_target_pick_demo.sh \
  --replay-scene .cache/graspnet/requests/<scene_id>/scene.npz --dry-run
```

Replay uses a new request ID and preserves the original capture. NumPy files are loaded with `allow_pickle=False`.

## Before live GraspNet pickup

The user confirmed standard Piper fingers. The restricted trial now implements CAD-based axes and the calibrated pad-midpoint adapter, plus a checked sequence. It requires an observed open-jaw alignment before the first pickup. This preview remains diagnostic; its pose samples cannot activate that sequence. A visible candidate alone is not a complete pick-and-place plan.

## Offline verification

39 unittest cases passed and 61 Python files compiled. The actual CUDA worker passed an end-to-end replay on saved historical calibration-board depth; this is software integration evidence, not validation of an object grasp. A hidden-window GUI check verified disabled motion controls, mask selection, background result delivery and rendering without connecting camera or robot.
