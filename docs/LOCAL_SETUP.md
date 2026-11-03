# Local setup - October 27, 2026

This is a separate checkout for Joel's Piper + wrist D435. The old Robot_arm
checkout and its environment have not been modified by this setup.

- Baseline: 831bb7bf99d002b5c29ed9e720bee7521842386c.
- Python 3.10 environment created with the repository setup script.
- uv is installed locally in .tools/bin; add that directory to PATH when rerunning setup.
- CPU PyTorch chosen with UV_TORCH_BACKEND=cpu.
- The configured package mirror resolved/downloaded dependencies faster than PyPI
  in this WSL environment. No global index configuration was changed.
- Pin both OpenCV distributions to 4.13.0.92: the upstream unbounded constraint
  selected 5.0.0.93, which lacks interpolateCornersCharuco and calibrateHandEye
  called by this application. Existing import tests did not detect this failure.
- Corrected PDF print scale without changing the board image or YAML geometry.
- The checkout does not contain yolo11m.pt despite the doctor listing it as a
  tracked asset. Click-to-pick can be used without YOLO. Obtain weights before
  enabling detection.
- The supplied TCP offset, saved keypoints and task offsets belong to the author.
  They have not been adopted for Joel's hardware. Capture local calibration and poses.
- The camera, CAN interface, robot, gripper and control modes have not been opened
  or commanded during this setup. Real hardware compatibility is still untested.

Full sequence: intrinsics -> provisional hand-eye if TCP estimation is needed ->
TCP estimate -> final hand-eye/validation -> local keypoints -> click-pick -> YOLO.
Use the upstream docs for the tools and the chat's piper-restart-plan for details.

## Calibration and execution frame correction

Read-only connections previously skipped set_tcp_offset, while motion connections
applied it. In this SDK the setter updates local transform matrices only. The client
now applies that host-side offset to every connection, keeping speed setup conditional.
The test tests/test_tcp_frame.py reproduces the mismatch before the fix and passes
afterward, including that read-only connection sends no speed request. No SDK vendor
code was changed. For the provisional flange calibration, explicitly pass
--tcp-offset 0 0 0 0 0 0 to both hand-eye calibration and validation. Use the adopted
measured TCP consistently for final calibration, validation and keypoint capture.

The optional model download from GitHub was cancelled after severe throttling.
The official Hugging Face model API was unreachable from WSL. No valid YOLO weight
file was left, and the incomplete file was removed. Use --no-yolo for click-pick.
Importing the new Ultralytics package automatically migrated its shared per-user
settings file; the old project source and virtualenv were not edited.

## D435i camera selection correction

The D435i was visible to pyrealsense2, but the camera adapter filtered out all
non-D405 device names. The selector now accepts D405 and D435-family names
(case-insensitive), preserving explicit serial selection and rejecting other
models. Existing class/module names remain for API compatibility. No robot
control or calibration values were changed. tests/test_camera_selection.py
checks model matching, serial selection, unknown serial and absent devices.

## Current printed board

User measured the whole pattern approximately 201 x 269 mm with a ruler.
Active local board config is configs/calibration/local_board/charuco_board.yaml:
33.6 mm squares, 24.64 mm markers retaining the original 22/30 print ratio.
Nominal pattern is 201.6 x 268.8 mm. Original 30/22 mm assets remain unchanged.
The user is continuing in this chat and skipped movement qualification.

## Provisional local TCP and first transfer preparation

The six aligned captures in tcp_offset_estimate.yaml give
[0.01002688067284757, -0.010866636913774009, 0.10323078233419246, 0, 0, 0].
Sample std norm is 5.014 mm; five-view hand-eye validation remains 6.464 mm.
This candidate is provisional, not a claim of absolute pickup accuracy.

Robot defaults and local task pose configuration now use that candidate.
All three active/Park/Tsai hand-eye files were re-expressed using
T_grip_camera = inverse(T_flange_grip) @ T_flange_camera.
The base-to-camera chain is unchanged; an offline re-solve of the same 38 samples
agrees to floating-point precision. No new camera capture session was performed.
Original zero-offset hand-eye and author robot/task settings are backed up in
configs/calibration/sessions/20261027_041924_tcp_adoption/.

Local pick_demo_points.yaml starts empty; no author poses were copied.
Capture local observe/home, drop_pose and staging before using click-pick.
The observe orientation becomes the grasp orientation.
Task click offset is zero for preview; object surface-to-centre correction must be
set for the selected object. Grasp descent is explicitly linear; jaw opening is
64.4 mm from the prior measured opening. Upstream force remains 1.0 N.
This configuration preparation issued no robot movement.

Any later zero-offset TCP estimator must explicitly load the backed-up flange
hand-eye, because active hand-eye now refers to the gripping-centre frame.

## First grey-block transfer

User confirmed the upright block is 84.5 x 24.5 x 15 mm. Select the centre of its
TOP face, and apply [0, 0, -0.04225] m in base axes to grip halfway down its height.
The supplied local observe/home and drop_pose were physically taught on this setup;
staging reuses the local observe pose. No author point was imported.

Fixed build_pick_plan so the object-specific pick offset does not also shift the
separately captured release TCP pose. Added a regression for the block-height
correction, release pose, and approach pose. Actual placement still uses the
existing saved drop joint pose. The existing sequence and fault handling remain.
Click-pick now prints the selected plan and existing UI step events to stdout so
the attempt can be inspected despite the Windows UI helper launch failure.
