# Project provenance and acknowledgements

## Maintainer and scope

Joel Tang maintains this adaptation for a Piper arm with a wrist-mounted RealSense D435i. Its portfolio contribution is system integration: local hardware calibration and configuration, perception adaptation, connecting a pretrained grasp model to planning and execution, failure investigation, and verification.

YOLOE, YOLO-World, GraspNet, the robot SDK, and the camera SDK are upstream work. Using pretrained models does not constitute training or inventing them.

## Starting application

This project derives from [xsb2lyang/piper-vision-grasp-new](https://github.com/xsb2lyang/piper-vision-grasp-new), starting locally at commit [831bb7bf99d002b5c29ed9e720bee7521842386c](https://github.com/xsb2lyang/piper-vision-grasp-new/commit/831bb7bf99d002b5c29ed9e720bee7521842386c).

That baseline already supplied the application/package structure, setup/configuration tools, calibration, monitoring, teleoperation, keypoint capture, click-to-pick, and YOLO target-pick foundations. It also included the vendored SDK and experimental YOLO workspace.

The original README remains in Git history and at the linked baseline. Existing commit authorship, copyright notices, and licenses are preserved.

## Work in this adaptation

Migration commit `39d312d` collects the local integration work as one snapshot, rather than a chronological account of every step.

| Area | Implementation evidence |
| --- | --- |
| YOLOE prompts, categories, and masks | `src/piper_app/perception/yolo.py`, `src/piper_app/yolo_target_pick_demo/gui.py` |
| Camera selection and captured-frame geometry | `src/piper_app/camera/d405.py`, `tests/test_camera_selection.py` |
| GraspNet process boundary, geometry, and previews | `src/piper_app/graspnet/client.py`, `worker.py`, `geometry.py`, `gui.py` |
| Gripper mapping, bounded planning, and route checks | `src/piper_app/graspnet/planning.py`, `swept.py` |
| Supervised execution and failures | `src/piper_app/graspnet/execution.py`, `live_gui.py` |
| Local calibration and task setup | `configs/calibration/`, `configs/robot/`, `configs/task/`, `configs/graspnet/` |
| Regression checks and operating notes | `tests/test_graspnet_*.py`, `tests/test_tcp_frame.py`, `docs/graspnet_pick_demo.md` |

These are adapted or added areas; the table does not claim that every algorithm or line originated here. Model conventions and third-party code retain their attribution.

## Upstream projects

- **[AgileX Robotics / pyAgxArm](https://github.com/agilexrobotics/pyAgxArm):** robot communication, data structures, kinematics conventions, and control primitives. Vendored under `third_party/pyAgxArm`.
- **[Ultralytics](https://github.com/ultralytics/ultralytics):** YOLO runtime and pretrained YOLOE/YOLO-World perception. See [YOLOE documentation](https://docs.ultralytics.com/models/yoloe/).
- **[GraspNet baseline](https://github.com/graspnet/graspnet-baseline)** and **[GraspNet API](https://github.com/graspnet/graspnetAPI):** grasp inference, representations, and baseline collision utilities. See [GraspNet-1Billion](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.html), by Hao-Shu Fang, Chenxi Wang, Minghao Gou, and Cewu Lu.
- **[RealSense SDK](https://github.com/realsenseai/librealsense):** RGB-D streaming, alignment, camera intrinsics, and deprojection.
- **[OpenCV](https://github.com/opencv/opencv)** and **[opencv_contrib](https://github.com/opencv/opencv_contrib):** image processing and ChArUco/calibration tools.
- The inherited **`third_party/yolo/新松-检测` workspace** and its nested contributors, including the CUTLASS and Mamba notices already listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Other dependencies, including PyTorch, NumPy, and SciPy, retain their own authorship and licenses.

The local `Robot_arm` checkout supplies the configured GraspNet source, checkpoint, and runtime. Those external files do not become part of this repository through a YAML reference.

## Licensing boundaries

Retain the inherited [LICENSE](LICENSE) and applicable component notices. This acknowledgement does not change their terms or grant redistribution rights for third-party models. It does not assert one license for the complete stack. GraspNet baseline has separate research-use restrictions; see [runtime/model notices](THIRD_PARTY_NOTICES.md#runtime-and-model-dependencies).
