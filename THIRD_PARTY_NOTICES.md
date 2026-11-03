# Third-Party Notices

This repository vendors third-party code for practical robotics development.

## Vendored Components

### `third_party/pyAgxArm`

- Upstream: AgileX `pyAgxArm`
- License text included at: [third_party/pyAgxArm/LICENSE](third_party/pyAgxArm/LICENSE)

### `third_party/yolo/新松-检测`

- Contains a vendored YOLO workspace used for local experiments and integration work.
- The repository should be reviewed before redistribution because nested components may carry their own notices.
- Known bundled notices include:
  - [third_party/yolo/新松-检测/ultralytics/nn/extra_modules/cutlass/LICENSE.txt](third_party/yolo/新松-检测/ultralytics/nn/extra_modules/cutlass/LICENSE.txt)
  - [third_party/yolo/新松-检测/ultralytics/nn/extra_modules/mamba/LICENSE](third_party/yolo/新松-检测/ultralytics/nn/extra_modules/mamba/LICENSE)

If you redistribute this workspace, review the original upstream repositories and bundled notices carefully.


## Application provenance

The starting application is [xsb2lyang/piper-vision-grasp-new](https://github.com/xsb2lyang/piper-vision-grasp-new), based locally on commit `831bb7bf99d002b5c29ed9e720bee7521842386c`. Its history, root license statement, and existing notices are retained. See [CREDITS.md](CREDITS.md).

## Runtime and model dependencies

These entries supplement the existing notices. They do not replace component terms or grant redistribution rights for model weights.

| Component | Source and role | Applicable notice |
| --- | --- | --- |
| Ultralytics YOLOE / YOLO-World | [Ultralytics](https://github.com/ultralytics/ultralytics), installed runtime and pretrained perception | See its [license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) and [licensing page](https://www.ultralytics.com/license). Community AGPL terms and separate enterprise licensing exist. The root LGPL statement does not relicense it. |
| GraspNet baseline | [GraspNet baseline](https://github.com/graspnet/graspnet-baseline), externally configured source/checkpoint/runtime | Its [agreement](https://github.com/graspnet/graspnet-baseline/blob/main/LICENSE) specifies noncommercial research use and restricts redistribution. It is not MIT-licensed. Obtain external components under their applicable terms. |
| GraspNet API | [graspnetAPI](https://github.com/graspnet/graspnetAPI), grasp representations/utilities | Retain notices supplied with the actual installed version. |
| RealSense SDK | [librealsense](https://github.com/realsenseai/librealsense), through pyrealsense2 | [Upstream license](https://github.com/realsenseai/librealsense/blob/master/LICENSE). |
| OpenCV / contrib | [OpenCV](https://github.com/opencv/opencv) and [opencv_contrib](https://github.com/opencv/opencv_contrib), image processing/calibration | Preserve notices for the installed distributions and bundled modules. |

PyTorch, NumPy, SciPy, encoders, and other dependencies retain their separate notices. Weights can have terms distinct from the application code.

The inherited root statement is retained without relicensing third-party code. This documentation does not certify the combined distribution under a single license. Review actual versions and applicable permissions before packaging or commercial distribution.
