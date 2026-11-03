# YOLO Target Pick Demo

Type a category, pause the frame, and execute the highest-confidence matching detection. The demo automatically selects its box centre; no image clicking or manual box drawing is needed.

## Launch with the local YOLOE checkpoint

From the repository in Ubuntu:

```bash
./scripts/run_yolo_target_pick_demo.sh --yolo-weights models/yoloe-26s-seg.pt --target-label "bottle" --dry-run
```

Keep your existing camera, CAN interface, channel and speed flags. Use `--no-dry-run` for the existing live pickup mode when ready.

## Workflow

1. Move to the saved Observe pose and keep the camera/arm stationary while framing the target.
2. Type a category, such as `bottle`, `wooden block`, `gray rectangular block`, or `computer mouse`, then press Enter. Selecting an existing dropdown entry also applies it.
3. With promptable YOLOE weights, the field configures the model's text prompt. This does not train the model. Recognition of every description or pose is not guaranteed.
4. Inspect the automatic box and, for segmentation weights, the green mask. `Show masks` toggles the overlay; the original-image boolean mask remains attached to each detection for future geometry processing.
5. Press Pause / Freeze, then Execute Pick. A category changed while paused is inferred again on that same frozen image. Old predictions/plans are cleared when prompts change or fail.
6. If multiple instances match, the highest-confidence one is selected. If none match or the selected centre has invalid depth, execution refuses to run.

Ordinary fixed-class YOLO weights and prompt-free/fused YOLOE checkpoints retain class filtering. Typing an unavailable class shows a message rather than pretending to configure an open-vocabulary prompt. YOLO-World supports text prompts but produces boxes without instance masks.

The text encoder may be loaded on the first prompt; use the already installed local encoder/cache. Prompt or dependency errors appear in Last Event. Changing the category, pausing/unpausing, clearing the target and reloading the workspace are blocked during a robot action.

## Current grasp limits

This update adds perception only. It uses the existing box centre, depth query, saved approach orientation, offsets, opening, force and placement sequence. A detected centre and valid depth do not establish a suitable grasp for arbitrary heights, widths or orientations. Detecting a horizontal bottle does not adapt its pickup angle or insertion depth; geometry-based grasp planning is the next stage.

Masks are displayed and retained, but do not yet determine the pickup pixel or gripper angle. Robot settings and the existing motion executor are unchanged.
