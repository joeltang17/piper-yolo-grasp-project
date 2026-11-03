# Supervised GraspNet pick trial

Responsiveness update, 2026-11-02: scene geometry and collision trees are reused; finite-radius queries and conservative spatial bounds reduce work without thinning the observed scene. Payload checks count distinct occupied scene voxels. Target-surface ownership follows the existing geometry tolerance, capped at 4 mm; independent nearby obstacles remain. Known pickup-support contact is allowed only while lifting away, with final clearance required; transport still checks observed tables and other objects.

Updated 2026-11-02. Start with one isolated, rigid, opaque block at the pickup area. The destination is the full taught TCP release pose above the garbage bin. Downward grasps are preferred where feasible; side and angled neural proposals remain eligible. Detection tuning, transparent-bottle reconstruction, and model training are separate work.

## Launch and first trial

From `/home/joeltang/piper-vision-grasp-new`:

```bash
./scripts/run_graspnet_pick_demo.sh
```

No extra parameters are needed. Dry-run starts checked. The older `run_graspnet_target_pick_demo.sh` remains preview-only; `run_yolo_target_pick_demo.sh` retains its centre-point pickup.

Saved observe remains a strictly in-limit commanded target. A measured capture can be at most 1 degree outside each SDK joint limit: a checked inward correction is then included at the start of execution, targeting 1 degree inside the violated boundary and leaving other joints unchanged. Larger overruns are refused. Planning, opening the GUI and dry-run never perform recovery. Commanded joint limits and saved poses are unchanged. The earlier saved observe-J5 issue has already been corrected by the user.

See [all rejection categories and the saved-scene diagnosis](graspnet_plan_rejections.md). A visually plausible grasp can still fail a later placement or path check. Even Approach Only currently requires a complete transfer plan; the first version is conservative and some refusals reflect planner limitations rather than an impossible grasp.

1. Close other camera-owning demos. Put the empty gripper in the useful viewing pose and the block on the pickup surface. The saved drop pose should put the gripper above the bin opening, with room to release and withdraw.
2. Type **wooden block**, press Enter, wait for stationary observations, then **Pause / Freeze**. Freeze now drains pending frames, waits for advancing camera frames, and infers the mask on that exact new image. Camera and computer wall clocks need not match.
3. Press **Plan Grasp**. The displayed first proposal is tried first; planning returns the first complete checked solution instead of comparing three. It searches 20 prioritized proposals initially and automatically uses the remaining retained alternatives only when necessary. No repeated Next Grasp clicks are required. Status shows the proposal being checked, elapsed time, grasp J5, and any small start recovery.
4. With Dry-run checked, **Execute Pick** records the planned sequence without sending commands.
5. For the first physical check, uncheck Dry-run and press **Approach Only**. It opens to the SDK maximum, performs any planned small inward recovery, then moves through the checked joint route to pregrasp and stops before contact. Recovery must settle with joint feedback within 0.1 degree and inside the nominal limits before continuing. Check actual jaw orientation and centre against the intended approach. If wrong, stop and report the result; do not confirm alignment.
6. If the open-jaw alignment is correct, press **Jaws Aligned**, then **Execute Pick**. This records an operator verification for the current TCP, hand-eye, opening and pad geometry. It is invalidated when those change. Dry-run cannot create this verification.

After the first verification, a new checked plan can run the whole transfer through Execute Pick. Report physical pickup, transport and release separately. Sequence completion means the commanded poses settled; it does not prove the object stayed in the jaws. The initial target is eight clean transfers out of ten over nearby positions, before broadening objects and orientations.

## What changed

- **Object depth:** the dominant connected surface inside the mask supplies ownership points. Adjacent pixels must also be within 8 mm in 3D. Disconnected background returns are removed; missing depth is not filled. Fragmented or insufficient geometry is refused.
- **Neural input:** half the samples come from the cleaned object and half from nearby scene context when available. This restores the baseline's scene context without allowing background to establish object ownership. All valid scene points remain collision context.
- **Jaw mapping:** standard CAD establishes approach along flange Z and closing along flange Y. The calibrated business TCP represents the pad midpoint. GraspNet's tip is at `depth` along its X axis, so the TCP target is at `depth - 0.070/2`. Parallel-jaw symmetry flips closing/normal axes together. No object-height correction or guessed Euler offset is added.
- **Orientation:** the old 35-degree approach, 20-degree closing-axis and 45-degree capture-orientation vetoes are removed. Old YAML cap keys are ignored. Both equivalent jaw orientations and up to seven deterministic IK seeds are considered. No geometric-centre target or new neural grasp is fabricated.
- **Movement:** SDK modified-DH FK and an analytic Jacobian drive bounded IK with the same 2 mm/1 degree accuracy and unchanged joint limits. The grasp is solved first and its entry traced backwards on a continuous branch. A shorter entry is allowed only after the observed object clears the finger span (at least 40 mm backoff); lift is planned between 30 and 100 mm where reachable and must finish above the pickup support. Joint routes retain at most 1 degree/joint and 10 mm/5 degree TCP sampling. New IK goals retain 0.1 degree of encoder/settling reserve inside the unchanged SDK bounds. Final insertion stays straight.
- **Placement:** default bin release uses a checked joint transfer from lift directly to the full taught release pose; the artificial extra 10 cm hover/straight reorientation is removed. Saved XYZ and orientation are preserved. Optional tabletop mode retains its separate support placement stages.
- **Execution:** open → optional small start recovery → joint pregrasp → linear grasp → close/contact indication → lift → checked joint transfer to taught bin release → open/verify → withdraw → rise → checked joint return to observe. No execution-time fallback is used. Return to View explicitly cancels a completed OPEN approach along a separately checked reverse route, ending at the viewing pose (or its in-limit recovered version). It never closes or carries an object.

The physical fully-open gap is the measured **100 mm**, with **70 mm** pads. The first mode opens to the SDK's reported maximum command, avoiding an unverified conversion between SDK travel and physical pad gap. Every execution path is checked at full physical opening before contact/release. The cyan dot marks the pad-midpoint TCP target; yellow shows the proposed virtual grasp at its predicted width.

The simplified hand/camera checks use observed scene points and conservative nominal body dimensions. Finger thickness, complete mounting geometry, unseen object surfaces, and full-arm collisions are not physically verified. This is a supervised bench trial for the supported task, not an unrestricted six-axis collision planner. The first observed approach remains necessary.

## Rejection and evidence

Execution requires fresh SDK feedback and a measured stationary joint/TCP window around verified fresh frame acquisition. The SDK target-reached bit is diagnostic; target-not-reached is not treated as measured movement. Controller faults and actual measured movement still refuse capture/execution. Before first movement, scene consistency and the original 180-second capture age remain checked. After Approach Only, a separate 60-second inspection clock starts on verified arrival; continuation verifies fresh stable feedback and the same pregrasp. The original capture timestamp is never rewritten. An expired inspection directs Return to View instead of an impossible resume/freeze action. Static-scene assumptions remain those of a supervised trial.

Current settings: execution.placement_mode=taught_release and release_clearance_m=0.05. These honor the taught bin release pose; no task point or calibration file was overwritten by this correction. Restart the GUI and freeze a new frame to use the new acquisition policy. Older saved scenes remain available for offline replay.

Each request folder under `.cache/graspnet/requests/<scene_id>/` includes the scene, network request/result, statistics, `motion_rejections.json`, a passing `motion_plan.json`, and per-attempt logs. Plans include nominal joint path samples, actual tilt/orientation differences, working/whole-route joint margins, J5 angles and recovery measurements/target. Validation history records tested indices and the feasible shortlist. The one-time observed mapping record is `.cache/graspnet/frame-verification.json`. Inspect/recover the arm before restarting a failed attempt.

The front-block saved scene previously admitted a background anchor at 462 mm. Isolation removes 287 disconnected pixels while retaining 98.9% of its valid masked points; the cleaned object ends at 305 mm. With local context, the actual CUDA replay returns 31 previews with anchors at 241–287 mm. The rotated block returns 40 previews. Counts and neural scores are not pickup-success measurements.

Offline checks cover depth contamination, frames, clock/capture validity, SDK FK/Jacobian/IK, sampled routes, preserved taught release, obstacle checks, recovery, first-valid search, automatic fallback and execution fault ordering. The user previously reported two successful physical wooden-block transfers; this responsiveness update was tested offline and has not itself been physically operated.

## Source locations

In VS Code, the new logic is in `src/piper_app/graspnet/planning.py`, `swept.py`, `execution.py`, and `live_gui.py`. Foreground isolation is in `geometry.py`; neural context is in `worker.py`. Settings are in `configs/graspnet/default.yaml`.

References: [GraspNet baseline](https://github.com/graspnet/graspnet-baseline), [official collision/frame conventions](https://github.com/graspnet/graspnet-baseline/blob/main/utils/collision_detector.py). The local standard-gripper URDF establishes axis directions; current user measurements and TCP calibration establish this adapter's pad length and midpoint reference.
