# Why GraspNet has previews but no complete execution plan

The yellow drawing is a camera-frame grasp proposal. The restricted pick demo approves only a complete sequence: viewing pose → pregrasp → grip → lift → destination → release → withdrawal → observe. Any failed stage rejects the complete plan. At present even Approach Only requires this complete-plan validation. That is an integration limitation, not a GraspNet requirement.

Updated 2026-11-02: the old orientation vetoes are retired, pregrasp uses sampled joint-space routes with bounded multi-seed IK, and a checked inward recovery can accommodate capture overruns of at most 1 degree. Commanded targets retain the same SDK limits. A refusal does not automatically establish that a visible grasp is physically bad. Do not repeatedly re-teach poses or recalibrate the camera in response to unrelated messages.

**Current task correction:** the user clarified that the saved destination is above a garbage bin. Default placement now honors the full taught release TCP pose, including height and orientation. Destination-table support/occupancy tests below apply only to explicit tabletop mode; they do not apply to this bin air-release. Pickup support and observed-scene path checks remain. The absolute camera-global-timestamp gate has also been replaced by verified new frame acquisition with a stationary pose bracket.

## Responsiveness corrections, November 2

Planning now returns the first fully checked solution, initially searching 20 proposals and automatically continuing through its retained fallback pool. It solves the grasp before the continuous entry, permits geometry-checked shorter entry/lift, and uses a checked joint transfer directly to the taught bin release. No joint range or final IK accuracy was relaxed. Reused scene structures, finite-radius queries and spatial bounds remove repeated work. Payload collision evidence counts distinct scene voxels, and known source support contact during upward separation is distinguished from a new obstacle. Target-surface tolerance is capped at 4 mm. The complete-transfer requirement before Approach Only remains.

## Before a scene can be used for motion

| Refusal | Exact meaning in this implementation |
|---|---|
| Missing/stale robot feedback | SDK TCP, joints and motion status must be present and have timestamps no more than 250 ms old; future timestamps beyond 50 ms are also refused. |
| Robot not stationary / moved during capture | Measured joint/TCP feedback must be stable; the SDK target-not-reached flag does not itself mean moving. Before/after TCP must differ by no more than 1.5 mm and 0.75°, and joint readings by no more than 0.01 rad. |
| No settled pose bracket / bracket exceeded | A before/after robot sample must surround the frame receipt within 350 ms. A preview-only frame cannot become an executable target. |
| Camera global timestamp stale/unavailable — retired | This used to compare camera and host wall time within 150/50 ms. It could reject real new images when clocks differed. The check is removed. Freeze now drains pending frames, waits across a new-frame barrier, requires an advancing color frame counter, and brackets the acquisition with stationary robot samples. Hardware/global SDK timestamps are diagnostic only. |
| New frame acquisition not verified | A cached/preview frame, repeated counter, missing new RGB-D pair or acquisition outside the pose bracket cannot supply an executable target. This protects image-to-arm-pose consistency without clock synchronization. |
| Wait for stationary observations | The application requires at least half a second of stable observations before qualifying a frame. |
| Different calibration / incompatible TCP axes | The frozen scene's frame fingerprint must match the current TCP, hand-eye, robot model and jaw dimensions. This adapter expects business-TCP axes aligned with flange axes. |

## Before GraspNet returns previews

| Refusal | Meaning |
|---|---|
| No category/detection, mask or intrinsics | The selected instance must have an aligned segmentation mask and captured camera geometry. Selecting a bounding-box centre is insufficient for this mode. |
| Insufficient/fragmented object depth | Valid depth must be 0.08–0.90 m, with at least 200 retained pixels and 15% mask coverage. The largest depth-connected surface must retain at least 55% of the valid eroded mask pixels. Missing depth is not invented. |
| No candidate survives | Neural candidates can fail finite-value/rotation/dimension checks, exceed the 96 mm grasp-width cap (100 mm gap less 4 mm opening allowance), lie over 15 mm from the cleaned object cloud, collide with scene points in the baseline model, or enclose insufficient observed object points. |
| Inference/configuration error | Missing checkpoint/runtime, checkpoint hash mismatch, invalid configuration, CUDA worker failure/90-second timeout, malformed results and stale result IDs are software/data failures. They are not diagnoses of a bad physical grasp. |

Scores are ranking values, not success probabilities. The baseline collision test is a simplified virtual-gripper check; it is not the complete robot path check.

## After previews: every deliberate motion-plan rejection category

| Message/category | What it checks; what it does not prove |
|---|---|
| Outside upright-object side-grasp family — retired | Former 35° approach/20° closing-axis policies are removed. Downward, side and angled proposals can be assessed; orientation is a preference, not a veto. |
| Too much wrist rotation — retired | Former 45° whole-hand orientation difference from capture is removed. It was never a J5 angle limit. |
| Capture outside small-start recovery allowance | A measured joint exceeds an SDK boundary by more than 1°. Smaller overruns receive a planned inward recovery; new commanded targets remain inside nominal limits. Recovery runs only after a live action, with verified arrival before pregrasp. |
| Saved observe joint outside SDK limits | The taught return pose is outside those limits. Update that record only. The user has already corrected the previous observe-J5 issue; the latest saved value is about 69.96°. Software limits alone do not diagnose harmful mechanical overextension. |
| Too little tabletop / insufficient consistent plane / support not horizontal | At least 300 consistent nearby non-object points must establish a table plane. Refinement uses a 4 mm residual band; slope must be within 10° of horizontal. A real table can fail this because of depth holes, view coverage or the simplistic plane-selection method. |
| Object bottom/support not established | The second percentile of visible object height above the fitted table must be between −8 mm and +15 mm. Missing the lower portion of the object can cause refusal even when it is visibly standing on the table. |
| Destination tabletop not visible | At least 50 points within 6 mm of the fitted plane must be visible inside the translated object footprint and configured margin. This concerns the saved drop XY, which may be off-camera while the pickup object is clearly visible. |
| Destination footprint may be occupied | More than 20 non-target points over 12 mm above that plane occur inside the footprint plus the configured geometry margin. This is depth evidence, not semantic recognition of an obstacle. The old oversized circle/15 mm padding produced a confirmed false positive; it now uses the translated XY convex hull with the configured 4 mm margin and excludes cleaned selected-object points. Hand/camera/payload checks remain separate. |
| Pose/path outside workspace | The planned TCP poses and sampled return path must remain inside the configured XYZ box. A grasp centre can be inside while its pregrasp, lift, withdrawal or return is outside. This is not a full-arm reach test. |
| Captured or saved TCP/joints disagree with FK | SDK FK of the recorded joints must match the stored TCP within 5 mm and 2°. Possible causes include mixed/stale records, changed TCP settings or manually editing only one representation. This is not automatically a request to redo camera calibration. |
| No nearby IK solution | A local least-squares search failed to find joints within SDK limits that achieve the requested pose within 2 mm and 1°. The displayed numbers are solver residuals, not measured arm motion. A local failure does not prove that no solution exists elsewhere. |
| Local IK path discontinuous — formerly “singularity” | Along a sampled Cartesian path, successive computed joint solutions differ by more than 20° in at least one joint. Samples are spaced at most 10 mm/5° in TCP space. This may reflect a branch change, local solver difficulty or poor conditioning. The code does not compute a Jacobian-based singularity diagnosis. The message now states the actual solution jump and says continuity was not verified. |
| Hand/camera envelope intersects scene | Observed points enter nominal finger/palm/camera boxes along the path. These are approximate geometry checks and can be affected by depth noise, robot self-returns and imperfect body dimensions. |
| Carried object intersects scene/support | Observed carried-object points would penetrate the support plane by over 6 mm, or approach another observed object's points too closely. The checker uses partial observed geometry; it is not a complete object/arm mesh simulation. |
| Degenerate footprint or invalid force/settings | A 2D placement footprint cannot be established, or a required numerical setting is invalid. These are distinct from a low neural score. |

The capture-to-pregrasp movement now uses a deliberately planned joint route, with curved FK samples used for workspace and observed-scene checks. Both equivalent jaw orientations and up to seven bounded IK seeds are tried. Final insertion remains straight. The planner does not search arbitrary detours around obstacles: if its direct joint routes intersect observed geometry, it refuses. The underlying SDK executes the joint endpoint command; sampling checks its nominal synchronized joint interpolation, not a certified full-arm swept volume.

## Later refusals and execution faults

Even a completed motion plan cannot run if another action is active, the displayed scene/candidate/configuration changed, the plan is over 180 seconds old on the monotonic clock, the robot moved from its expected starting pose, the camera/target changed, the final frame is stale, or the gripper mapping lacks the observed first-approach verification. Older plans without a monotonic capture age require a new freeze. An approached plan cannot be replaced midway. A failed attempt requires inspection/recovery before restarting.

Before initial movement a measured stationary window is required, and controller faults remain errors. After an open approach, the confirmation budget is 60 seconds from arrival, with fresh stable feedback at the same pregrasp; Return to View can cancel even after this budget expires. Once motion starts, missing/unhealthy gripper feedback, failed arm enable, failure to settle within 5 mm/3°, an unexpected joint branch, empty close, incomplete opening, timeout or operator stop prevents later stages. Those are execution faults, not reasons the original preview lacked a complete plan. No move-P fallback is introduced.

## Historical evidence before the November 2 movement changes

- Upright wooden block: 40 previews; 19 outside the orientation family, 13 beyond the wrist-rotation policy, and 8 blocked by destination visibility. No complete plan passed.
- Cup frame with the false occupancy refusal: 40 previews; 23 outside the family, 11 beyond wrist policy, 6 blocked by the old occupancy circle. That circle counted 217 points in small patches on apparently empty paper. The corrected footprint counts 0 obstacle points and 140 support points. The same 6 candidates then reach the next checks: 2 fail pregrasp IK continuity and 4 fail the local pregrasp IK search. This fixes one false refusal without proving the whole transfer feasible.
- Three of those six cup pregrasp endpoints solve in a separate offline endpoint test. Their complete approach paths have not been approved. The distinction between endpoint reachability and route validation matters.

These results identify execution-planner limitations; the user does not need to study or retrain GraspNet to diagnose them. All analyses used saved data and issued no arm commands.

Rejection logs: motion_rejections.json records the latest validation; motion_validation_history.jsonl now preserves prior checks when Next Grasp is used. Last Event groups reasons by category rather than showing one unexplained failure as though it describes every candidate.
