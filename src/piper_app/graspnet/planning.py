"""Observed-scene grasp planning. Pure geometry/SDK FK; never opens CAN.

Joint transfers use sampled SDK FK; final insertion uses local Cartesian IK.
The hand/camera checks are not a full-arm collision planner.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from pyAgxArm import create_agx_arm_config, PiperFW
from pyAgxArm.utiles.mdh_kinematics import get_mdh, fk_from_mdh
from piper_app.calibration.transforms import pose6_to_matrix, matrix_to_pose6, translation_rotation_delta
from .geometry import SceneSelection, selected_point_clouds, validate_candidates


@dataclass(frozen=True)
class MotionStage:
    name: str
    tcp_pose: tuple[float, ...]
    joint_solution: tuple[float, ...]
    motion_kind: str = "linear"
    joint_path: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class GraspMotionPlan:
    scene_id: str
    candidate_index: int
    stages: tuple[MotionStage, ...]
    start_pose: tuple[float, ...]
    start_joints: tuple[float, ...]
    tcp_offset: tuple[float, ...]
    frame_signature: str
    config_signature: str
    capture_wall_s: float
    force_n: float
    physical_opening_m: float
    details_json: str
    capture_monotonic_s: float | None = None
    return_view: MotionStage | None = None

    @property
    def pregrasp_index(self):
        return next(i for i, stage in enumerate(self.stages) if stage.name == "pregrasp")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def frame_signature(workspace, config):
    return signature({"robot": workspace.robot, "tcp_offset": workspace.tcp_offset, "handeye": workspace.T_tcp_camera.tolist(),
                      "pad_length_m": config["pad_length_m"], "max_opening_m": config["max_opening_m"],
                      "axes": "G+X=P+Z,G+Y=P+Y,G+Z=-P+X; TCP=pad midpoint; v1"})


def workspace_signature(workspace, config):
    return signature({"frame": frame_signature(workspace, config), "task": workspace.task_defaults,
                      "drop": workspace.drop_pose.tcp_pose, "observe": workspace.observe.tcp_pose,
                      "execution": config["execution"], "planner": "first-feasible-grasp-seeded-v3.1"})


def piper_pad_tcp(row, pad_length_m, *, swap_jaws=False):
    """T_camera_businessTCP using the calibrated centre of the 70 mm pads.

    Standard CAD establishes approach=flange Z and closing=flange Y. The
    business TCP axes match flange axes for this calibration. GraspNet's tip
    lies at G.x=depth, so pad midpoint lies at depth-pad_length/2, NOT depth.
    Jaw symmetry changes Y/Z signs together, never the approach direction.
    """
    row = validate_candidates(np.asarray(row).reshape(1, 17))[0]
    R = row[4:13].reshape(3, 3).copy()
    if swap_jaws:
        R = R @ np.diag([1., -1., -1.])
    R_P_G = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    result = np.eye(4)
    result[:3, :3] = R @ R_P_G.T
    result[:3, 3] = row[13:16] + R[:, 0] * (row[3] - pad_length_m / 2)
    return result


def telemetry_sample(client, *, require_stationary=True):
    """Timestamped SDK feedback; reads only. SocketCAN timestamps use Unix time."""
    wall, mono = time.time(), time.monotonic()
    tcp, joints, status = client.robot.get_tcp_pose(), client.robot.get_joint_angles(), client.get_arm_status()
    if tcp is None or joints is None or status is None:
        raise RuntimeError("Fresh TCP, joint and motion feedback are required.")
    if any(not -0.05 <= wall - float(msg.timestamp) <= 0.25 for msg in [tcp, joints, status]):
        raise RuntimeError("Robot feedback is stale; capture a new settled frame.")
    # SDK motion_status is TARGET REACHED, not measured velocity. Stability is
    # checked over samples by capture qualification / execution, not this bit.
    if int(getattr(status.msg,"arm_status",0)) != 0:
        raise RuntimeError("Controller reports an arm fault; inspect/reset it before planning or movement.")
    return {"wall_s": wall, "monotonic_s": mono, "tcp_pose": list(tcp.msg), "joints": list(joints.msg),
            "motion_status": int(status.msg.motion_status), "joint_feedback_s": float(joints.timestamp)}


def samples_stable(samples, *, translation_m=.0015, rotation_deg=.75, joint_rad=.01):
    if not samples:
        return False
    anchor = samples[0]
    for sample in samples[1:]:
        d,a = translation_rotation_delta(pose6_to_matrix(anchor["tcp_pose"]),pose6_to_matrix(sample["tcp_pose"]))
        if d > translation_m or a > rotation_deg or np.max(np.abs(np.asarray(anchor["joints"])-sample["joints"])) > joint_rad:
            return False
    return True


def settled_feedback(client, *, duration_s=.3):
    """Measure a stationary window, independently of the target-reached flag."""
    samples = [telemetry_sample(client,require_stationary=False)]
    deadline = time.monotonic()+duration_s
    while time.monotonic() < deadline:
        time.sleep(.05)
        samples.append(telemetry_sample(client,require_stationary=False))
    if not samples_stable(samples):
        raise RuntimeError("Measured joint/TCP feedback is moving. Wait for the arm to settle.")
    if samples[-1]["joint_feedback_s"] <= samples[0]["joint_feedback_s"]:
        raise RuntimeError("Joint feedback did not advance during the stationary check.")
    return samples[-1]


def qualify_capture(bundle, before, after, workspace, config):
    """A bounded stationary capture, not a claim of hardware clock synchronization."""
    if before is None or after is None:
        raise RuntimeError("No settled robot-pose bracket for this frame. Resume and freeze again.")
    dt = after["monotonic_s"] - before["monotonic_s"]
    if not 0 <= dt <= 0.35:
        raise RuntimeError("Frame/pose bracket exceeded 350 ms. Resume and freeze again.")
    if bundle.received_monotonic_s is None or not before["monotonic_s"] <= bundle.received_monotonic_s <= after["monotonic_s"]:
        raise RuntimeError("Frame receipt does not belong to this pose bracket.")
    if not getattr(bundle, "fresh_frame", False):
        raise RuntimeError("Execution needs a newly acquired frame. Resume and freeze again.")
    read_start = getattr(bundle, "read_started_monotonic_s", None)
    previous_number = getattr(bundle, "previous_frame_number", None)
    if (read_start is None or not before["monotonic_s"] <= read_start <= bundle.received_monotonic_s
            or bundle.frame_number is None or previous_number is None or bundle.frame_number <= previous_number):
        raise RuntimeError("Fresh frame acquisition could not be verified against the pose bracket.")
    delta = translation_rotation_delta(pose6_to_matrix(before["tcp_pose"]), pose6_to_matrix(after["tcp_pose"]))
    if delta[0] > 0.0015 or delta[1] > 0.75 or np.max(np.abs(np.array(before["joints"]) - after["joints"])) > 0.01:
        raise RuntimeError("Arm moved during capture; resume and freeze a stationary scene.")
    if not np.allclose(workspace.tcp_offset[3:], 0, atol=1e-8):
        raise RuntimeError("This jaw adapter requires the locally calibrated flange-aligned TCP axes.")
    return {"pose": after["tcp_pose"], "joints": after["joints"], "wall_s": after["wall_s"],
            "monotonic_s": after["monotonic_s"],
            "bracket_s": dt, "pose_delta_m_deg": delta,
            "frame_signature": frame_signature(workspace, config),
            "policy": "stationary pose bracket + drained stream and advancing frame counter; wall clocks not used"}


class PiperKinematics:
    def __init__(self, robot, tcp_offset):
        cfg = create_agx_arm_config(robot=robot, firmeware_version=PiperFW.DEFAULT)
        limits = np.asarray([cfg["joint_limits"][name] for name in cfg["joint_names"]])
        self.lower, self.upper = limits[:, 0], limits[:, 1]
        # Leave a small goal reserve for CAN angle quantization / settling.
        # Factory command limits, measured starts and taught poses are unchanged.
        self.encoder_reserve = np.deg2rad(.1)
        self.mdh, self.offset = get_mdh(robot), pose6_to_matrix(tcp_offset)

    def forward(self, joints):
        return self.forward_jacobian(joints)[0]

    def forward_jacobian(self, joints):
        """SDK modified-DH chain and geometric TCP Jacobian, without Euler round trips."""
        t = np.eye(4); axes, origins = [], []
        for q, (d, a, alpha, theta_offset) in zip(joints, self.mdh):
            ca, sa = math.cos(alpha), math.sin(alpha)
            ct, st = math.cos(q+theta_offset), math.sin(q+theta_offset)
            link = np.array([[ct,-st,0.,a], [ca*st,ca*ct,-sa,-sa*d],
                             [sa*st,sa*ct,ca,ca*d], [0.,0.,0.,1.]])
            t = t @ link
            axes.append(t[:3,2].copy()); origins.append(t[:3,3].copy())
        actual = t @ self.offset
        axes, origins = np.asarray(axes), np.asarray(origins)
        jac = np.r_[np.cross(axes, actual[:3,3]-origins).T, axes.T]
        return actual, jac

    def require_limits(self, joints, name):
        q = np.asarray(joints, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise RuntimeError(f"{name} must contain six finite joint angles.")
        bad = np.flatnonzero((q < self.lower) | (q > self.upper))
        if len(bad):
            j = int(bad[0])
            remedy = ("Return to an in-limit viewing pose and freeze again." if "capture" in name.lower()
                      else "Re-capture that saved pose within SDK limits.")
            raise RuntimeError(f"{name} joint {j+1} is {np.degrees(q[j]):.2f} deg, outside SDK limits "
                               f"[{np.degrees(self.lower[j]):.2f}, {np.degrees(self.upper[j]):.2f}]. {remedy}")

    def recovery_target(self, joints, execution):
        """Keep the measured start; only an explicitly bounded inward entry is allowed."""
        q = np.asarray(joints, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise RuntimeError("Capture must contain six finite joint angles.")
        overrun = np.maximum(np.maximum(self.lower-q, q-self.upper), 0)
        maximum = float(execution.get("start_recovery_max_overrun_deg", 1.0))
        inset = float(execution.get("start_recovery_inset_deg", 1.0))
        if not 0 <= maximum <= 1 or not 0 < inset <= 1:
            raise ValueError("Start recovery permits at most 1 deg overrun and at most 1 deg inset.")
        if np.max(overrun) > np.deg2rad(maximum) + 1e-10:
            j = int(np.argmax(overrun))
            raise RuntimeError(f"Capture joint {j+1} is outside SDK limits by {np.degrees(overrun[j]):.2f} deg; "
                               f"small-start recovery permits at most {maximum:g} deg. Move to an in-limit pose and freeze again.")
        if not np.any(overrun > 0):
            return None
        target = q.copy()
        target[q < self.lower] = self.lower[q < self.lower] + np.deg2rad(inset)
        target[q > self.upper] = self.upper[q > self.upper] - np.deg2rad(inset)
        self.require_limits(target, "Recovery target")
        return target

    def joint_samples(self, start, end, *, recovery=False):
        """Sample the curved TCP route of nominal synchronized joint interpolation."""
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        self.require_limits(b, "Joint target")
        if not recovery:
            self.require_limits(a, "Joint start")
        count = max(1, int(np.ceil(np.max(np.abs(b-a)) / np.deg2rad(1))))
        while True:
            joints = [a*(1-f)+b*f for f in np.linspace(0, 1, count+1)]
            poses = [self.forward(q) for q in joints]
            if all(d <= .01 and r <= 5 for d, r in
                   (translation_rotation_delta(x, y) for x, y in zip(poses, poses[1:]))):
                return joints[1:], poses[1:]
            count *= 2
            if count > 4096:
                raise RuntimeError("Joint path could not be sampled within the geometry resolution.")

    def pregrasp_solutions(self, target, start, observe, *, label="pregrasp"):
        """Bounded deterministic multi-start search; never changes joint limits."""
        seeds = [np.asarray(start), np.asarray(observe), (self.lower+self.upper)/2]
        for shoulder, elbow, roll, bend in [(1., -1.2, 0., -.6),
                                           (2.3, -1., 0., .3),
                                           (2.3, -1., .8, -.3),
                                           (2.3, -1., -.8, .3)]:
            seed = np.asarray(start).copy()
            seed[1:3] = [shoulder, elbow]; seed[3:6] = [roll, bend, 0.]
            seeds.append(np.clip(seed, self.lower, self.upper))
        solutions = []
        last_error = None
        for seed in seeds:
            try:
                q = self.inverse(target, seed)
            except RuntimeError as exc:
                last_error = exc
                continue
            if not any(np.max(np.abs(q-other)) < .02 for other in solutions):
                solutions.append(q)
        if not solutions:
            raise RuntimeError(f"{label}: no nearby IK solution from {len(seeds)} bounded seeds. {last_error}")
        def cost(q):
            margin = np.min(np.minimum(q-self.lower, self.upper-q))
            return np.linalg.norm(q-start) + .3*abs(q[4]) - .5*margin
        return sorted(solutions, key=cost)

    def reachable_prefix(self, start, end, seed, minimum_m, *, joint_trace=None):
        """Plan a shorter clear entry/lift when the arbitrary maximum is unreachable.

        This is offline route construction, never an execution-time fallback.
        Every retained pose meets the same joint/IK/continuity requirements.
        """
        distance,_=translation_rotation_delta(start,end)
        count=max(1,int(np.ceil(distance/.005)))
        q=np.asarray(seed); last=start; reached=0.; failure=None
        for f in np.linspace(0,1,count+1)[1:]:
            target=start.copy();target[:3,3]=start[:3,3]*(1-f)+end[:3,3]*f
            try:
                solved=self.inverse(target,q)
                if np.max(np.abs(solved-q))>np.deg2rad(20):
                    raise RuntimeError('Local IK path is discontinuous during entry/lift.')
            except RuntimeError as exc:
                failure=exc;break
            q=solved;last=target;reached=distance*f
            if joint_trace is not None:joint_trace.append(q)
        if reached+1e-9<minimum_m:
            raise RuntimeError(f'No continuous IK prefix reaches required {minimum_m*1000:.0f} mm clearance. {failure}')
        return q,last,reached

    def inverse(self, target, seed):
        last_q, last_values = None, None
        def evaluate(q):
            nonlocal last_q, last_values
            if last_q is not None and np.array_equal(q, last_q):
                return last_values
            actual, geometric = self.forward_jacobian(q)
            rotvec = Rotation.from_matrix(target[:3,:3].T @ actual[:3,:3]).as_rotvec()
            theta = np.linalg.norm(rotvec)
            x,y,z = rotvec
            skew = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
            factor = (1/12 + theta*theta/720) if theta < 1e-4 else (1-.5*theta/math.tan(.5*theta))/(theta*theta)
            left_inverse = np.eye(3)-.5*skew+factor*(skew @ skew)
            residual = np.r_[actual[:3,3]-target[:3,3], .12*rotvec]
            derivative = np.r_[geometric[:3], .12*left_inverse @ target[:3,:3].T @ geometric[3:]]
            last_q, last_values = q.copy(), (residual, derivative)
            return last_values
        seed = np.asarray(seed)
        if np.any(seed < self.lower) or np.any(seed > self.upper):
            raise RuntimeError("Joint seed is outside the SDK limits.")
        inner_lower,inner_upper=self.lower+self.encoder_reserve,self.upper-self.encoder_reserve
        guess=np.clip(seed,inner_lower,inner_upper)
        result = least_squares(lambda q: evaluate(q)[0], guess, jac=lambda q:evaluate(q)[1],
                               bounds=(inner_lower, inner_upper), max_nfev=150,
                               ftol=1e-7, xtol=1e-7, gtol=1e-7)
        error = translation_rotation_delta(self.forward(result.x), target)
        if error[0] > 0.002 or error[1] > 1.0:
            raise RuntimeError(f"No nearby IK solution found: solver residual {error[0]*1000:.1f} mm, {error[1]:.1f} deg "
                               "(required at most 2 mm and 1 deg).")
        return result.x

    def check_segment(self, start, end, seed, *, joint_trace=None):
        distance, angle = translation_rotation_delta(start, end)
        count = max(1, int(np.ceil(distance / 0.01)), int(np.ceil(angle / 5)))
        rotations = Slerp([0, 1], Rotation.from_matrix(np.array([start[:3, :3], end[:3, :3]])))
        q = np.asarray(seed)
        for f in np.linspace(0, 1, count + 1)[1:]:
            target = np.eye(4); target[:3, :3] = rotations(f).as_matrix()
            target[:3, 3] = start[:3, 3] * (1-f) + end[:3, 3] * f
            solved = self.inverse(target, q)
            joint_change = float(np.degrees(np.max(np.abs(solved - q))))
            if joint_change > 20:
                raise RuntimeError(f"Local IK path is discontinuous: a joint solution changed {joint_change:.1f} deg "
                                   "between path samples (limit 20 deg). A continuous path was not verified.")
            q = solved
            if joint_trace is not None:
                joint_trace.append(q)
        return q


def support_plane(scene_points, object_points):
    """Fit the locally visible horizontal support, excluding the selected object."""
    lo, hi = object_points.min(axis=0), object_points.max(axis=0)
    outside = np.any((scene_points[:, :2] < lo[:2] - .005) | (scene_points[:, :2] > hi[:2] + .005), axis=1)
    near = np.all((scene_points[:, :2] >= lo[:2] - .20) & (scene_points[:, :2] <= hi[:2] + .20), axis=1)
    points = scene_points[outside & near & (scene_points[:, 2] < hi[2] - .01)]
    if len(points) < 300:
        raise RuntimeError("Too little visible pickup support around the object.")
    hist, bins = np.histogram(points[:, 2], bins=np.arange(points[:, 2].min()-.002, points[:, 2].max()+.004, .004))
    centre = (bins[np.argmax(hist)] + bins[np.argmax(hist)+1]) / 2
    inliers = points[np.abs(points[:, 2] - centre) < .008]
    for _ in range(3):
        a = np.c_[inliers[:, :2], np.ones(len(inliers))]
        plane = np.linalg.lstsq(a, inliers[:, 2], rcond=None)[0]
        errors = np.abs(points[:, 2] - np.c_[points[:, :2], np.ones(len(points))] @ plane)
        inliers = points[errors < .004]
        if len(inliers) < 300:
            raise RuntimeError("Pickup support plane has insufficient consistent depth.")
    if np.linalg.norm(plane[:2]) > np.tan(np.deg2rad(10)):
        raise RuntimeError("The initial pickup mode requires a nearly horizontal support.")
    return plane, inliers


def shifted(matrix, vector):
    result = matrix.copy(); result[:3, 3] += vector
    return result


def destination_geometry(context, object_points, target_object, plane, margin):
    """Check the translated object footprint, excluding the selected object.

    The former enclosing circle included empty corners and unrelated depth
    outside the carried object's footprint. Use its XY convex hull and the
    SAME configured geometry margin as the hand checks. Other geometry remains
    available for the later hand/camera/payload sweep.
    """
    from scipy.spatial import ConvexHull, QhullError, cKDTree
    try:
        hull = ConvexHull(target_object[:, :2])
    except QhullError as exc:
        raise RuntimeError("Object footprint is too degenerate to validate placement.") from exc
    xy = target_object[:, :2]
    near = np.all((context[:, :2] >= xy.min(axis=0)-margin)
                  & (context[:, :2] <= xy.max(axis=0)+margin), axis=1)
    points = context[near]
    equations = hull.equations
    inside = np.all(points[:, :2] @ equations[:, :2].T + equations[:, 2] <= margin, axis=1)
    points = points[inside]
    heights = points[:, 2] - np.c_[points[:, :2], np.ones(len(points))] @ plane
    elevated = points[heights > .012]
    # Object-owned points will move with the arm; they are not stationary
    # obstacles even when the destination overlaps the current object position.
    distances = cKDTree(object_points).query(elevated)[0] if len(elevated) else np.empty(0)
    return {"destination_support_pixels": int(np.count_nonzero(np.abs(heights) < .006)),
            "destination_obstacle_pixels": int(np.count_nonzero(distances > .003)),
            "destination_selected_object_pixels_excluded": int(np.count_nonzero(distances <= .003)),
            "destination_margin_m": float(margin), "destination_region": "translated object XY convex hull"}


def release_target(grasp, obj, context, plane, bottom, workspace, config):
    """Honor a taught air-release TCP; tabletop placement remains explicit opt-in."""
    ec = config["execution"]
    mode = ec.get("placement_mode", "taught_release")
    if mode == "taught_release":
        pose = np.asarray(workspace.drop_pose.tcp_pose, dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise RuntimeError("Taught release pose must contain six finite TCP values.")
        return pose6_to_matrix(pose), {"placement_mode": mode,
            "placement": "full taught TCP release pose above bin; no destination support/occupancy inference"}
    if mode != "tabletop":
        raise ValueError("placement_mode must be taught_release or tabletop.")
    centre = np.median(obj, axis=0)
    destination = np.asarray(workspace.drop_pose.tcp_pose[:2])
    delta_xy = destination - centre[:2]
    delta = np.r_[delta_xy, float(plane[:2] @ delta_xy) - bottom + ec["release_gap_m"]]
    stats = destination_geometry(context, obj, obj+delta, plane, ec["geometry_margin_m"])
    if stats["destination_support_pixels"] < 50:
        raise RuntimeError(f"Destination tabletop is not sufficiently visible: {stats['destination_support_pixels']} "
                           "support pixels in the translated object footprint (need 50).")
    if stats["destination_obstacle_pixels"] > 20:
        raise RuntimeError(f"Destination footprint may be occupied: {stats['destination_obstacle_pixels']} elevated depth pixels "
                           "not belonging to the selected object (limit 20). Inspect the destination depth.")
    return shifted(grasp, delta), {**stats, "placement_mode": mode,
            "placement": "preserve orientation; visible tabletop support at saved drop XY"}


def orientation_metrics(grasp, start):
    approach, closing = grasp[:3, 2], grasp[:3, 1]
    return {"approach_tilt_deg": float(np.degrees(np.arcsin(np.clip(abs(approach[2]), 0, 1)))),
            "closing_axis_tilt_deg": float(np.degrees(np.arcsin(np.clip(abs(closing[2]), 0, 1)))),
            "capture_orientation_change_deg": translation_rotation_delta(start, grasp)[1],
            "downward_alignment": float(np.clip(-approach[2], 0, 1))}


def candidate_preference(scene, row, workspace, config):
    """Only orders existing neural proposals; it never rejects or invents one."""
    capture = json.loads(scene.metadata_json)["execution_capture"]
    start = pose6_to_matrix(capture["pose"])
    grasp = start @ workspace.T_tcp_camera @ piper_pad_tcp(row, config["pad_length_m"])
    metrics = orientation_metrics(grasp, start)
    quality = float(np.arctan(max(0., row[0])) / (np.pi/2))
    return quality + .5*metrics["downward_alignment"]


def plan_preference(plan, row):
    """Rank feasible plans using quality, downward approach and actual joint margin."""
    metrics = json.loads(plan.details_json)
    quality = float(np.arctan(max(0., row[0])) / (np.pi/2))
    margin = min(metrics.get("working_joint_margin_deg", metrics["minimum_joint_margin_deg"])/15, 1.)
    wrist = min(max(abs(metrics.get("pregrasp_j5_deg", metrics["maximum_abs_j5_deg"])),
                    abs(metrics.get("grasp_j5_deg", metrics["maximum_abs_j5_deg"]))) / 70, 1.)
    return quality + .5*metrics["downward_alignment"] + .25*margin - .15*wrist


@dataclass(frozen=True)
class PlanningGeometry:
    scene_id: str
    configuration: str
    obj: np.ndarray
    context: np.ndarray
    plane: np.ndarray
    bottom: float
    clearance: tuple


def prepare_geometry(scene, workspace, config):
    capture = json.loads(scene.metadata_json).get("execution_capture")
    if not capture or capture["frame_signature"] != frame_signature(workspace, config):
        raise RuntimeError("Capture has no matching stationary pose. Freeze a new scene.")
    start = pose6_to_matrix(capture["pose"])
    camera = start @ workspace.T_tcp_camera
    obj, context, _ = selected_point_clouds(scene, config)
    from .swept import transform_points, prepare_scene
    obj = transform_points(obj, camera[:3,:3].T, camera[:3,3])
    context = transform_points(context, camera[:3,:3].T, camera[:3,3])
    plane, _ = support_plane(context, obj)
    bottom = float(np.percentile(obj[:,2]-(obj[:,0]*plane[0]+obj[:,1]*plane[1]+plane[2]), 2))
    if not -.008 <= bottom <= .015:
        raise RuntimeError("Object bottom/pickup support cannot be established from this view.")
    clearance = prepare_scene(start, obj, context, plane, config)
    for a in (obj, context, plane): a.setflags(write=False)
    return PlanningGeometry(scene.scene_id, signature([workspace_signature(workspace,config),config]),
                            obj, context, plane, bottom, clearance)


def build_motion_plan(scene, row, index, workspace, config, *, geometry=None):
    meta = json.loads(scene.metadata_json); capture = meta.get("execution_capture")
    if not capture or capture["frame_signature"] != frame_signature(workspace, config):
        raise RuntimeError("This is a preview-only or differently calibrated scene. Capture a new stationary frame.")
    row = validate_candidates(np.asarray(row).reshape(1, 17))[0]
    if row[1] + config["opening_margin_m"] > config["max_opening_m"]:
        raise RuntimeError("Candidate exceeds the measured gripper opening with clearance.")
    start = pose6_to_matrix(capture["pose"])
    base_camera = start @ workspace.T_tcp_camera
    ec = config["execution"]
    if not np.isfinite(workspace.task_defaults["gripper_force_n"]) or workspace.task_defaults["gripper_force_n"] <= 0:
        raise ValueError("Gripper force must be finite positive newtons.")
    kin = PiperKinematics(workspace.robot, workspace.tcp_offset)
    recovery_q = kin.recovery_target(capture["joints"], ec)
    # Saved observe is a NEW commanded target, so it still has strict limits.
    kin.require_limits(workspace.observe.joint_angles, "Saved observe")
    capture_q = np.asarray(capture["joints"], dtype=float)
    fk_error = translation_rotation_delta(kin.forward(capture_q), start)
    if fk_error[0] > .005 or fk_error[1] > 2:
        raise RuntimeError("Captured joint/TCP feedback does not agree with SDK FK.")
    geometry = prepare_geometry(scene, workspace, config) if geometry is None else geometry
    if geometry.scene_id != scene.scene_id or geometry.configuration != signature([workspace_signature(workspace,config),config]):
        raise RuntimeError("Prepared geometry does not belong to this scene/configuration.")
    obj, context, plane, bottom = geometry.obj, geometry.context, geometry.plane, geometry.bottom
    lower = np.asarray(workspace.task_defaults["workspace_min_m"])
    upper = np.asarray(workspace.task_defaults["workspace_max_m"])
    def require_workspace(poses, name):
        if any(np.any(t[:3, 3] < lower) or np.any(t[:3, 3] > upper) for t in poses):
            raise RuntimeError(f"{name} joint/TCP path exceeds the configured workspace.")
    prefix_stages, prefix_paths = [], []
    route_start, route_q = start, capture_q
    if recovery_q is not None:
        qs, poses = kin.joint_samples(capture_q, recovery_q, recovery=True)
        require_workspace([start]+poses, "Start recovery")
        prefix_stages.append(MotionStage("recover_start", tuple(matrix_to_pose6(poses[-1])),
                                       tuple(recovery_q), "joint", tuple(tuple(q) for q in qs)))
        prefix_paths.append(poses)
        route_start, route_q = poses[-1], recovery_q
    observe_q = np.asarray(workspace.observe.joint_angles)
    observe_t = pose6_to_matrix(workspace.observe.tcp_pose)
    observe_error = translation_rotation_delta(kin.forward(observe_q), observe_t)
    if observe_error[0] > .005 or observe_error[1] > 2:
        raise RuntimeError("Saved observe TCP and joints disagree with SDK FK. Re-teach that pose.")
    alternatives = [base_camera @ piper_pad_tcp(row, config["pad_length_m"], swap_jaws=swap)
                    for swap in [False, True]]
    alternatives.sort(key=lambda t: translation_rotation_delta(start, t)[1])
    errors = []
    from .swept import check_gripper_paths, check_open_path
    for grasp in alternatives:
        approach = grasp[:3, 2]
        place, placement_stats = release_target(grasp, obj, context, plane, bottom, workspace, config)
        retreat = max(ec["approach_m"], config["pad_length_m"] + .02)
        pregrasp = shifted(grasp, -approach * retreat)
        clear = shifted(place, -place[:3, 2] * retreat)
        targets = [("grasp", grasp), ("lift", shifted(grasp, [0, 0, ec["lift_m"]]))]
        # A taught air-release is already above the bin. Avoid an artificial
        # extra 10cm hover and simultaneous straight-line reorientation.
        if ec.get("placement_mode", "taught_release") == "tabletop":
            targets.append(("above_place", shifted(place, [0, 0, ec["lift_m"]])))
        targets += [("place", place), ("clear", clear),
                    ("clear_up", shifted(clear, [0, 0, ec.get("release_clearance_m", .05)]))]
        try:
            require_workspace([t for name,t in targets if name!='lift']+[observe_t], "Endpoint")
            # Solve the actual grasp, then trace its insertion backwards.
            # This preserves a continuous nearby IK branch at contact instead
            # of solving a distant pregrasp from a folded viewing-pose seed.
            grasp_solutions = kin.pregrasp_solutions(grasp, route_q, observe_q,label='grasp endpoint')
            solutions = []
            nearest=float(np.min((obj-grasp[:3,3]) @ approach))
            minimum_entry=min(retreat,max(.04,config['pad_length_m']/2-nearest+ec['geometry_margin_m']))
            for grasp_q in grasp_solutions:
                try:
                    pre_q,entry_pose,entry_distance=kin.reachable_prefix(grasp,pregrasp,grasp_q,minimum_entry)
                    solutions.append((pre_q,entry_distance))
                except RuntimeError as exc:
                    errors.append(f"pregrasp insertion: {exc}")
            if not solutions:
                continue
        except RuntimeError as exc:
            errors.append("grasp endpoint: " + str(exc)); continue
        for pre_q,entry_distance in solutions:
            try:
                stages, paths = list(prefix_stages), list(prefix_paths)
                trace = [route_q]
                qs, poses = kin.joint_samples(route_q, pre_q)
                require_workspace([route_start]+poses, "pregrasp")
                trace.extend(qs)
                stages.append(MotionStage("pregrasp", tuple(matrix_to_pose6(poses[-1])), tuple(pre_q),
                                          "joint", tuple(tuple(q) for q in qs)))
                paths.append(poses)
                q, previous = pre_q, poses[-1]
                for name, target in targets:
                    try:
                        if name == "place" and ec.get("placement_mode", "taught_release") == "taught_release":
                            choices = kin.pregrasp_solutions(target, q, observe_q,label='taught release')
                            q = choices[0]
                            qs, carried_path = kin.joint_samples(stages[-1].joint_solution, q)
                            require_workspace(carried_path, "Carry to release")
                            trace.extend(qs)
                            stages.append(MotionStage(name,tuple(matrix_to_pose6(target)),tuple(q),
                                                      "joint",tuple(tuple(v) for v in qs)))
                            paths.append(carried_path)
                            previous = carried_path[-1]
                            continue
                        if name=='lift':
                            q,target,lift_distance=kin.reachable_prefix(previous,target,q,
                                min(ec['lift_m'],ec.get('minimum_lift_m',.03)),joint_trace=trace)
                            require_workspace([target],'Lift')
                        else:
                            q = kin.check_segment(previous, target, q, joint_trace=trace)
                    except RuntimeError as exc:
                        raise RuntimeError(f"{name}: {exc}") from exc
                    stages.append(MotionStage(name, tuple(matrix_to_pose6(target)), tuple(q)))
                    paths.append(None)
                    previous = target
                qs, poses = kin.joint_samples(q, observe_q)
                require_workspace(poses, "Return")
                trace.extend(qs)
                stages.append(MotionStage("observe", tuple(matrix_to_pose6(poses[-1])), tuple(observe_q),
                                          "joint", tuple(tuple(q) for q in qs)))
                paths.append(poses)
                matrices = [(s.name, pose6_to_matrix(s.tcp_pose)) for s in stages]
                check_gripper_paths(start, matrices, base_camera, row, obj, context, plane, config,
                                    segment_paths=paths, prepared=geometry.clearance)
                # The open approach route is reversible while no object has
                # been closed on. Its in-limit viewing endpoint also works
                # when the original capture required a small inward recovery.
                view_qs, view_poses = kin.joint_samples(pre_q, route_q)
                require_workspace(view_poses,"Return to view")
                check_open_path([kin.forward(pre_q)]+view_poses,geometry.clearance[0],workspace.T_tcp_camera,
                                config["max_opening_m"],config,"return_view",tree=geometry.clearance[4])
                return_view = MotionStage("return_view",tuple(matrix_to_pose6(view_poses[-1])),tuple(route_q),
                                          "joint",tuple(tuple(v) for v in view_qs))
                trace = np.asarray(trace)
                margin = np.min(np.minimum(trace-kin.lower, kin.upper-trace))
                working = np.asarray([s.joint_solution for s in stages if s.name in {"pregrasp", "grasp", "lift"}])
                working_margin = np.min(np.minimum(working-kin.lower, kin.upper-working))
                metrics = orientation_metrics(grasp, start)
                details = {"pickup_table_z_ax_by_c": plane.tolist(),
                           "object_bottom_above_table_m": float(bottom), **placement_stats, **metrics,
                           "object_transfer_transform": (place @ np.linalg.inv(grasp)).tolist(),
                           "minimum_joint_margin_deg": float(np.degrees(margin)),
                           "working_joint_margin_deg": float(np.degrees(working_margin)),
                           "ik_goal_reserve_deg":float(np.degrees(kin.encoder_reserve)),
                           "maximum_abs_j5_deg": float(np.degrees(np.max(np.abs(trace[:, 4])))),
                           "pregrasp_j5_deg": float(np.degrees(pre_q[4])),
                           "grasp_j5_deg": float(np.degrees(stages[len(prefix_stages)+1].joint_solution[4])),
                           "entry_distance_m":entry_distance,"lift_distance_m":lift_distance,
                           "start_recovery": {"required": recovery_q is not None,
                               "measured_joints_deg": np.degrees(capture_q).tolist(),
                               "target_joints_deg": None if recovery_q is None else np.degrees(recovery_q).tolist()},
                           "orientation_policy": "soft preference for downward approaches; no family/capture-angle veto",
                           "reachability": "SDK MDH analytic-Jacobian IK; grasp-seeded entry; sampled joint transfers; linear insertion; no full-arm collision model",
                           "jaw_mapping": "CAD axes plus calibrated pad midpoint; observed open-jaw verification required"}
                plan = GraspMotionPlan(scene.scene_id, index, tuple(stages), tuple(capture["pose"]),
                       tuple(capture["joints"]), tuple(workspace.tcp_offset), frame_signature(workspace, config),
                       workspace_signature(workspace, config), capture["wall_s"],
                       float(workspace.task_defaults["gripper_force_n"]), config["max_opening_m"],
                       json.dumps(details), capture_monotonic_s=capture.get("monotonic_s"),return_view=return_view)
                return plan
            except RuntimeError as exc:
                errors.append(str(exc))
    raise RuntimeError(f"No checked route ({len(errors)} configurations): " + " | ".join(errors[-3:]))

