"""Bounded sequence through PiperRobotClient; no alternate controller/fallback."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import threading
import time

import numpy as np
from piper_app.calibration.transforms import pose6_to_matrix, translation_rotation_delta
from .planning import telemetry_sample, PiperKinematics, samples_stable, settled_feedback


def scene_unchanged(scene, bundle):
    if bundle is None or bundle.received_monotonic_s is None or time.monotonic() - bundle.received_monotonic_s > .5:
        raise RuntimeError("No fresh camera frame for the final scene check.")
    if bundle.serial != json.loads(scene.metadata_json)["serial"] or bundle.depth_m.shape != scene.depth_m.shape:
        raise RuntimeError("Camera changed since planning.")
    valid = scene.mask & np.isfinite(scene.depth_m) & (scene.depth_m > 0)
    if not np.any(valid):
        raise RuntimeError("No original depth for the final scene check.")
    agreement = np.mean(np.isfinite(bundle.depth_m[valid]) & (np.abs(bundle.depth_m[valid] - scene.depth_m[valid]) < .005))
    rgb_error = float(np.mean(np.abs(bundle.color_rgb[scene.mask].astype(float) - scene.rgb[scene.mask].astype(float))))
    if agreement < .85 or rgb_error > 15:
        raise RuntimeError("Target scene changed since planning. Resume and freeze again.")


class GraspExecutor:
    def __init__(self, client, config):
        self.client = client
        self.config = json.loads(json.dumps(config))
        self.state = "idle"
        self.plan_token = None
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._pregrasp_reached_s = None

    def stop(self):
        self._cancel.set()
        self.state = "failed"
        self.client.electronic_emergency_stop()

    def _check_cancel(self):
        if self._cancel.is_set():
            raise RuntimeError("Execution stopped; no further stages were sent.")

    def _near(self, actual, target):
        error = translation_rotation_delta(pose6_to_matrix(actual), pose6_to_matrix(target))
        ec = self.config["execution"]
        return error[0] <= ec["pose_translation_tol_m"] and error[1] <= ec["pose_rotation_tol_deg"]

    def _move(self, stage):
        self._check_cancel()
        self._kin.require_limits(stage.joint_solution, f"Planned {stage.name}")
        if stage.motion_kind == "joint":
            self.client.robot.move_j(list(stage.joint_solution))
        elif stage.motion_kind == "linear":
            flange = self.client.get_tcp2flange_pose(list(stage.tcp_pose))
            self.client.robot.move_l(flange)
        else:
            raise RuntimeError("Unknown planned motion kind.")
        ec = self.config["execution"]; deadline = time.monotonic() + ec["timeout_s"]
        settled = []
        while time.monotonic() < deadline:
            self._check_cancel()
            sample = telemetry_sample(self.client, require_stationary=False)
            if stage.name == "recover_start":
                q = np.asarray(sample["joints"])
                overrun = np.maximum(np.maximum(self._kin.lower-q, q-self._kin.upper), 0)
                initial = np.asarray(self._execution_start_joints)
                initial_overrun = np.maximum(np.maximum(self._kin.lower-initial, initial-self._kin.upper), 0)
                if np.any(overrun > initial_overrun + np.deg2rad(.1)):
                    raise RuntimeError("Start recovery moved farther outside the measured joint envelope.")
            if self._near(sample["tcp_pose"], stage.tcp_pose):
                tolerance = np.deg2rad(.1 if stage.name == "recover_start" else .5) if stage.motion_kind == "joint" else .20
                if stage.motion_kind == "linear" and np.max(np.abs(np.asarray(sample["joints"]) - stage.joint_solution)) > tolerance:
                    raise RuntimeError(f"Unexpected IK branch at {stage.name}.")
                if np.max(np.abs(np.asarray(sample["joints"]) - stage.joint_solution)) > tolerance:
                    settled = []
                    time.sleep(.08)
                    continue
                if stage.name == "recover_start":
                    self._kin.require_limits(sample["joints"], "Recovery feedback")
                settled.append(sample); settled = settled[-3:]
                if len(settled) >= 3 and samples_stable(settled,translation_m=.0005,rotation_deg=.25,joint_rad=np.deg2rad(.1)):
                    return
            else:
                settled = []
            time.sleep(.08)
        raise RuntimeError(f"{stage.name} did not settle. No fallback or return was sent.")

    def _gripper(self, command, *, require_open=False, require_contact=False):
        self._check_cancel()
        gs = self.client.get_gripper_status()
        if gs is None or not gs.msg.foc_status.driver_enable_status or gs.msg.foc_status.driver_error_status:
            raise RuntimeError("Gripper feedback is missing, disabled or reports an error.")
        self.client.move_gripper_width(command, force_n=self._force, settle_s=.8)
        gs = self.client.get_gripper_status()
        if gs is None or gs.msg.foc_status.driver_error_status or time.time() - gs.timestamp > .25:
            raise RuntimeError("Gripper did not return fresh healthy feedback.")
        if require_open and float(gs.msg.value) < command - .003:
            raise RuntimeError("Gripper did not reach its open command; withdrawal/pickup stopped.")
        if require_contact and float(gs.msg.value) <= .001:
            raise RuntimeError("Gripper closed empty; lift was not sent.")

    def run(self, plan, *, approach_only, dry_run, mapping_verified, log_path, progress=lambda value: None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Another grasp execution is running.")
        token = (plan.scene_id, plan.candidate_index, plan.config_signature)
        records = []
        def event(name):
            records.append({"stage": name, "wall_s": time.time()}); progress(name)
        try:
            self._cancel.clear()
            pregrasp_index = plan.pregrasp_index
            if self.state == "failed":
                raise RuntimeError("Previous attempt failed. Inspect/recover the arm before restarting this demo.")
            if not dry_run and not approach_only and not mapping_verified:
                raise RuntimeError("Run Approach Only and verify the observed jaw alignment before the first pickup.")
            if self.state == "pregrasp" and self.plan_token != token:
                raise RuntimeError("The approached plan changed. Return/reobserve before planning another grasp.")
            if dry_run:
                for stage in plan.stages[:pregrasp_index+1] if approach_only else plan.stages:
                    event("dry-run: " + stage.name)
                return
            if plan.capture_monotonic_s is None:
                raise RuntimeError("Plan has no steady-clock capture age. Resume and freeze a new frame.")
            if self.state == "pregrasp":
                if self._pregrasp_reached_s is None:
                    raise RuntimeError("Approach confirmation time is missing. Use Return to View.")
                age = time.monotonic()-self._pregrasp_reached_s
                if age < 0 or age > self.config['execution'].get('pregrasp_confirmation_max_age_s',60):
                    raise RuntimeError("Inspection pause expired. Use Return to View, then freeze a new scene.")
            else:
                age = time.monotonic() - plan.capture_monotonic_s
                if age < 0 or age > self.config["execution"]["snapshot_max_age_s"]:
                    raise RuntimeError("Plan expired before approach. Resume and freeze a new scene.")
            sample = settled_feedback(self.client)
            self._execution_start_joints = tuple(sample["joints"])
            expected_stage = plan.stages[pregrasp_index] if self.state == "pregrasp" else None
            expected = expected_stage.tcp_pose if expected_stage else plan.start_pose
            if not self._near(sample["tcp_pose"], expected):
                raise RuntimeError("Robot moved away from the plan's expected starting pose.")
            if not np.allclose(self.client.config.tcp_offset, plan.tcp_offset, atol=1e-8):
                raise RuntimeError("Robot TCP changed after planning.")
            expected_joints = expected_stage.joint_solution if expected_stage else plan.start_joints
            if np.max(np.abs(np.asarray(sample["joints"])-expected_joints)) > np.deg2rad(.25):
                raise RuntimeError("Joint feedback changed since the checked starting pose. Freeze and plan again.")
            self._kin = PiperKinematics(getattr(self.client.config, "robot", "piper"), plan.tcp_offset)
            actual_recovery = self._kin.recovery_target(sample["joints"], self.config["execution"])
            if actual_recovery is not None and (self.state == "pregrasp" or not any(s.name == "recover_start" for s in plan.stages[:pregrasp_index])):
                raise RuntimeError("Current joints need an unplanned recovery. Freeze and plan again.")
            for stage in plan.stages:
                self._kin.require_limits(stage.joint_solution, f"Planned {stage.name}")
            teaching = self.client.get_gripper_teaching_pendant_param(timeout=.5, min_interval=0)
            if teaching is None:
                raise RuntimeError("SDK gripper maximum command is unavailable.")
            full_open = float(teaching.msg.max_range_config)
            if not np.isfinite(full_open) or not .01 <= full_open <= .15:
                raise RuntimeError("SDK gripper maximum command is invalid.")
            self._force = plan.force_n
            self._check_cancel()
            if not self.client.enable_and_wait(timeout=3):
                raise RuntimeError("Arm enable did not complete.")
            self._check_cancel()
            self.client.set_speed_percent(self.config["execution"]["speed_percent"])
            self.plan_token = token
            try:
                if self.state != "pregrasp":
                    event("open"); self._gripper(full_open, require_open=True)
                    for stage in plan.stages[:pregrasp_index+1]:
                        event(stage.name); self._move(stage); self._check_cancel()
                    self.state = "pregrasp"
                    self._pregrasp_reached_s = time.monotonic()
                if approach_only:
                    event("approach complete; jaws remain open")
                    return
                for stage in plan.stages[pregrasp_index+1:]:
                    event(stage.name); self._move(stage); self._check_cancel()
                    if stage.name == "grasp":
                        event("close"); self._gripper(0., require_contact=True)
                    elif stage.name == "place":
                        event("release"); self._gripper(full_open, require_open=True)
                self._check_cancel(); self.state = "idle"; self.plan_token = None; self._pregrasp_reached_s = None
                event("sequence complete; observe settled; inspect physical outcome")
            except Exception:
                self.state = "failed"
                # A feedback/timeout failure must also stop a motion already
                # accepted by the controller, not just suppress later stages.
                try:
                    self.client.electronic_emergency_stop()
                except Exception:
                    pass
                raise
        except Exception as exc:
            if self._cancel.is_set():
                self.state = "failed"
                self.client.electronic_emergency_stop()
            event("failed: " + str(exc))
            raise
        finally:
            try:
                Path(log_path).write_text(json.dumps({"plan": asdict(plan), "dry_run": dry_run,
                                                    "events": records, "state": self.state}, indent=2))
            finally:
                self._lock.release()

    def return_to_view(self, plan, *, dry_run, log_path, progress=lambda text:None):
        """Operator-requested cancellation of an OPEN approach; no automatic motion on expiry."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Another grasp execution is running.")
        events=[]
        started=False
        try:
            if self.state != 'pregrasp' or self.plan_token != (plan.scene_id,plan.candidate_index,plan.config_signature):
                raise RuntimeError("Return to View requires the same completed open approach.")
            if plan.return_view is None:
                raise RuntimeError("This older plan has no checked return-to-view route.")
            if dry_run:
                progress('dry-run: return_view'); return
            sample=settled_feedback(self.client)
            if not np.allclose(self.client.config.tcp_offset,plan.tcp_offset,atol=1e-8):
                raise RuntimeError('TCP configuration changed; return route cannot be reused.')
            pre=plan.stages[plan.pregrasp_index]
            if not self._near(sample['tcp_pose'],pre.tcp_pose) or np.max(np.abs(np.asarray(sample['joints'])-pre.joint_solution))>np.deg2rad(.25):
                raise RuntimeError("Arm moved away from the checked open pregrasp; return route cannot be reused.")
            self._cancel.clear()
            self._kin=PiperKinematics(getattr(self.client.config,'robot','piper'),plan.tcp_offset)
            teaching=self.client.get_gripper_teaching_pendant_param(timeout=.5,min_interval=0)
            if teaching is None or not .01<=float(teaching.msg.max_range_config)<=.15:
                raise RuntimeError('Gripper opening command unavailable.')
            self._force=plan.force_n
            started=True
            self._gripper(float(teaching.msg.max_range_config),require_open=True)
            self.client.set_speed_percent(self.config['execution']['speed_percent'])
            events.append('return_view');progress('return_view')
            self._move(plan.return_view)
            self.state='idle';self.plan_token=None;self._pregrasp_reached_s=None
        except Exception:
            if started:
                self.state='failed'
                try:self.client.electronic_emergency_stop()
                except Exception:pass
            raise
        finally:
            try:
                Path(log_path).write_text(json.dumps({'action':'return_view','events':events,'state':self.state,'dry_run':dry_run},indent=2))
            finally:self._lock.release()
