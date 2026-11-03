"""Restricted GraspNet pickup GUI, sharing the preview and existing Piper client."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import queue
import threading
import time
from collections import deque, Counter
import tkinter as tk
from tkinter import ttk

from piper_app.config import repo_root
from piper_app.graspnet.gui import GraspNetTargetPreviewGui
from piper_app.graspnet.planning import (build_motion_plan, telemetry_sample, qualify_capture,
                                        frame_signature, workspace_signature, PiperKinematics,
                                        candidate_preference, prepare_geometry, samples_stable)
from piper_app.calibration.transforms import pose6_to_matrix, translation_rotation_delta
from piper_app.graspnet.execution import GraspExecutor, scene_unchanged
from piper_app.yolo_target_pick_demo.gui import YoloTargetPickDemoGuiApp


def rejection_category(reason):
    text = reason.lower()
    for token, category in [
        ("small-start recovery", "start recovery"),
        ("envelope", "hand/camera clearance"), ("carried object", "object clearance"),
        ("side-grasp family", "orientation policy"), ("wrist rotation", "wrist rotation policy"),
        ("ik path is discontinuous", "IK path continuity"), ("nearby ik solution", "IK solver"),
        ("outside sdk limits", "joint limits"), ("destination tabletop", "destination visibility"),
        ("footprint may be occupied", "destination depth"), ("envelope", "hand/camera clearance"),
        ("carried object", "object clearance"), ("support", "support geometry"),
        ("tabletop", "support geometry"), ("workspace", "workspace bounds"), ("sdk fk", "feedback consistency")]:
        if token in text:
            return category
    return "other validation"


class GraspNetTargetPickGui(GraspNetTargetPreviewGui):
    def __init__(self, root, args):
        self._motion_plan = None
        self._motion_results = queue.Queue()
        self._capture_before = self._live_capture = self._frozen_capture = None
        self._executor = None
        self._motion_generation = None
        self._stationary_history = deque(maxlen=30)
        super().__init__(root, args)
        self.root.title("Piper GraspNet Pick — supervised grasp trial")
        if args.speed_percent is not None:
            if not 1 <= args.speed_percent <= 20:
                raise ValueError("Initial grasp speed must be 1–20 percent.")
            self._grasp_client.config['execution']['speed_percent'] = int(args.speed_percent)
        self._executor = GraspExecutor(self.client, self._grasp_client.config)

    def _build_action_panel(self, parent, row):
        YoloTargetPickDemoGuiApp._build_action_panel(self, parent, row)
        frame = next(w for w in parent.winfo_children() if isinstance(w, ttk.LabelFrame) and w.cget("text") == "Actions")
        for w in frame.winfo_children():
            if isinstance(w, ttk.Button):
                if w.cget("text") == "Move Observe":
                    w.configure(text="Return to View",command=lambda:self.run_action(self._return_to_view))
                elif w.cget("text") in {"Move Home", "Open Gripper", "Close Gripper"}:
                    w.configure(state="disabled")
                elif w.cget("text") == "Execute Pick":
                    w.configure(text="Plan Grasp", command=lambda: self.run_action(self.plan_grasp))
            elif isinstance(w, ttk.Label) and str(w.cget("text")).startswith(("Flow:", "Pickup uses")):
                w.configure(text="Category → Freeze → Plan Grasp. First trial: Approach Only, check jaws, then Execute Pick.")
        ttk.Label(frame, textvariable=self.grasp_status_var, wraplength=490, justify="left").grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Button(frame, text="Next Grasp", command=lambda: self.run_action(self.next_grasp)).grid(row=6, column=0, sticky="ew")
        ttk.Button(frame, text="Approach Only", command=lambda: self.run_action(lambda: self._start_execution(True))).grid(row=6, column=1, sticky="ew")
        ttk.Button(frame, text="Jaws Aligned", command=lambda: self.run_action(self.verify_alignment)).grid(row=6, column=2, sticky="ew")
        ttk.Button(frame, text="Execute Pick", command=lambda: self.run_action(lambda: self._start_execution(False))).grid(row=6, column=3, sticky="ew")
        ttk.Button(frame, text="Stop motion", command=lambda: self.run_action(self.stop_motion)).grid(row=7, column=3, sticky="ew", pady=(6, 0))

    def build_note_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Initial GraspNet pickup", padding=10)
        frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(frame, text=("Start with an isolated opaque block and your taught release pose above the bin. "
            "Dry-run is on initially. The full saved release pose, including height and orientation, is used. "
            "Downward grasps are preferred when feasible. A small resting-joint correction may precede the approach after you start a live action. "
            "Planning stops at the first checked solution. Return to View cancels an open approach. The first live Approach Only leaves the jaws open before contact; check that their direction/centre match the preview, "
            "then confirm Jaws Aligned only if this setup has not been verified. Execute Pick closes, lifts, places, opens, withdraws along the approach and rises before returning to observe. "
            "Scene checks cover a simplified hand/camera and visible geometry, not full-arm collision avoidance."),
            wraplength=1500, justify="left").grid(sticky="w")

    def connect_robot(self):
        if self._action_running:
            raise RuntimeError("An arm action is running.")
        if self.workspace is None:
            raise RuntimeError("Load the task workspace first.")
        if self.robot_var.get() != self.workspace.robot:
            raise RuntimeError("Robot model must match the loaded task workspace.")
        self.client.config.tcp_offset = list(self.workspace.tcp_offset)
        self.client.config.robot = self.robot_var.get()
        self.client.config.interface = self.interface_var.get()
        self.client.config.channel = self.channel_var.get()
        self.client.config.bitrate = int(self.bitrate_var.get())
        self.client.connect(configure_robot=False, init_gripper=True)
        self._connected = True; self._firmware_info = self.client.firmware_info
        self._last_event_text = "Connected for feedback. Motion starts only through a live action."
        self._poll_data()

    def disconnect_robot(self):
        if self._action_running:
            raise RuntimeError("Stop/finish the running arm action before disconnecting.")
        self._invalidate_grasp()
        super().disconnect_robot()

    def on_robot_polled(self):
        self.observe_ready_text = "Measured joint/TCP stability required for capture"
        gs = self.client.get_gripper_status()
        if gs is not None:
            self._gripper_value_m = float(gs.msg.value)
            self._gripper_driver_enabled = bool(gs.msg.foc_status.driver_enable_status)
            self._gripper_homing = bool(gs.msg.foc_status.homing_status)
            self._gripper_driver_error = bool(gs.msg.foc_status.driver_error_status)

    def _update_camera_frame(self):
        self._capture_before = None
        if self._connected and not self._action_running:
            try:
                self._capture_before = telemetry_sample(self.client)
            except RuntimeError:
                pass
        super()._update_camera_frame()

    def on_camera_bundle(self, bundle):
        after = None
        try:
            after = telemetry_sample(self.client) if self._connected else None
            if after is None:
                raise RuntimeError("Robot feedback is required for execution-qualified capture.")
            self._stationary_history.append(after)
            history = [p for p in self._stationary_history if after['monotonic_s'] - p['monotonic_s'] <= 3.0]
            if not history or after['monotonic_s'] - history[0]['monotonic_s'] < .5:
                raise RuntimeError("Wait for at least half a second of stationary camera/arm observations, then freeze.")
            for previous in history:
                if not samples_stable([previous,after]):
                    self._stationary_history.clear()
                    raise RuntimeError("Arm has not settled. Wait, resume and freeze again.")
            proof = qualify_capture(bundle, self._capture_before, after, self.workspace, self._grasp_client.config)
            self._live_capture = (bundle.frame_number, proof, None)
        except Exception as exc:
            if after is None or self._action_running:
                self._stationary_history.clear()
            self._live_capture = (bundle.frame_number, None, str(exc))
        super().on_camera_bundle(bundle)

    def toggle_pause(self):
        if self._action_running or (self._executor and self._executor.state == "pregrasp"):
            return
        if not self._paused:
            try:
                self._apply_target_category()
                try:
                    self._capture_before = telemetry_sample(self.client) if self._connected else None
                except RuntimeError:
                    self._capture_before = None
                bundle = self._camera.read_frames(fresh=True)
                if bundle is None:
                    raise RuntimeError("Camera did not deliver a new RGB-D frame.")
                self._camera_bundle = bundle
                # Infer on this exact fresh image before freezing its mask.
                self.on_camera_bundle(bundle)
            except Exception as exc:
                self._last_event_text = f"Cannot acquire a fresh planning frame: {exc}"
                self.grasp_status_var.set(str(exc))
                return
        super().toggle_pause()
        if self._paused and self._frozen_bundle is not None:
            self._frozen_capture = self._live_capture

    def _capture_scene(self):
        scene = super()._capture_scene()
        proof = self._frozen_capture
        meta = json.loads(scene.metadata_json)
        if proof and proof[0] == self._frozen_bundle.frame_number and proof[1] is not None:
            meta["execution_capture"] = proof[1]
        else:
            meta["execution_capture_error"] = proof[2] if proof else "No robot/camera pose bracket."
        return replace(scene, metadata_json=json.dumps(meta))

    def _invalidate_grasp(self):
        if self._action_running or (self._executor and self._executor.state == "pregrasp"):
            raise RuntimeError("Complete the approached plan before replacing its target/configuration.")
        self._motion_plan = None
        if self._motion_generation is not None:
            self._planning = False
            self._motion_generation = None
        super()._invalidate_grasp()

    def plan_grasp(self):
        if self._action_running or (self._executor and self._executor.state != "idle"):
            raise RuntimeError("Finish/recover the current arm attempt before planning again.")
        self._motion_plan = None
        super().plan_grasp()

    def _accept_result(self, generation, scene, result, error):
        super()._accept_result(generation, scene, result, error)
        if not error and self._grasp_result is result and len(result.candidates):
            self._build_motion_plans(list(range(len(result.candidates))))

    def _build_motion_plans(self, indices):
        scene, result = self._grasp_scene, self._grasp_result
        capture = json.loads(scene.metadata_json)
        if "execution_capture" not in capture:
            self.grasp_status_var.set("Preview only: " + capture.get("execution_capture_error", "capture has no bound pose"))
            return
        self._planning = True; self._motion_plan = None
        generation = self._grasp_generation
        self._motion_generation = generation
        workspace, config = copy.deepcopy(self.workspace), copy.deepcopy(self._grasp_client.config)
        self.grasp_status_var.set("Checking grasp candidates and complete movement paths…")
        def worker():
            reasons = []; plan = None; checked = []; began=time.monotonic()
            try:
                PiperKinematics(workspace.robot, workspace.tcp_offset).require_limits(workspace.observe.joint_angles, "Saved observe")
                geometry=prepare_geometry(scene,workspace,config)
            except Exception as exc:
                reasons.append({"candidate": "workspace", "reason": str(exc)})
                indices_to_check = []
            else:
                ranked = sorted(indices,
                    key=lambda i: candidate_preference(scene, result.candidates[i], workspace, config), reverse=True)
                # Respect the proposal already displayed, then search the
                # rest automatically. The second batch preserves alternatives.
                indices_to_check=([indices[0]]+[i for i in ranked if i!=indices[0]]) if indices else []
            for index in indices_to_check:
                if self._closed or generation != self._grasp_generation:
                    break
                try:
                    checked.append(index)
                    batch=config['execution'].get('initial_candidates',20)
                    phase='automatic fallback' if len(checked)>batch else 'initial search'
                    message=f"{phase}: checking proposal {index+1}/{len(result.candidates)}; {len(checked)} checked, {time.monotonic()-began:.1f}s"
                    self._motion_results.put(('planning_progress',generation,message,None))
                    plan = build_motion_plan(scene,result.candidates[index],index,workspace,config,geometry=geometry)
                    break
                except Exception as exc:
                    reasons.append({"candidate": index + 1, "reason": str(exc)})
            folder = Path(result.replay_dir)
            try:
                (folder/"motion_rejections.json").write_text(json.dumps(reasons, indent=2))
                # Next Grasp validates one candidate; preserve earlier runs
                # instead of leaving only the most recent rejection on disk.
                with (folder/"motion_validation_history.jsonl").open("a") as history:
                    history.write(json.dumps({"wall_s": time.time(), "candidate_indices": checked,
                                              "accepted_candidate": None if plan is None else plan.candidate_index,
                                              "planning_seconds":time.monotonic()-began,
                                              "selection":"first complete checked plan; automatic second batch when needed",
                                              "reasons": reasons}) + "\n")
                if plan is not None:
                    plan.save(folder/"motion_plan.json")
            except OSError as exc:
                plan = None
                reasons.append({"candidate": "save", "reason": f"Could not save checked plan: {exc}"})
            self._motion_results.put(("plan", generation, plan, reasons))
        threading.Thread(target=worker, name="grasp-motion-planning", daemon=True).start()

    def _drain_results(self):
        if self._closed:
            return
        try:
            while True:
                kind, generation, value, detail = self._motion_results.get_nowait()
                if kind == "progress":
                    self._last_event_text = value
                elif kind == 'planning_progress' and generation==self._grasp_generation:
                    self.grasp_status_var.set(value)
                elif kind == "execution":
                    self._action_running = False
                    self._last_event_text = value
                    self.grasp_status_var.set(value)
                    if self._executor.state == "idle" and not detail:
                        self._motion_plan = None
                elif generation == self._grasp_generation:
                    self._planning = False; self._motion_plan = value; self._motion_generation = None
                    if value:
                        self._grasp_index = value.candidate_index
                        self._update_grasp_status()
                        details = json.loads(value.details_json)
                        recovery = " | small start recovery planned" if details["start_recovery"]["required"] else ""
                        self.grasp_status_var.set(self.grasp_status_var.get() +
                            f" | ready: grasp J5 {details['grasp_j5_deg']:.1f}°" + recovery)
                    else:
                        reason = detail[-1]["reason"] if detail else "No candidates."
                        candidate = detail[-1]['candidate'] if detail else '--'
                        self.grasp_status_var.set(f"No complete plan. Last refusal was proposal {candidate}: " + reason)
                        counts=Counter(rejection_category(item['reason']) for item in detail)
                        summary='; '.join(f'{count} × {text}' for text,count in counts.most_common(3))
                        self._last_event_text = "Rejection summary: " + summary + ". Full details saved in motion_rejections.json."
        except queue.Empty:
            pass
        super()._drain_results()

    def _update_grasp_status(self):
        super()._update_grasp_status()
        self.grasp_status_var.set(self.grasp_status_var.get().replace("preview only", "restricted task"))

    def next_grasp(self):
        if self._planning or self._action_running or (self._executor and self._executor.state != "idle"):
            return
        super().next_grasp()
        if self._grasp_result is not None and len(self._grasp_result.candidates):
            self._build_motion_plans([self._grasp_index])

    def _verification_path(self):
        return repo_root() / self._grasp_client.config["execution"]["frame_verification_path"]

    def _mapping_verified(self):
        try:
            value = json.loads(self._verification_path().read_text())
            return isinstance(value, dict) and value.get("frame_signature") == frame_signature(self.workspace, self._grasp_client.config)
        except (OSError, ValueError):
            return False

    def verify_alignment(self):
        if self._action_running or self._executor.state != "pregrasp" or self._motion_plan is None:
            raise RuntimeError("First finish a LIVE Approach Only and observe the open jaws.")
        sample = telemetry_sample(self.client)
        if not self._executor._near(sample["tcp_pose"], self._motion_plan.stages[self._motion_plan.pregrasp_index].tcp_pose):
            raise RuntimeError("Arm is no longer at the checked pregrasp.")
        path = self._verification_path(); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"frame_signature": self._motion_plan.frame_signature,
            "scene_id": self._motion_plan.scene_id, "observed_pregrasp_pose": sample["tcp_pose"],
            "verified_wall_s": time.time(), "verification": "operator pressed Jaws Aligned after live open-jaw approach"}, indent=2))
        self._last_event_text = "Observed jaw alignment recorded for this TCP/hand-eye/pad geometry. Execute Pick is available."

    def _start_execution(self, approach_only):
        if self._planning or self._action_running or self._motion_plan is None:
            raise RuntimeError("Wait for a checked motion plan before executing.")
        plan = self._motion_plan
        if (self._grasp_scene is None or self._grasp_result is None
                or plan.scene_id != self._grasp_scene.scene_id or plan.scene_id != self._grasp_result.scene_id
                or plan.candidate_index != self._grasp_index):
            raise RuntimeError("Motion plan does not belong to the displayed scene/candidate. Replan.")
        if plan.config_signature != workspace_signature(self.workspace, self._grasp_client.config):
            raise RuntimeError("Workspace/configuration changed; replan.")
        dry_run = bool(self.dry_run_var.get())
        if not dry_run and self._executor.state == "idle":
            current = telemetry_sample(self.client)
            if not self._executor._near(current["tcp_pose"], plan.start_pose):
                raise RuntimeError("Arm moved since scene capture. Resume and freeze again.")
            scene_unchanged(self._grasp_scene, self._camera_bundle)
        verified = self._mapping_verified()
        folder = Path(self._grasp_result.replay_dir)
        log = folder / f"attempt-{time.time_ns()}.json"
        self._action_running = True
        def worker():
            try:
                self._executor.run(plan, approach_only=approach_only, dry_run=dry_run, mapping_verified=verified,
                    log_path=log, progress=lambda text: self._motion_results.put(("progress", 0, text, None)))
                message = "Dry-run complete; no motion." if dry_run else (f"Approach reached; jaws open. Execute within {self._executor.config['execution'].get('pregrasp_confirmation_max_age_s',60):g}s, or Return to View." if approach_only
                          else "Sequence reached observe. Inspect and report pickup/release outcome.")
            except Exception as exc:
                message = f"Execution stopped: {exc}"
            self._motion_results.put(("execution", 0, message, dry_run))
        threading.Thread(target=worker, name="grasp-execution", daemon=True).start()

    def execute_pick_sequence(self):
        # UI actions must capture all Tk state on the main thread.
        self._start_execution(False)

    def _return_to_view(self):
        if self._action_running or self._motion_plan is None or self._executor.state!='pregrasp':
            raise RuntimeError('First complete an open Approach Only; this cancels that approach.')
        plan=self._motion_plan
        if plan.config_signature!=workspace_signature(self.workspace,self._grasp_client.config):
            raise RuntimeError('Configuration changed; the stored return route cannot be reused.')
        dry=bool(self.dry_run_var.get())
        log=Path(self._grasp_result.replay_dir)/f'return-view-{time.time_ns()}.json'
        self._action_running=True
        def worker():
            try:
                self._executor.return_to_view(plan,dry_run=dry,log_path=log,
                    progress=lambda text:self._motion_results.put(('progress',0,text,None)))
                message='Dry-run return; no movement.' if dry else 'Returned to the viewing pose. Resume, then freeze a new scene.'
            except Exception as exc: message=f'Return stopped: {exc}'
            self._motion_results.put(('execution',0,message,dry))
        threading.Thread(target=worker,name='grasp-return-view',daemon=True).start()

    def stop_motion(self):
        if self._executor is None or not self._connected:
            raise RuntimeError("Robot is not connected.")
        self._executor.stop()
        self._last_event_text = "Electronic stop sent. Inspect/recover before another attempt."

    def handle_key_action(self, key):
        if not self._action_running:
            super().handle_key_action(key)

    def on_close(self):
        if self._action_running:
            self._last_event_text = "An arm action is running. Stop/finish it before closing."
            return
        super().on_close()


def run(args):
    root = tk.Tk(); app = GraspNetTargetPickGui(root, args)
    app.start(); root.mainloop()
