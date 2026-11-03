"""Category-driven GraspNet candidate preview; no arm-motion execution path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

from PIL import Image

from piper_app.calibration.viewer_base import CalibrationViewerBase
from piper_app.graspnet.client import GraspNetClient, load_config
from piper_app.graspnet.geometry import SceneSelection, render_grasp
from piper_app.yolo_target_pick_demo.gui import YoloTargetPickDemoGuiApp


class GraspNetTargetPreviewGui(YoloTargetPickDemoGuiApp):
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self._grasp_client = GraspNetClient(load_config(args.graspnet_config))
        self._grasp_generation = 0
        self._grasp_scene = None
        self._grasp_result = None
        self._grasp_index = 0
        self._planning = False
        self._closed = False
        self._results = queue.Queue()
        self._result_after = None
        self._frame_pose_sample = None
        self._frame_pose_time = None
        self._frozen_pose_sample = None
        self._frozen_pose_time = None
        super().__init__(root, args)
        self.root.title("Piper GraspNet Target Preview — no arm motion")
        self._result_after = self.root.after(100, self._drain_results)

    def _build_base_vars(self):
        super()._build_base_vars()
        self.grasp_status_var = tk.StringVar(value="Freeze a detected object, then Plan Grasp.")

    def _build_action_panel(self, parent, row):
        super()._build_action_panel(parent, row)
        frame = next(widget for widget in parent.winfo_children()
                     if isinstance(widget, ttk.LabelFrame) and widget.cget("text") == "Actions")
        for widget in frame.winfo_children():
            if isinstance(widget, ttk.Button):
                text = widget.cget("text")
                if text in {"Move Home", "Move Observe", "Open Gripper", "Close Gripper"}:
                    widget.configure(state="disabled")
                elif text == "Execute Pick":
                    widget.configure(text="Plan Grasp (preview)", command=lambda: self.run_action(self.plan_grasp))
            elif isinstance(widget, ttk.Checkbutton) and widget.cget("text") == "Dry-run":
                widget.configure(state="disabled")
        ttk.Label(frame, textvariable=self.grasp_status_var, wraplength=490, justify="left").grid(
            row=5, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Button(frame, text="Next Grasp", command=self.next_grasp).grid(row=6, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(frame, text="Yellow = virtual gripper. No robot motion in this window.", wraplength=370).grid(
            row=6, column=1, columnspan=3, sticky="w", pady=(8, 0))

    def build_note_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="GraspNet preview", padding=10)
        frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(frame, text=(
            "Type a category + Enter → Pause / Freeze → Plan Grasp. The mask and aligned depth produce 3D grasp proposals. "
            "This first stage checks the baseline virtual gripper against scene points; physical gripper mapping, "
            "arm reachability and placement still need verification before executing these grasps. "
            "Use the existing target-pick demo for its current centre-point pickup."
        ), wraplength=1500, justify="left").grid(sticky="w")

    def connect_robot(self):
        # Read-only pose telemetry: no speed/configure/enable or gripper initialization.
        CalibrationViewerBase.connect_robot(self)

    def on_robot_polled(self):
        self.observe_ready_text = "Pose telemetry only; preview has no arm motion"

    def _invalidate_grasp(self):
        self._grasp_generation += 1
        self._grasp_scene = self._grasp_result = None
        self._grasp_index = 0
        if hasattr(self, "grasp_status_var"):
            self.grasp_status_var.set("Selection changed. Freeze a target and Plan Grasp.")

    def _apply_target_category(self):
        if self._current_target_label() != self._configured_target_label:
            self._invalidate_grasp()
        super()._apply_target_category()

    def reload_workspace(self):
        self._invalidate_grasp()
        super().reload_workspace()

    def clear_selection(self):
        self._invalidate_grasp()
        super().clear_selection()

    def on_camera_bundle(self, bundle):
        # This sample is recorded for replay diagnostics only. It is not claimed
        # to be hardware-synchronized, nor used to command a base-frame grasp.
        pose = self.client.get_tcp_pose() if self._connected else None
        self._frame_pose_sample = tuple(pose) if pose is not None else None
        self._frame_pose_time = time.monotonic() if pose is not None else None
        super().on_camera_bundle(bundle)

    def toggle_pause(self):
        was_paused = self._paused
        self._invalidate_grasp()
        super().toggle_pause()
        if self._paused and not was_paused:
            self._frozen_pose_sample = self._frame_pose_sample
            self._frozen_pose_time = self._frame_pose_time

    def _auto_select_target_in_current_view(self):
        # A valid mask can have an invalid/background box centre. GraspNet uses
        # the mask's depth pixels and must not inherit the point-pick heuristic.
        detection = self._find_matching_detection(self._current_prediction(), self._current_target_label())
        self.selected_plan = None
        self._selected_pixel = None
        self._selected_detection = detection
        if detection is None:
            self._selected_detection_text = "No detection selected."
            self.selected_warning_text = "No matching detection in the frozen frame."
            return False
        self._selected_detection_text = f"{detection.label} @ {detection.confidence:.2f} box={detection.bbox_xyxy}"
        self.selected_warning_text = "Mask selected. Plan Grasp will check its depth geometry."
        return True

    def plan_grasp(self):
        if self._planning:
            raise RuntimeError("A GraspNet request is still running.")
        if not self._paused or self._frozen_bundle is None:
            raise RuntimeError("Pause / Freeze a detected object first.")
        if self._current_target_label() != self._configured_target_label:
            raise RuntimeError("Press Enter to apply the category before planning.")
        if not self._auto_select_target_in_current_view():
            raise RuntimeError(self.selected_warning_text)
        scene = self._capture_scene()
        self._grasp_scene, self._grasp_result = scene, None
        generation = self._grasp_generation
        self._planning = True
        self.grasp_status_var.set("Building masked point cloud and running GraspNet on the GPU…")
        self._last_event_text = f"GraspNet request {scene.scene_id}. No robot motion."
        print(self._last_event_text, flush=True)

        def worker():
            try:
                result, error = self._grasp_client.plan(scene), None
            except Exception as exc:
                result, error = None, f"{type(exc).__name__}: {exc}"
            self._results.put((generation, scene, result, error))

        threading.Thread(target=worker, name="graspnet-preview", daemon=True).start()
        self.refresh_ui()

    def _capture_scene(self):
        return SceneSelection.capture(self._frozen_bundle, self._selected_detection,
                                      tcp_pose=self._frozen_pose_sample, pose_time=self._frozen_pose_time)

    def _accept_result(self, generation, scene, result, error):
        self._planning = False
        if generation != self._grasp_generation or self._grasp_scene is not scene:
            self.grasp_status_var.set("Discarded an outdated grasp result. Plan the current target again.")
            return
        if error:
            self.grasp_status_var.set("GraspNet failed; see Last Event.")
            self._last_event_text = error
            print(error, flush=True)
            return
        self._grasp_result = result
        stats = json.loads(result.statistics_json)
        if not len(result.candidates):
            self.grasp_status_var.set("No candidate passed opening, ownership and baseline scene checks. Reframe and retry.")
        else:
            self._grasp_index = 0
            self._update_grasp_status()
            preview = render_grasp(scene.rgb, result.candidates[0], scene.camera, pad_length_m=self._grasp_client.config["pad_length_m"])
            Image.fromarray(preview).save(Path(result.replay_dir) / "preview.png")
        self._last_event_text = f"GraspNet: {stats['returned']} previews; saved scene/results to {result.replay_dir}"
        print(self._last_event_text, flush=True)

    def _drain_results(self):
        if self._closed:
            return
        try:
            while True:
                item = self._results.get_nowait()
                try:
                    self._accept_result(*item)
                except Exception as exc:
                    self._last_event_text = f"Grasp preview error: {exc}"
                    self.grasp_status_var.set("Could not render the grasp. Saved 3D data remains available.")
                self.refresh_ui()
        except queue.Empty:
            pass
        self._result_after = self.root.after(100, self._drain_results)

    def _update_grasp_status(self):
        row = self._grasp_result.candidates[self._grasp_index]
        self.grasp_status_var.set(
            f"Grasp {self._grasp_index + 1}/{len(self._grasp_result.candidates)} | score {row[0]:.3f} | "
            f"opening {row[1] * 1000:.1f} mm | depth {row[3] * 1000:.1f} mm | preview only")

    def next_grasp(self):
        if self._grasp_result is not None and len(self._grasp_result.candidates):
            self._grasp_index = (self._grasp_index + 1) % len(self._grasp_result.candidates)
            self._update_grasp_status()

    def get_display_images(self, bundle):
        rgb, depth = super().get_display_images(bundle)
        if (self._paused and self._grasp_result is not None and len(self._grasp_result.candidates)
                and self._current_target_label() == self._configured_target_label):
            rgb = render_grasp(rgb, self._grasp_result.candidates[self._grasp_index], self._grasp_scene.camera,
                               pad_length_m=self._grasp_client.config["pad_length_m"])
        return rgb, depth

    def execute_pick_sequence(self):
        raise RuntimeError("GraspNet preview does not execute robot motion. Use Plan Grasp in this window.")

    def move_home(self):
        raise RuntimeError("Robot motion is disabled in the GraspNet preview window.")

    move_observe = open_gripper = close_gripper = move_home

    def handle_key_action(self, key):
        focused = self.root.focus_get()
        if focused is not None and focused.winfo_class() in {"Entry", "TEntry", "TCombobox", "Spinbox", "TSpinbox"}:
            return
        if key == "space":
            self.toggle_pause()
        elif key in {"return", "kp_enter", "p"}:
            self.plan_grasp()
        elif key == "c":
            self.clear_selection()

    def on_close(self):
        self._closed = True
        self._grasp_client.close()
        if self._result_after is not None:
            self.root.after_cancel(self._result_after)
        super().on_close()


def run(args):
    root = tk.Tk()
    app = GraspNetTargetPreviewGui(root, args)
    app.start()
    root.mainloop()
