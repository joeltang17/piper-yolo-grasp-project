from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import ttk
from typing import Optional


from piper_app.camera.d405 import D405FrameBundle
from piper_app.perception.yolo import YoloDetection, YoloPrediction, render_detections
from piper_app.pick_demo.gui import ClickPickDemoGuiApp
from piper_app.pick_demo.task import build_pick_plan, compute_base_point_from_camera, validate_workspace_point


class YoloTargetPickDemoGuiApp(ClickPickDemoGuiApp):
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self._configured_target_label: Optional[str] = None
        self._available_yolo_labels: list[str] = []
        self._target_object_text = str(getattr(args, "target_label", "")).strip()
        super().__init__(root, args)
        self.root.title("Piper YOLO Target Pick Demo")

    def _build_base_vars(self) -> None:
        super()._build_base_vars()
        self.target_object_var = tk.StringVar(value=self._target_object_text)
        self.show_masks_var = tk.BooleanVar(value=True)

    def build_left_panel(self, parent: ttk.Frame) -> None:
        for row in range(7):
            parent.rowconfigure(row, weight=0)
        parent.rowconfigure(6, weight=1)

        self.build_robot_connection_panel(parent, 0, mode_label="Mode: yolo-target-pick")
        self.build_robot_status_panel(
            parent,
            1,
            extra_rows=[
                ("Workspace", self.config_path_var),
                ("Hand-Eye", self.handeye_path_var),
                ("Observe Readiness", self.observe_ready_var),
                ("View Mode", self.view_mode_var),
                ("YOLO", self.yolo_status_var),
                ("Target Object", self.target_object_var),
                ("Selected Pixel", self.selected_pixel_var),
                ("Selected Detection", self.selected_detection_var),
                ("Camera Point", self.selected_camera_point_var),
                ("Base Point", self.selected_base_point_var),
                ("Selection", self.selected_warning_var),
                ("Plan", self.plan_var),
                ("Gripper", self.gripper_state_var),
                ("Gripper Width", self.gripper_width_var),
                ("Gripper Max", self.gripper_max_range_var),
            ],
            title="YOLO Pick Status",
        )
        self._build_action_panel(parent, 2)
        self.build_pose_panel(parent, 3, title="Current TCP Pose")
        self.build_joint_panel(parent, 4, title="Current Joint Angles")

    def build_note_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Notes", padding=10)
        frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(
            frame,
            text=(
                "Move to Observe, type a target category and press Enter, pause the frame, then run Execute Pick. "
                "Promptable YOLOE weights accept descriptions such as wooden block. Boxes and masks are automatic. "
                "If the paused frame does not contain the chosen object, Execute Pick will refuse to run."
            ),
            wraplength=1500,
            justify="left",
        ).grid(sticky="w")

    def _build_action_panel(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        for idx in range(4):
            frame.columnconfigure(idx, weight=1)

        ttk.Button(frame, text="Reload Config", command=lambda: self.run_action(self.reload_workspace)).grid(
            row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 6)
        )
        ttk.Button(frame, text="Pause / Freeze", command=self.toggle_pause).grid(
            row=0, column=1, sticky="ew", padx=(0, 6), pady=(0, 6)
        )
        ttk.Button(frame, text="Move Home", command=lambda: self.run_background_action(self.move_home)).grid(
            row=0, column=2, sticky="ew", padx=(0, 6), pady=(0, 6)
        )
        ttk.Button(frame, text="Move Observe", command=lambda: self.run_background_action(self.move_observe)).grid(
            row=0, column=3, sticky="ew", pady=(0, 6)
        )

        ttk.Button(frame, text="Open Gripper", command=lambda: self.run_background_action(self.open_gripper)).grid(
            row=1, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(frame, text="Clear Target", command=self.clear_selection).grid(
            row=1, column=1, sticky="ew", padx=(0, 6)
        )
        ttk.Button(frame, text="Close Gripper", command=lambda: self.run_background_action(self.close_gripper)).grid(
            row=1, column=2, sticky="ew", padx=(0, 6)
        )
        ttk.Button(frame, text="Execute Pick", command=lambda: self.run_background_action(self.execute_pick_sequence)).grid(
            row=1, column=3, sticky="ew"
        )

        ttk.Label(frame, text="Target Object").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.target_object_combo = ttk.Combobox(frame, textvariable=self.target_object_var, state="normal")
        self.target_object_combo.grid(row=2, column=1, columnspan=3, sticky="ew", pady=(8, 0))
        self.target_object_combo.bind("<<ComboboxSelected>>", self._on_target_object_input)
        self.target_object_combo.bind("<Return>", self._on_target_object_input)
        self.target_object_combo.bind("<FocusOut>", self._on_target_object_input)

        ttk.Label(
            frame,
            text="Flow: type category + Enter -> pause frame -> Execute Pick",
            wraplength=520,
            justify="left",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Checkbutton(frame, text="Dry-run", variable=self.dry_run_var).grid(row=3, column=3, sticky="e", pady=(8, 0))
        ttk.Checkbutton(frame, text="Show masks", variable=self.show_masks_var).grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(frame, text="Pickup uses the saved approach and gripper settings.", wraplength=390).grid(
            row=4, column=1, columnspan=3, sticky="w", pady=(8, 0)
        )

    def reload_workspace(self) -> None:
        if self._action_running:
            return
        super().reload_workspace()
        self.selected_warning_text = "Workspace loaded. Choose a target object, pause the frame, then execute the pick."
        self._last_event_text = "YOLO target-pick workspace reloaded."
        self.refresh_ui()

    def refresh_custom_ui(self) -> None:
        self._target_object_text = self._current_target_label()
        super().refresh_custom_ui()

    def _normalize_label(self, label: str) -> str:
        return str(label).strip().casefold()

    def _current_prediction(self) -> Optional[YoloPrediction]:
        if self._paused:
            return self._frozen_prediction
        return self._live_prediction

    def _current_bundle(self) -> Optional[D405FrameBundle]:
        if self._paused and self._frozen_bundle is not None:
            return self._frozen_bundle
        return self._camera_bundle

    def _current_target_label(self) -> str:
        return str(self.target_object_var.get()).strip()

    def _sync_yolo_label_options(self) -> None:
        labels = self._yolo_detector.suggested_class_names
        if not labels or labels == self._available_yolo_labels:
            return
        self._available_yolo_labels = list(labels)
        if hasattr(self, "target_object_combo"):
            self.target_object_combo["values"] = self._available_yolo_labels

    def _find_matching_detection(self, prediction: Optional[YoloPrediction], target_label: str) -> Optional[YoloDetection]:
        if prediction is None or not target_label:
            return None
        normalized_target = self._normalize_label(target_label)
        matches = [
            detection for detection in prediction.detections
            if self._normalize_label(detection.label) == normalized_target
        ]
        if not matches:
            return None
        return max(matches, key=lambda detection: detection.confidence)

    def _filtered_detections_for_display(self, prediction: Optional[YoloPrediction]) -> list[YoloDetection]:
        if prediction is None:
            return []
        target_label = self._current_target_label()
        if not target_label:
            return list(prediction.detections)
        normalized_target = self._normalize_label(target_label)
        return [
            detection for detection in prediction.detections
            if self._normalize_label(detection.label) == normalized_target
        ]

    def _render_filtered_prediction(self, bundle: D405FrameBundle, prediction: Optional[YoloPrediction]):
        detections = self._filtered_detections_for_display(prediction)
        if prediction is None:
            return bundle.color_rgb
        if not detections and self._current_target_label():
            return bundle.color_rgb

        return render_detections(bundle.color_rgb, detections, show_masks=bool(self.show_masks_var.get()))

    def _select_target_from_source_pixel(
        self,
        src_point: tuple[int, int],
        *,
        bundle: D405FrameBundle,
        detection: Optional[YoloDetection],
    ) -> bool:
        if self.workspace is None:
            self.selected_warning_text = "Workspace not loaded."
            self.refresh_ui()
            return False
        if self._measured_tcp_pose is None:
            self.selected_warning_text = "Current TCP pose is unavailable."
            self.refresh_ui()
            return False

        query = self.camera.query_point_from_bundle(bundle, *src_point)
        if not query.valid or query.point_m is None:
            self.selected_warning_text = f"Depth is invalid at pixel {src_point}."
            self.selected_plan = None
            self._selected_pixel = None
            self._selected_detection = detection
            self.refresh_ui()
            return False

        base_point = compute_base_point_from_camera(self._measured_tcp_pose, self.workspace.T_tcp_camera, query.point_m)
        self.selected_plan = build_pick_plan(
            selected_pixel=src_point,
            camera_point_m=query.point_m,
            base_point_m=base_point,
            workspace=self.workspace,
        )
        warning = validate_workspace_point(self.selected_plan.base_point_m, self.workspace)
        self._selected_pixel = src_point
        self._selected_detection = detection
        if detection is not None:
            self._selected_detection_text = f"{detection.label} @ {detection.confidence:.2f} box={detection.bbox_xyxy}"
        else:
            self._selected_detection_text = "No detection selected."
        if warning:
            self.selected_warning_text = warning
        else:
            self.selected_warning_text = "Target object selected from the paused detection frame."
        self._last_event_text = f"Selected paused bbox center pixel {src_point}."
        self._draw_hover_overlay()
        self.refresh_ui()
        return warning is None

    def _auto_select_target_in_current_view(self) -> bool:
        target_label = self._current_target_label()
        prediction = self._current_prediction()
        bundle = self._current_bundle()
        if not target_label or prediction is None or bundle is None:
            return False
        detection = self._find_matching_detection(prediction, target_label)
        if detection is None:
            self.selected_plan = None
            self._selected_pixel = None
            self._selected_detection = None
            self._selected_detection_text = "No detection selected."
            self.selected_warning_text = f"No '{target_label}' detection in the current frame."
            self.refresh_ui()
            return False
        x1, y1, x2, y2 = detection.bbox_xyxy
        center_pixel = ((x1 + x2) // 2, (y1 + y2) // 2)
        return self._select_target_from_source_pixel(center_pixel, bundle=bundle, detection=detection)

    def _apply_target_category(self) -> None:
        target_label = self._current_target_label()
        if target_label == self._configured_target_label:
            return
        # Predictions belong to a particular prompt. Never retain an old target
        # or plan when a new prompt fails or has not been inferred yet.
        self.selected_plan = None
        self._selected_pixel = None
        self._selected_detection = None
        self._selected_detection_text = "No detection selected."
        self._live_prediction = None
        self._frozen_prediction = None
        self._yolo_detector.set_target_label(target_label)
        self._configured_target_label = target_label
        self._target_object_text = target_label
        self._yolo_enabled = True
        self._sync_yolo_label_options()

    def _on_target_object_input(self, _event=None) -> None:
        if self._action_running:
            self.target_object_var.set(self._configured_target_label or "")
            return
        try:
            changed = self._current_target_label() != self._configured_target_label
            self._apply_target_category()
            target_label = self._current_target_label()
            if self._paused and self._frozen_bundle is not None and (changed or self._frozen_prediction is None):
                self._frozen_prediction = self._yolo_detector.predict(self._frozen_bundle.color_rgb)
                self._auto_select_target_in_current_view()
            mode = "Text prompt" if self._yolo_detector.supports_text_prompts else "Class filter"
            self._last_event_text = f"{mode}: '{target_label or 'all default categories'}'."
        except Exception as exc:
            self.selected_plan = None
            self._live_prediction = None
            self._frozen_prediction = None
            self.selected_warning_text = str(exc)
            self._last_event_text = f"Target category failed: {type(exc).__name__}: {exc}"
        self.refresh_ui()

    def clear_selection(self) -> None:
        if not self._action_running:
            super().clear_selection()

    def _on_canvas_click(self, _event) -> None:
        self._last_event_text = "Manual point clicks are disabled in this demo."
        self.selected_warning_text = "Choose a target object, pause the frame, and let the demo auto-select the bbox center."
        self.refresh_ui()

    def toggle_pause(self) -> None:
        if self._action_running:
            return
        if not self._paused:
            try:
                changed = self._current_target_label() != self._configured_target_label
                self._apply_target_category()
                if (changed or self._live_prediction is None) and self._camera_bundle is not None:
                    self._live_prediction = self._yolo_detector.predict(self._camera_bundle.color_rgb)
            except Exception as exc:
                self.selected_warning_text = str(exc)
                self._last_event_text = f"Cannot freeze target: {exc}"
                self.refresh_ui()
                return
        self._paused = not self._paused
        if self._paused:
            self._frozen_bundle = self._camera_bundle
            self._frozen_prediction = self._live_prediction
            target_label = self._current_target_label()
            if target_label:
                if self._auto_select_target_in_current_view():
                    self._last_event_text = f"Paused current frame and auto-selected '{target_label}'."
                else:
                    self._last_event_text = f"Paused current frame. No '{target_label}' detection found."
            else:
                self._last_event_text = "Paused current frame. Choose a target object to auto-select a bbox center."
        else:
            self._frozen_bundle = None
            self._frozen_prediction = None
            self._last_event_text = "Returned to live camera view."
        self.refresh_ui()

    def on_camera_bundle(self, bundle: D405FrameBundle) -> None:
        if self._yolo_enabled and self._configured_target_label is None:
            try:
                self._apply_target_category()
            except Exception as exc:
                self._yolo_enabled = False
                self._last_event_text = f"Target category failed: {exc}"
                self.selected_warning_text = str(exc)
        super().on_camera_bundle(bundle)
        if self._yolo_enabled:
            self._sync_yolo_label_options()

    def get_display_images(self, bundle: D405FrameBundle) -> tuple[object, Optional[object]]:
        if self._paused and self._frozen_bundle is not None:
            bundle = self._frozen_bundle
            prediction = self._frozen_prediction
        else:
            prediction = self._live_prediction

        color_rgb = self._render_filtered_prediction(bundle, prediction)
        return color_rgb, bundle.depth_visual_rgb

    def execute_pick_sequence(self) -> None:
        target_label = self._current_target_label()
        if not target_label:
            raise RuntimeError("Choose a target object from the dropdown, or type a recognized class name, before executing.")
        if not self._paused:
            raise RuntimeError("Pause the frame first so the target object is locked before executing the pick.")
        if target_label != self._configured_target_label:
            raise RuntimeError("Press Enter to apply the target category before executing the pick.")
        if not self._auto_select_target_in_current_view():
            raise RuntimeError(self.selected_warning_text or f"No paused '{target_label}' object is available to pick.")
        super().execute_pick_sequence()


def run(args: argparse.Namespace) -> None:
    root = tk.Tk()
    app = YoloTargetPickDemoGuiApp(root, args)
    app.start()
    root.mainloop()
