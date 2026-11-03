"""Local file-based boundary to the existing isolated CUDA environment."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import threading

import numpy as np
import yaml

from piper_app.config import repo_root
from .geometry import SceneSelection, frozen_array, selected_point_clouds, validate_candidates


def load_config(path_like: str) -> dict:
    path = Path(path_like)
    if not path.is_absolute():
        path = repo_root() / path
    config = yaml.safe_load(path.read_text())
    for key in ["runtime_python", "source_root", "checkpoint", "replay_dir"]:
        value = Path(config[key]).expanduser()
        absolute = value if value.is_absolute() else repo_root() / value
        # Resolving the executable symlink escapes a venv and loses its packages.
        config[key] = os.path.abspath(absolute) if key == "runtime_python" else str(absolute.resolve())
    for key in ["runtime_python", "source_root", "checkpoint"]:
        if not Path(config[key]).exists():
            raise RuntimeError(f"GraspNet {key} not found: {config[key]}")
    for key in ["num_points", "min_object_points", "max_collision_candidates", "max_results"]:
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    for key in ["max_opening_m", "pad_length_m", "opening_margin_m", "voxel_size_m", "timeout_s", "ownership_radius_m", "approach_distance_m", "foreground_neighbor_m", "network_context_m"]:
        if not np.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if float(config["depth_min_m"]) <= 0 or float(config["depth_max_m"]) <= float(config["depth_min_m"]):
        raise ValueError("Invalid GraspNet depth range.")
    if not 0 < float(config["min_valid_fraction"]) <= 1 or int(config["mask_erode_pixels"]) < 0:
        raise ValueError("Invalid mask coverage/erosion settings.")
    if not 0 < float(config["foreground_min_fraction"]) <= 1:
        raise ValueError("Invalid foreground component fraction.")
    if not 0 < float(config["collision_threshold"]) <= 1:
        raise ValueError("Invalid collision threshold.")
    ec = config["execution"]
    ec.setdefault("placement_mode", "taught_release")
    ec.setdefault("release_clearance_m", 0.05)
    ec.setdefault("start_recovery_max_overrun_deg", 1.0)
    ec.setdefault("start_recovery_inset_deg", 1.0)
    ec.setdefault("initial_candidates",20)
    ec.setdefault("pregrasp_confirmation_max_age_s",60)
    ec.setdefault("minimum_lift_m",min(.03,ec['lift_m']))
    if not np.isfinite(ec['minimum_lift_m']) or not 0<ec['minimum_lift_m']<=ec['lift_m']:
        raise ValueError('execution.minimum_lift_m must be positive and no greater than lift_m.')
    if not 1<=int(ec['initial_candidates'])<=40:
        raise ValueError('execution.initial_candidates must be 1–40.')
    if not np.isfinite(ec['pregrasp_confirmation_max_age_s']) or not 0<ec['pregrasp_confirmation_max_age_s']<=180:
        raise ValueError('Pregrasp confirmation allowance must be positive and at most 180 seconds.')
    if (not np.isfinite(ec["start_recovery_max_overrun_deg"])
            or not 0 <= ec["start_recovery_max_overrun_deg"] <= 1
            or not np.isfinite(ec["start_recovery_inset_deg"])
            or not 0 < ec["start_recovery_inset_deg"] <= 1):
        raise ValueError("Start recovery permits 0–1 deg overrun and a positive inset of at most 1 deg.")
    if ec["placement_mode"] not in {"taught_release", "tabletop"}:
        raise ValueError("execution.placement_mode must be taught_release or tabletop.")
    # Old orientation-cap keys in local YAML are ignored; they are not hardware limits.
    for key in ["approach_m", "lift_m", "release_gap_m", "geometry_margin_m", "pose_translation_tol_m",
                "pose_rotation_tol_deg", "timeout_s", "snapshot_max_age_s", "release_clearance_m"]:
        if not np.isfinite(ec[key]) or float(ec[key]) <= 0:
            raise ValueError(f"execution.{key} must be finite and positive.")
    if not 1 <= ec["speed_percent"] <= 20:
        raise ValueError("Initial grasp execution speed must be 1–20 percent.")
    return config


@dataclass(frozen=True)
class GraspPreview:
    scene_id: str
    candidates: np.ndarray
    statistics_json: str
    replay_dir: str


class GraspNetClient:
    def __init__(self, config: dict):
        self.config = dict(config)
        self._lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._process = None
        self._closed = False

    def plan(self, scene: SceneSelection) -> GraspPreview:
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("Another GraspNet request is still running.")
        try:
            return self._plan(scene)
        finally:
            self._request_lock.release()

    def _plan(self, scene: SceneSelection) -> GraspPreview:
        with self._lock:
            if self._closed:
                raise RuntimeError("GraspNet worker is closed.")
            if self._process is not None:
                raise RuntimeError("Another GraspNet request is still running.")
        folder = Path(self.config["replay_dir"]) / scene.scene_id
        folder.mkdir(parents=True, exist_ok=False)
        scene.save(folder / "scene.npz")
        object_points, scene_points, geometry_stats = selected_point_clouds(scene, self.config)
        metadata = {"scene_id": scene.scene_id, "config": self.config}
        request_path, result_path = folder / "request.npz", folder / "result.npz"
        np.savez_compressed(request_path, object_points=object_points, scene_points=scene_points,
                            metadata=np.asarray(json.dumps(metadata)))
        worker = Path(__file__).with_name("worker.py")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        # The worker imports only the explicitly configured GraspNet source tree.
        env.pop("PYTHONPATH", None)
        with self._lock:
            if self._closed:
                raise RuntimeError("GraspNet worker is closed.")
            process = subprocess.Popen([self.config["runtime_python"], str(worker), "--request", str(request_path),
                                        "--result", str(result_path)], cwd=self.config["source_root"], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self._process = process
        try:
            try:
                output, _ = process.communicate(timeout=float(self.config["timeout_s"]))
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate()
                (folder / "worker.log").write_text(output)
                raise RuntimeError("GraspNet inference timed out; only the inference process was stopped.")
            (folder / "worker.log").write_text(output)
            if process.returncode != 0 or not result_path.exists():
                raise RuntimeError(f"GraspNet failed. {output[-1800:]} (full log: {folder / 'worker.log'})")
        finally:
            with self._lock:
                self._process = None
        with np.load(result_path, allow_pickle=False) as result:
            metadata = json.loads(str(result["metadata"].item()))
            rows = validate_candidates(result["candidates"])
        if metadata["scene_id"] != scene.scene_id or metadata.get("frame") != "color_optical":
            raise RuntimeError("GraspNet returned a stale result or unexpected coordinate frame.")
        if len(rows) and np.any(rows[:, 1] + float(self.config["opening_margin_m"]) > float(self.config["max_opening_m"])):
            raise RuntimeError("GraspNet returned a grasp wider than the configured opening limit.")
        stats = {**geometry_stats, **metadata["stats"]}
        (folder / "summary.json").write_text(json.dumps(stats, indent=2))
        return GraspPreview(scene.scene_id, frozen_array(rows), json.dumps(stats), str(folder))

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
