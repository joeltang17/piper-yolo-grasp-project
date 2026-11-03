"""Immutable RGB-D selections and camera-frame GraspNet preview geometry."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import time
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class CameraGeometry:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: int = 0
    coeffs: tuple[float, ...] = (0.0,) * 5

    @classmethod
    def from_dict(cls, value: dict) -> "CameraGeometry":
        camera = cls(**{**value, "coeffs": tuple(value.get("coeffs", [0.0] * 5))})
        if camera.width <= 0 or camera.height <= 0 or camera.fx <= 0 or camera.fy <= 0:
            raise ValueError("Invalid camera dimensions or focal lengths.")
        if len(camera.coeffs) != 5 or not np.isfinite([camera.fx, camera.fy, camera.ppx, camera.ppy, *camera.coeffs]).all():
            raise ValueError("Invalid camera intrinsics/distortion coefficients.")
        return camera

    def to_dict(self) -> dict:
        return {**self.__dict__, "coeffs": list(self.coeffs)}

    def native(self):
        import pyrealsense2 as rs
        intrinsics = rs.intrinsics()
        intrinsics.width, intrinsics.height = self.width, self.height
        intrinsics.fx, intrinsics.fy = self.fx, self.fy
        intrinsics.ppx, intrinsics.ppy = self.ppx, self.ppy
        intrinsics.model, intrinsics.coeffs = rs.distortion(self.model), list(self.coeffs)
        return intrinsics


@lru_cache(maxsize=4)
def camera_rays(camera: CameraGeometry) -> np.ndarray:
    """Use the captured stream geometry; never substitute calibration-board K."""
    yy, xx = np.indices((camera.height, camera.width), dtype=np.float32)
    if camera.model == 0:
        rays = np.stack(((xx - camera.ppx) / camera.fx, (yy - camera.ppy) / camera.fy, np.ones_like(xx)), axis=-1)
    else:
        import pyrealsense2 as rs
        intrinsics = camera.native()
        rays = np.asarray([
            rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], 1.0)
            for y in range(camera.height) for x in range(camera.width)
        ], dtype=np.float32).reshape(camera.height, camera.width, 3)
    rays.setflags(write=False)
    return rays


def frozen_array(value, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SceneSelection:
    scene_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    mask: np.ndarray
    camera: CameraGeometry
    metadata_json: str

    @classmethod
    def capture(cls, bundle, detection, *, tcp_pose=None, pose_time=None) -> "SceneSelection":
        if bundle.intrinsics is None:
            raise RuntimeError("This frame has no captured intrinsics. Resume and freeze a new frame.")
        if detection.mask is None:
            raise RuntimeError("GraspNet needs the selected object's mask. Use the YOLOE segmentation checkpoint.")
        camera = CameraGeometry.from_dict(bundle.intrinsics)
        shape = (camera.height, camera.width)
        if bundle.depth_m.shape != shape or detection.mask.shape != shape or bundle.color_rgb.shape[:2] != shape:
            raise ValueError("RGB, depth, mask and captured intrinsics must have the same resolution.")
        if not np.any(detection.mask):
            raise RuntimeError("The selected instance mask is empty.")
        scene_id = uuid.uuid4().hex
        metadata = {
            "scene_id": scene_id, "instance_id": scene_id + ":0", "frame": "color_optical",
            "label": detection.label, "detection_confidence": float(detection.confidence),
            "bbox_xyxy": list(detection.bbox_xyxy), "serial": bundle.serial,
            "camera": camera.to_dict(), "depth_units": "metres",
            "frame_number": bundle.frame_number, "camera_timestamp_ms": bundle.timestamp_ms,
            "camera_timestamp_domain": bundle.timestamp_domain,
            "received_monotonic_s": bundle.received_monotonic_s,
            "read_started_monotonic_s": getattr(bundle, "read_started_monotonic_s", None),
            "fresh_frame": bool(getattr(bundle, "fresh_frame", False)),
            "previous_frame_number": getattr(bundle, "previous_frame_number", None),
            "discarded_frames": int(getattr(bundle, "discarded_frames", 0)),
            "frozen_monotonic_s": time.monotonic(),
            "tcp_pose_sample": list(tcp_pose) if tcp_pose is not None else None,
            "tcp_pose_sample_monotonic_s": pose_time,
            "pose_binding": "read-only sample after frame receipt; preview only, not execution-qualified",
        }
        return cls(scene_id, frozen_array(bundle.color_rgb, np.uint8), frozen_array(bundle.depth_m, np.float32),
                   frozen_array(detection.mask, bool), camera, json.dumps(metadata))

    def save(self, path) -> None:
        np.savez_compressed(path, rgb=self.rgb, depth_m=self.depth_m, mask=self.mask,
                            metadata=np.asarray(self.metadata_json))

    @classmethod
    def load(cls, path) -> "SceneSelection":
        with np.load(path, allow_pickle=False) as data:
            metadata_json = str(data["metadata"].item())
            metadata = json.loads(metadata_json)
            if metadata.get("frame") != "color_optical" or metadata.get("depth_units") != "metres":
                raise ValueError("Replay must contain camera-frame metric depth.")
            camera = CameraGeometry.from_dict(metadata["camera"])
            scene = cls(str(metadata["scene_id"]), frozen_array(data["rgb"], np.uint8),
                        frozen_array(data["depth_m"], np.float32), frozen_array(data["mask"], bool), camera, metadata_json)
        if (scene.depth_m.shape != (camera.height, camera.width) or scene.mask.shape != scene.depth_m.shape
                or scene.rgb.shape != (camera.height, camera.width, 3)):
            raise ValueError("Replay scene resolution does not match its intrinsics.")
        return scene


def selected_point_clouds(scene: SceneSelection, config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    depth = scene.depth_m
    valid = np.isfinite(depth) & (depth >= float(config["depth_min_m"])) & (depth <= float(config["depth_max_m"]))
    kernel = np.ones((3, 3), dtype=np.uint8)
    selected = cv2.erode(scene.mask.astype(np.uint8), kernel, iterations=int(config["mask_erode_pixels"])).astype(bool)
    selected &= valid
    input_count = int(selected.sum())
    points = camera_rays(scene.camera) * depth[..., None]
    selected, component_stats = foreground_component(points, selected, config)
    count = int(selected.sum())
    fraction = count / max(1, int(scene.mask.sum()))
    if count < int(config["min_object_points"]) or fraction < float(config["min_valid_fraction"]):
        raise RuntimeError(f"Insufficient object depth: {count} pixels, {fraction:.0%} valid mask coverage.")
    object_points = np.ascontiguousarray(points[selected], dtype=np.float32)
    scene_points = np.ascontiguousarray(points[valid], dtype=np.float32)
    stats = {"object_unique_pixels": count, "scene_points": len(scene_points), "valid_mask_fraction": fraction,
             "masked_valid_pixels_before_isolation": input_count,
             "foreground_removed_pixels": input_count - count, **component_stats,
             "projection_policy": "captured RealSense color intrinsics and SDK distortion deprojection"}
    return object_points, scene_points, stats


def foreground_component(points, selected, config):
    """Keep the dominant depth-connected surface, without inventing missing depth.

    Adjacent image pixels must also be close in metric 3D. A semantic mask edge
    containing background returns therefore cannot establish grasp ownership.
    The caller retains the unfiltered scene separately for collision checking.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    count = int(selected.sum())
    if not count:
        return selected, {"foreground_components": 0, "foreground_retained_fraction": 0.0}
    ids = np.full(selected.shape, -1, dtype=np.int32)
    ids[selected] = np.arange(count)
    sources, targets = [], []
    distance = float(config.get("foreground_neighbor_m", 0.008))
    for dy, dx in [(0, 1), (1, -1), (1, 0), (1, 1)]:
        ya, yb = slice(0, selected.shape[0] - dy), slice(dy, selected.shape[0])
        xa = slice(max(0, -dx), min(selected.shape[1], selected.shape[1] - dx))
        xb = slice(max(0, dx), min(selected.shape[1], selected.shape[1] + dx))
        joined = selected[ya, xa] & selected[yb, xb]
        joined &= np.linalg.norm(points[ya, xa] - points[yb, xb], axis=-1) <= distance
        sources.append(ids[ya, xa][joined]); targets.append(ids[yb, xb][joined])
    rows, cols = np.concatenate(sources), np.concatenate(targets)
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(count, count))
    n, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    chosen = int(np.argmax(sizes))
    fraction = float(sizes[chosen] / count)
    if fraction < float(config.get("foreground_min_fraction", 0.55)):
        raise RuntimeError(f"Fragmented object depth: largest connected surface has {fraction:.0%} of valid pixels.")
    keep = np.zeros_like(selected)
    keep[selected] = labels == chosen
    return keep, {"foreground_components": int(n), "foreground_retained_fraction": fraction,
                  "foreground_policy": "largest 8-neighbour metric-depth component; no hole filling",
                  "foreground_depth_p05_p50_p95_m": np.percentile(points[keep, 2], [5, 50, 95]).tolist()}


def validate_candidates(rows) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 17 or not np.isfinite(rows).all():
        raise ValueError("GraspNet results must be finite N x 17 arrays.")
    if len(rows):
        rotations = rows[:, 4:13].reshape(-1, 3, 3)
        if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-3):
            raise ValueError("GraspNet returned non-orthogonal rotations.")
        if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-3):
            raise ValueError("GraspNet returned improper rotations.")
        if np.any(rows[:, 1:4] <= 0) or np.any(rows[:, 15] <= 0):
            raise ValueError("Grasp widths, heights, depths and camera Z must be positive.")
    return frozen_array(rows)


def virtual_gripper_segments(row, finger_length_m=0.07):
    width, depth = float(row[1]), float(row[3])
    rotation, translation = row[4:13].reshape(3, 3), row[13:16]
    rear = depth - finger_length_m
    local = np.array([[depth, -width / 2, 0], [rear, -width / 2, 0],
                      [rear, width / 2, 0], [depth, width / 2, 0],
                      [rear - 0.03, 0, 0], [rear, 0, 0]], dtype=np.float64)
    return local @ rotation.T + translation, [(0, 1), (1, 2), (2, 3), (4, 5)]


def render_grasp(image_rgb, row, camera: CameraGeometry, *, pad_length_m=0.07) -> np.ndarray:
    points, segments = virtual_gripper_segments(row, finger_length_m=pad_length_m)
    import pyrealsense2 as rs
    native = camera.native()
    pixels = [rs.rs2_project_point_to_pixel(native, point.tolist()) if point[2] > 0 else None for point in points]
    image = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(image)
    for first, second in segments:
        if pixels[first] is not None and pixels[second] is not None:
            xy = tuple(float(v) for pixel in (pixels[first], pixels[second]) for v in pixel)
            if np.isfinite(xy).all():
                draw.line(xy, fill="#facc15", width=4)
    R = row[4:13].reshape(3, 3)
    centre = row[13:16] + R[:, 0] * (row[3] - pad_length_m / 2)
    if centre[2] > 0:
        u, v = rs.rs2_project_point_to_pixel(native, centre.tolist())
        if np.isfinite([u, v]).all():
            draw.ellipse((u-4, v-4, u+4, v+4), fill="#22d3ee", outline="black")
    draw.text((8, 8), f"VIRTUAL GRASP | score {row[0]:.3f} | width {1000 * row[1]:.1f} mm", fill="#facc15", stroke_width=1, stroke_fill="black")
    draw.text((8, 24), "Cyan = calibrated pad-midpoint TCP target", fill="#22d3ee", stroke_width=1, stroke_fill="black")
    return np.asarray(image)


def grasp_to_business_tcp(T_base_camera, row, T_fingertip_tcp, *, physical_frame_verified=False):
    """Explicit depth-dependent adapter; no guessed Piper axes or legacy offset."""
    if not physical_frame_verified:
        raise RuntimeError("Physical fingertip/TCP frame has not been verified for GraspNet execution.")
    row = validate_candidates(np.asarray(row).reshape(1, 17))[0]
    T_camera_grasp = np.eye(4)
    T_camera_grasp[:3, :3], T_camera_grasp[:3, 3] = row[4:13].reshape(3, 3), row[13:16]
    T_grasp_fingertip = np.eye(4)
    T_grasp_fingertip[0, 3] = row[3]
    for matrix in [T_base_camera, T_fingertip_tcp]:
        matrix = np.asarray(matrix)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("Frame transforms must be finite rigid 4 x 4 matrices.")
        if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(matrix[:3, :3]), 1):
            raise ValueError("Frame transform rotation is invalid.")
    return np.asarray(T_base_camera) @ T_camera_grasp @ T_grasp_fingertip @ np.asarray(T_fingertip_tcp)
