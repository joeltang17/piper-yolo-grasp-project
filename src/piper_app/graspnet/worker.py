"""Standalone inference worker. Imports no robot, camera or application controller."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np


def infer(request_path: Path, result_path: Path) -> None:
    with np.load(request_path, allow_pickle=False) as request:
        metadata = json.loads(str(request["metadata"].item()))
        object_points = np.asarray(request["object_points"], dtype=np.float32)
        scene_points = np.asarray(request["scene_points"], dtype=np.float32)
    config = metadata["config"]
    for name, points in [("object", object_points), ("scene", scene_points)]:
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError(f"Invalid {name} point cloud.")
        if np.any(points[:, 2] <= 0) or np.max(np.abs(points)) > 10:
            raise ValueError("Point-cloud coordinates must be camera-frame metres.")
    source = Path(config["source_root"])
    checkpoint = Path(config["checkpoint"])
    actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_sha != config["checkpoint_sha256"]:
        raise RuntimeError("GraspNet checkpoint differs from the configured verified asset.")
    paths = [source / "models", source / "utils", source / "pointnet2", source / "knn"]
    for folder in ["pointnet2", "knn"]:
        paths.extend(sorted((source / folder / "build").glob("lib*")))
    sys.path[:0] = [str(path) for path in paths]
    import torch
    from graspnet import GraspNet, pred_decode
    from collision_detector import ModelFreeCollisionDetector
    from scipy.spatial import cKDTree

    if not torch.cuda.is_available():
        raise RuntimeError("This GraspNet native-operator runtime requires the existing CUDA GPU environment.")
    torch.set_num_threads(2)
    torch.manual_seed(int(config["seed"]))
    rng = np.random.default_rng(int(config["seed"]))
    # The baseline was trained on scenes, not an isolated planar face. Include
    # local context for objectness/shape reasoning while retaining CLEAN object
    # points as the sole ownership reference. Background never grants ownership.
    margin = float(config.get("network_context_m", 0.06))
    nearby = np.all((scene_points >= object_points.min(axis=0) - margin)
                    & (scene_points <= object_points.max(axis=0) + margin), axis=1)
    context_points = scene_points[nearby]
    if len(context_points):
        distance, _ = cKDTree(object_points).query(context_points, k=1)
        context_points = context_points[distance > 0.002]
    total = int(config["num_points"])
    object_count = total // 2 if len(context_points) else total
    sample = object_points[rng.choice(len(object_points), object_count, replace=len(object_points) < object_count)]
    if len(context_points):
        context_count = total - object_count
        sample = np.concatenate((sample, context_points[rng.choice(len(context_points), context_count,
                                      replace=len(context_points) < context_count)]))
    sample = np.ascontiguousarray(sample[rng.permutation(total)], dtype=np.float32)
    network = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                      cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    network.load_state_dict(checkpoint_data["model_state_dict"])
    network.cuda().eval()
    started = time.monotonic()
    with torch.no_grad():
        data = {"point_clouds": torch.from_numpy(sample[None]).cuda()}
        rows = pred_decode(network(data))[0].cpu().numpy().astype(np.float64)
    torch.cuda.synchronize()
    stats = {"decoded": len(rows), "forward_seconds": time.monotonic() - started,
             "device": torch.cuda.get_device_name(0), "checkpoint_sha256": actual_sha,
             "neural_input": "clean object plus local scene context; balanced sampling", "network_context_points": len(context_points),
             "collision_input": "all valid scene points",
             "max_opening_m": float(config["max_opening_m"]),
             "opening_margin_m": float(config["opening_margin_m"]),
             "pad_length_m": float(config["pad_length_m"]),
             "collision_geometry": "measured longitudinal pad span; baseline 10 mm finger thickness and predicted height; palm/mount unverified"}
    valid = np.isfinite(rows).all(axis=1)
    if len(rows):
        rotations = rows[:, 4:13].reshape(-1, 3, 3)
        valid &= np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3)), axis=(1, 2)) < 1e-3
        valid &= np.abs(np.linalg.det(rotations) - 1) < 1e-3
        valid &= (rows[:, 0] >= 0) & np.all(rows[:, 1:4] > 0, axis=1) & (rows[:, 15] > 0)
    stats["invalid_rejected"] = int((~valid).sum())
    rows = rows[valid]
    fits = rows[:, 1] + float(config["opening_margin_m"]) <= float(config["max_opening_m"])
    stats["opening_rejected"] = int((~fits).sum())
    rows = rows[fits]
    if len(rows):
        distances, _ = cKDTree(object_points).query(rows[:, 13:16], k=1)
        owned = distances <= float(config["ownership_radius_m"])
        stats["ownership_rejected"] = int((~owned).sum())
        rows = rows[owned]
    rows = rows[np.argsort(-rows[:, 0])[:int(config["max_collision_candidates"])]]
    stats["collision_checked"] = len(rows)
    collision = ModelFreeCollisionDetector(scene_points, voxel_size=float(config["voxel_size_m"]))
    # Use the measured longitudinal pad span instead of the baseline 60 mm.
    # This remains a simplified model, not a verified full finger/palm model.
    collision.finger_length = float(config["pad_length_m"])
    kept = []
    collision_rejected = empty_rejected = 0
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        group = SimpleNamespace(translations=batch[:, 13:16], rotation_matrices=batch[:, 4:13].reshape(-1, 3, 3),
                                widths=batch[:, 1], heights=batch[:, 2], depths=batch[:, 3])
        hit, empty = collision.detect(group, approach_dist=float(config["approach_distance_m"]),
                                      collision_thresh=float(config["collision_threshold"]), return_empty_grasp=True)
        collision_rejected += int(hit.sum())
        empty_rejected += int((empty & ~hit).sum())
        kept.extend(batch[~hit & ~empty])
    stats.update(collision_rejected=collision_rejected, empty_rejected=empty_rejected)
    diverse = []
    for row in kept:
        rotation = row[4:13].reshape(3, 3)
        duplicate = False
        for previous in diverse:
            angular = np.arccos(np.clip((np.trace(rotation.T @ previous[4:13].reshape(3, 3)) - 1) / 2, -1, 1))
            if np.linalg.norm(row[13:16] - previous[13:16]) < 0.015 and angular < np.deg2rad(15):
                duplicate = True
                break
        if not duplicate:
            diverse.append(row)
        if len(diverse) >= int(config["max_results"]):
            break
    rows = np.asarray(diverse, dtype=np.float64).reshape(-1, 17)
    stats["returned"] = len(rows)
    metadata = {"scene_id": metadata["scene_id"], "frame": "color_optical", "stats": stats}
    temporary = result_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, candidates=rows, metadata=np.asarray(json.dumps(metadata)))
    temporary.replace(result_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference-only GraspNet worker")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    infer(args.request, args.result)


if __name__ == "__main__":
    main()
