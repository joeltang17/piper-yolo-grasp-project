from __future__ import annotations

import argparse
import json
from pathlib import Path

from piper_app.cli.yolo_target_pick_demo import build_parser as target_parser


def build_parser() -> argparse.ArgumentParser:
    parser = target_parser()
    parser.description = "Category-driven GraspNet candidate preview (no robot motion)."
    parser.set_defaults(dry_run=True, yolo_weights="models/yoloe-26s-seg.pt")
    parser.add_argument("--graspnet-config", default="configs/graspnet/default.yaml")
    parser.add_argument("--replay-scene", type=Path, help="Plan a saved scene.npz offline without camera or CAN.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("This first GraspNet stage is preview only. Use --dry-run; the existing target-pick launcher retains its live mode.")
    if args.replay_scene is not None:
        from dataclasses import replace
        import uuid
        from PIL import Image
        from piper_app.graspnet.client import GraspNetClient, load_config
        from piper_app.graspnet.geometry import SceneSelection, render_grasp
        scene = SceneSelection.load(args.replay_scene)
        # Replays are separate requests and never overwrite the original capture.
        new_id = uuid.uuid4().hex
        metadata = json.loads(scene.metadata_json)
        metadata.update(source_scene_id=scene.scene_id, scene_id=new_id, instance_id=new_id + ":0")
        scene = replace(scene, scene_id=new_id, metadata_json=json.dumps(metadata))
        client = GraspNetClient(load_config(args.graspnet_config))
        try:
            result = client.plan(scene)
            if len(result.candidates):
                Image.fromarray(render_grasp(scene.rgb, result.candidates[0], scene.camera, pad_length_m=client.config["pad_length_m"])).save(Path(result.replay_dir) / "preview.png")
            print(json.dumps({"replay_dir": result.replay_dir, "statistics": json.loads(result.statistics_json)}, indent=2))
        finally:
            client.close()
    else:
        from piper_app.graspnet.gui import run
        run(args)


if __name__ == "__main__":
    main()
