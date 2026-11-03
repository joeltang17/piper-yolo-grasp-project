"""No hardware: snapshots, deprojection, model boundary and preview-only routing."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch
import json

import numpy as np

from piper_app.camera.d405 import D405FrameBundle, D405RealSenseCamera, D405CameraConfig
from piper_app.graspnet.geometry import (
    CameraGeometry, SceneSelection, camera_rays, grasp_to_business_tcp,
    selected_point_clouds, validate_candidates, virtual_gripper_segments,
)
from piper_app.graspnet.client import GraspNetClient, GraspPreview, load_config
from piper_app.graspnet.gui import GraspNetTargetPreviewGui
from piper_app.perception.yolo import YoloDetection
from piper_app.cli.graspnet_target_pick_demo import build_parser


def row(width=0.025):
    value = np.zeros(17)
    value[:4] = [0.7, width, 0.02, 0.03]
    value[4:13] = np.eye(3).ravel()
    value[13:16] = [0.0, 0.0, 0.3]
    value[16] = -1
    return value


def bundle():
    return D405FrameBundle(
        np.zeros((30, 40, 3), dtype=np.uint8), np.full((30, 40), 0.3, dtype=np.float32),
        np.zeros((30, 40, 3), dtype=np.uint8), 40, 30, "saved",
        {"width": 40, "height": 30, "fx": 50.0, "fy": 50.0, "ppx": 20.0, "ppy": 15.0,
         "model": 0, "coeffs": [0.0] * 5}, frame_number=42, received_monotonic_s=10.0)


def capture():
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:25, 10:30] = True
    return SceneSelection.capture(bundle(), YoloDetection("block", 0.8, (10, 5, 30, 25), mask))


class GeometryTests(TestCase):
    def config(self):
        return {"depth_min_m": 0.08, "depth_max_m": 0.9, "mask_erode_pixels": 1,
                "min_object_points": 100, "min_valid_fraction": 0.15}

    def test_snapshot_owns_frozen_copies_and_preserves_metadata(self):
        original = bundle()
        mask = np.ones((30, 40), dtype=bool)
        scene = SceneSelection.capture(original, YoloDetection("bottle", 0.5, (0, 0, 39, 29), mask), tcp_pose=[0]*6)
        original.depth_m[:] = 0
        mask[:] = False
        self.assertGreater(float(scene.depth_m[0, 0]), 0)
        self.assertTrue(scene.mask.all())
        self.assertEqual(json.loads(scene.metadata_json)["frame_number"], 42)
        with self.assertRaises(ValueError):
            scene.depth_m[0, 0] = 1

    def test_object_mask_and_full_scene_remain_separate(self):
        scene = capture()
        obj, context, stats = selected_point_clouds(scene, self.config())
        self.assertEqual(len(obj), 18 * 18)
        self.assertEqual(len(context), 30 * 40)
        self.assertEqual(stats["object_unique_pixels"], 324)
        np.testing.assert_allclose(obj[:, 2], 0.3)

    def test_invalid_object_depth_does_not_become_table_or_zero_points(self):
        original = bundle()
        original.depth_m[5:25, 10:30] = np.nan
        mask = np.zeros((30, 40), dtype=bool)
        mask[5:25, 10:30] = True
        scene = SceneSelection.capture(original, YoloDetection("block", 0.8, (10, 5, 30, 25), mask))
        with self.assertRaisesRegex(RuntimeError, "Insufficient object depth"):
            selected_point_clouds(scene, self.config())

    def test_mask_and_intrinsics_are_required(self):
        with self.assertRaisesRegex(RuntimeError, "mask"):
            SceneSelection.capture(bundle(), YoloDetection("block", 0.8, (0, 0, 10, 10)))
        original = bundle()
        original.intrinsics = None
        with self.assertRaisesRegex(RuntimeError, "intrinsics"):
            SceneSelection.capture(original, YoloDetection("block", 0.8, (0, 0, 10, 10)))

    def test_saved_scene_round_trip(self):
        scene = capture()
        with TemporaryDirectory() as folder:
            path = Path(folder)/"scene.npz"
            scene.save(path)
            replay = SceneSelection.load(path)
        self.assertEqual(replay.scene_id, scene.scene_id)
        np.testing.assert_array_equal(replay.depth_m, scene.depth_m)
        self.assertFalse(replay.mask.flags.writeable)

    def test_projection_agrees_with_native_sdk_for_distorted_stream(self):
        import pyrealsense2 as rs
        camera = CameraGeometry(4, 3, 5, 6, 2, 1, int(rs.distortion.brown_conrady), (0.1, 0.01, 0, 0, 0))
        rays = camera_rays(camera)
        expected = rs.rs2_deproject_pixel_to_point(camera.native(), [3.0, 2.0], 1.0)
        np.testing.assert_allclose(rays[2, 3], expected, atol=1e-7)

    def test_bundle_point_query_uses_its_intrinsics_after_camera_state_changes(self):
        original = bundle()
        camera = D405RealSenseCamera(D405CameraConfig())
        # No live intrinsics or device is needed for a captured bundle query.
        result = camera.query_point_from_bundle(original, 25, 15)
        self.assertTrue(result.valid)
        np.testing.assert_allclose(result.point_m, [0.03, 0, 0.3], atol=1e-7)

    def test_virtual_fingertip_depends_on_depth_and_rotation(self):
        value = row()
        points, _ = virtual_gripper_segments(value)
        np.testing.assert_allclose(points[0], [0.03, -0.0125, 0.3])
        # The displayed contact span now matches the user's 70 mm measurement.
        self.assertAlmostEqual(float(np.linalg.norm(points[0] - points[1])), 0.070)
        alternate, _ = virtual_gripper_segments(value, finger_length_m=0.080)
        self.assertAlmostEqual(float(np.linalg.norm(alternate[0] - alternate[1])), 0.080)

    def test_frame_adapter_requires_verification_and_applies_depth_once(self):
        value = row()
        with self.assertRaisesRegex(RuntimeError, "not been verified"):
            grasp_to_business_tcp(np.eye(4), value, np.eye(4))
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]
        actual = grasp_to_business_tcp(transform, value, np.eye(4), physical_frame_verified=True)
        np.testing.assert_allclose(actual[:3, 3], [1.03, 2, 3.3])

    def test_bad_rotations_and_nonfinite_candidates_are_rejected(self):
        invalid = row()
        invalid[4] = -1
        with self.assertRaisesRegex(ValueError, "improper"):
            validate_candidates(invalid[None])
        invalid = row()
        invalid[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_candidates(invalid[None])


class BoundaryTests(TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = load_config("configs/graspnet/default.yaml")
        self.config.update(replay_dir=self.temp.name, min_object_points=100)

    def fake_process(self, wrong_id=False, wide=False, width=None):
        def start(args, **kwargs):
            with np.load(args[args.index("--request")+1], allow_pickle=False) as data:
                request = json.loads(str(data["metadata"].item()))
                self.assertGreater(len(data["scene_points"]), len(data["object_points"]))
            metadata = {"scene_id": "wrong" if wrong_id else request["scene_id"],
                        "frame": "color_optical", "stats": {"returned": 1}}
            np.savez_compressed(args[args.index("--result")+1],
                                candidates=row(width if width is not None else 0.101 if wide else 0.025)[None],
                                metadata=np.asarray(json.dumps(metadata)))
            return SimpleNamespace(communicate=lambda timeout: ("saved model output", None), returncode=0)
        return start

    def test_runtime_path_preserves_the_virtual_environment_executable(self):
        self.assertTrue(self.config["runtime_python"].endswith("Robot_arm/.venv/bin/python"))

    def test_separate_worker_receives_points_and_returns_bound_result(self):
        client = GraspNetClient(self.config)
        scene = capture()
        with patch("piper_app.graspnet.client.subprocess.Popen", side_effect=self.fake_process()):
            result = client.plan(scene)
        self.assertEqual(result.scene_id, scene.scene_id)
        self.assertEqual(result.candidates.shape, (1, 17))
        self.assertTrue((Path(result.replay_dir)/"scene.npz").exists())

    def test_measured_100mm_gap_accepts_90mm_grasp(self):
        client = GraspNetClient(self.config)
        with patch("piper_app.graspnet.client.subprocess.Popen", side_effect=self.fake_process(width=0.090)):
            preview = client.plan(capture())
        self.assertAlmostEqual(preview.candidates[0, 1], 0.090)

    def test_4mm_clearance_is_still_required_inside_100mm_gap(self):
        client = GraspNetClient(self.config)
        with patch("piper_app.graspnet.client.subprocess.Popen", side_effect=self.fake_process(width=0.097)):
            with self.assertRaisesRegex(RuntimeError, "opening limit"):
                client.plan(capture())

    def test_stale_result_and_excessive_width_are_rejected(self):
        for wrong_id, wide in [(True, False), (False, True)]:
            client = GraspNetClient(self.config)
            with patch("piper_app.graspnet.client.subprocess.Popen", side_effect=self.fake_process(wrong_id, wide)):
                with self.assertRaises(RuntimeError):
                    client.plan(capture())

    def test_closed_client_cannot_launch_model(self):
        client = GraspNetClient(self.config)
        client.close()
        with patch("piper_app.graspnet.client.subprocess.Popen") as process:
            with self.assertRaisesRegex(RuntimeError, "closed"):
                client.plan(capture())
            process.assert_not_called()

    def test_only_one_request_in_flight(self):
        client = GraspNetClient(self.config)
        client._request_lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, "still running"):
                client.plan(capture())
        finally:
            client._request_lock.release()


class PreviewGuiTests(TestCase):
    def test_motion_entry_points_refuse_before_accessing_robot_client(self):
        gui = GraspNetTargetPreviewGui.__new__(GraspNetTargetPreviewGui)
        for method in ["execute_pick_sequence", "move_home", "move_observe", "open_gripper", "close_gripper"]:
            with self.assertRaises(RuntimeError):
                getattr(gui, method)()

    def test_stale_async_result_is_not_displayed(self):
        gui = GraspNetTargetPreviewGui.__new__(GraspNetTargetPreviewGui)
        gui._grasp_generation = 2
        gui._grasp_scene = object()
        gui._grasp_result = None
        gui.grasp_status_var = SimpleNamespace(set=Mock())
        gui._accept_result(1, object(), object(), None)
        self.assertIsNone(gui._grasp_result)
        self.assertFalse(gui._planning)

    def test_category_mode_needs_no_valid_box_center_or_point_pick_plan(self):
        gui = GraspNetTargetPreviewGui.__new__(GraspNetTargetPreviewGui)
        detection = YoloDetection("block", 0.7, (1, 2, 3, 4), np.ones((4, 4), dtype=bool))
        gui._current_prediction = lambda: object()
        gui._current_target_label = lambda: "block"
        gui._find_matching_detection = lambda *_: detection
        self.assertTrue(gui._auto_select_target_in_current_view())
        self.assertIs(gui._selected_detection, detection)
        self.assertIsNone(gui.selected_plan)

    def test_cli_defaults_to_preview_and_accepts_offline_replay(self):
        args = build_parser().parse_args(["--target-label", "wooden block", "--replay-scene", "/tmp/scene.npz"])
        self.assertTrue(args.dry_run)
        self.assertEqual(args.target_label, "wooden block")


if __name__ == "__main__":
    main()
