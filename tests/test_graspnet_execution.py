"""Regression checks for geometry mistakes and command ordering; no hardware."""
from dataclasses import replace
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace as NS
from unittest import TestCase, main
from unittest.mock import Mock, patch

import numpy as np
from piper_app.calibration.transforms import pose6_to_matrix, translation_rotation_delta
from piper_app.graspnet.client import load_config
from piper_app.graspnet.geometry import selected_point_clouds, SceneSelection, CameraGeometry
from piper_app.graspnet.planning import (piper_pad_tcp, qualify_capture, PiperKinematics,
    MotionStage, GraspMotionPlan, support_plane, frame_signature, workspace_signature, build_motion_plan)
from piper_app.graspnet.execution import GraspExecutor, scene_unchanged
from piper_app.graspnet.live_gui import GraspNetTargetPickGui
from piper_app.robot.client import PiperRobotClient
from piper_app.pick_demo.task import load_pick_workspace
from test_graspnet_preview import row, capture, bundle


def workspace():
    return load_pick_workspace('configs/task/pick_demo_points.yaml', 'configs/calibration/handeye_active.yaml')


def plan():
    poses = [('pregrasp', [.10, 0, .25, 0, 0, 0]), ('grasp', [.19, 0, .25, 0, 0, 0]),
             ('lift', [.19, 0, .35, 0, 0, 0]), ('above_place', [.29, 0, .35, 0, 0, 0]),
             ('place', [.29, 0, .25, 0, 0, 0]), ('clear', [.20, 0, .25, 0, 0, 0]),
             ('clear_up', [.20, 0, .35, 0, 0, 0]), ('observe', [.10, 0, .35, 0, 0, 0])]
    return GraspMotionPlan('scene', 0, tuple(MotionStage(n, tuple(p), (0.,)*6, 'joint' if n=='observe' else 'linear') for n, p in poses),
                           (.1, 0, .35, 0, 0, 0), (0.,)*6, (0.,)*6, 'frame', 'config', time.time(), 1., .1, '{}',
                           capture_monotonic_s=time.monotonic())


def synthetic_scene(*, grasp_joints=None, object_z_bounds=(-.04, .04)):
    """Known geometry at a synthetic raised table; real SDK FK/IK and hand checks."""
    from piper_app.graspnet.planning import shifted, matrix_to_pose6
    from piper_app.pick_demo.task import PickDemoPose
    w = workspace(); cfg = load_config('configs/graspnet/default.yaml')
    cfg['execution']['placement_mode'] = 'tabletop'  # this fixture explicitly tests support placement
    w.task_defaults['workspace_max_m'][2] = .7  # synthetic raised support, not local hardware settings
    kin = PiperKinematics(w.robot, w.tcp_offset)
    q = [.1, .6, -.8, 0, .2873, .1] if grasp_joints is None else grasp_joints
    grasp = kin.forward(q)
    start = shifted(shifted(grasp, -grasp[:3, 2]*.09), [0, 0, .03])
    joints = kin.inverse(start, q)
    w.observe = PickDemoPose('observe', matrix_to_pose6(start), joints.tolist())
    w.drop_pose.tcp_pose[:2] = (grasp[:2, 3] + [.08, 0]).tolist()
    camera = start @ w.T_tcp_camera
    x, y = np.meshgrid(np.linspace(grasp[0, 3]-.15, grasp[0, 3]+.22, 100), np.linspace(-.2, .2, 100))
    table = np.c_[x.ravel(), y.ravel(), np.full(x.size, grasp[2, 3]+object_z_bounds[0])]
    obj = np.random.default_rng(4).uniform([-.018, -.012, object_z_bounds[0]],
                                         [.018, .012, object_z_bounds[1]], (5000, 3)) + grasp[:3, 3]
    cloud = (np.r_[table, obj] - camera[:3, 3]) @ camera[:3, :3]
    cloud = cloud[(cloud[:, 2] > .08) & (cloud[:, 2] < .9)]
    object_camera = (obj - camera[:3, 3]) @ camera[:3, :3]
    R_P_G = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    R_base_grasp = grasp[:3, :3] @ R_P_G
    rotation = camera[:3, :3].T @ R_base_grasp
    anchor = grasp[:3, 3] + R_base_grasp[:, 0]*.005
    translation = (anchor - camera[:3, 3]) @ camera[:3, :3]
    candidate = np.r_[.8, .03, .02, .03, rotation.ravel(), translation, -1.]
    meta = {'execution_capture': {'pose': matrix_to_pose6(start), 'joints': joints.tolist(),
            'frame_signature': frame_signature(w, cfg), 'wall_s': 0}}
    scene = SceneSelection('synthetic', np.zeros((30, 40, 3), np.uint8), np.full((30, 40), .3, np.float32),
                           np.ones((30, 40), bool), CameraGeometry(40, 30, 50, 50, 20, 15), json.dumps(meta))
    return scene, candidate, w, cfg, (object_camera, cloud, {})


class GeometryRegressionTests(TestCase):
    def setUp(self):
        self.cfg = load_config('configs/graspnet/default.yaml')

    def test_background_inside_mask_cannot_grant_ownership_but_remains_obstacle(self):
        scene = capture()
        depth = scene.depth_m.copy(); depth[5:25, 25:30] = .46
        depth.setflags(write=False); scene = replace(scene, depth_m=depth)
        obj, context, stats = selected_point_clouds(scene, {**self.cfg, 'min_object_points': 100})
        self.assertLess(obj[:, 2].max(), .31)
        self.assertGreater(context[:, 2].max(), .45)
        self.assertGreater(stats['foreground_removed_pixels'], 0)

    def test_fragmented_mask_refuses_without_filling_holes(self):
        scene = capture(); depth = scene.depth_m.copy()
        depth[5:25, 20:30] = .46
        depth.setflags(write=False); scene = replace(scene, depth_m=depth)
        with self.assertRaisesRegex(RuntimeError, 'Fragmented'):
            selected_point_clouds(scene, {**self.cfg, 'min_object_points': 100})

    def test_pad_midpoint_depth_axes_and_jaw_symmetry(self):
        value = row(); value[3] = .01
        t = piper_pad_tcp(value, .07)
        np.testing.assert_allclose(t[:3, 3], [-.025, 0, .3])
        np.testing.assert_allclose(t[:3, 2], [1, 0, 0])  # Piper approach Z
        np.testing.assert_allclose(t[:3, 1], [0, 1, 0])  # Piper closing Y
        swapped = piper_pad_tcp(value, .07, swap_jaws=True)
        np.testing.assert_allclose(swapped[:3, 3], t[:3, 3])
        np.testing.assert_allclose(swapped[:3, 2], t[:3, 2])
        np.testing.assert_allclose(swapped[:3, 1], -t[:3, 1])
        value[3] = .04
        self.assertAlmostEqual(piper_pad_tcp(value, .07)[0, 3] - t[0, 3], .03)

    def test_geodesic_error_handles_pi_wrap_and_euler_equivalence(self):
        a = [0, 0, 0, 0, 0, np.pi-.001]
        b = [0, 0, 0, 0, 0, -np.pi+.001]
        self.assertLess(PiperRobotClient._pose_error(a, b)[1], .12)

    def test_support_plane_excludes_object_and_recovers_tilted_table(self):
        x, y = np.meshgrid(np.linspace(.1, .5, 80), np.linspace(-.2, .2, 80))
        table = np.c_[x.ravel(), y.ravel(), (.01*x-.02*y-.03).ravel()]
        rng = np.random.default_rng(4)
        obj = rng.uniform([.27, -.02, -.02], [.32, .02, .08], (1000, 3))
        plane, _ = support_plane(np.r_[table, obj], obj)
        np.testing.assert_allclose(plane, [.01, -.02, -.03], atol=1e-6)

    def test_local_sdk_ik_recovers_reachable_pose_and_refuses_far_target(self):
        w = workspace(); kin = PiperKinematics(w.robot, w.tcp_offset)
        seed = np.array(w.home.joint_angles); goal = seed.copy(); goal[0] += .02; goal[1] += .02
        target = kin.forward(goal); solved = kin.check_segment(kin.forward(seed), target, seed)
        error = translation_rotation_delta(kin.forward(solved), target)
        self.assertLess(error[0], .002); self.assertLess(error[1], 1)
        target[0, 3] = 2
        with self.assertRaisesRegex(RuntimeError, 'IK solution'):
            kin.inverse(target, seed)

    def test_preview_scene_cannot_become_executable(self):
        with self.assertRaisesRegex(RuntimeError, 'preview-only'):
            build_motion_plan(capture(), row(), 0, workspace(), self.cfg)

    def test_calibration_or_endpoint_changes_invalidate_different_signatures(self):
        w = workspace(); frame = frame_signature(w, self.cfg); whole = workspace_signature(w, self.cfg)
        w.drop_pose.tcp_pose[0] += .01
        self.assertEqual(frame, frame_signature(w, self.cfg))
        self.assertNotEqual(whole, workspace_signature(w, self.cfg))
        w.tcp_offset[2] += .001
        self.assertNotEqual(frame, frame_signature(w, self.cfg))

    def test_qualified_capture_requires_stationary_bracket_and_fresh_camera(self):
        w = workspace(); b = bundle(); b.timestamp_domain = 'timestamp_domain.global_time'; b.timestamp_ms = 100050
        b.received_monotonic_s = 10.05
        b.fresh_frame = True; b.previous_frame_number = 41; b.read_started_monotonic_s = 10.01
        before = dict(monotonic_s=10., wall_s=100., tcp_pose=[0.]*6, joints=[0.]*6)
        after = dict(monotonic_s=10.1, wall_s=100.1, tcp_pose=[0.]*6, joints=[0.]*6)
        proof = qualify_capture(b, before, after, w, self.cfg)
        self.assertEqual(proof['pose'], [0.]*6)
        moving = copy.deepcopy(after); moving['tcp_pose'][0] = .01
        with self.assertRaisesRegex(RuntimeError, 'moved'):
            qualify_capture(b, before, moving, w, self.cfg)
        b.timestamp_ms = 99000
        # Different camera/host wall clocks do not invalidate verified new I/O.
        self.assertIn('monotonic_s', qualify_capture(b, before, after, w, self.cfg))
        b.previous_frame_number = b.frame_number
        with self.assertRaisesRegex(RuntimeError, 'acquisition'):
            qualify_capture(b, before, after, w, self.cfg)

    def test_changed_scene_and_stale_camera_refuse(self):
        scene = capture(); b = bundle(); b.received_monotonic_s = time.monotonic()
        scene_unchanged(scene, b)
        b.depth_m[:] = .4
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            scene_unchanged(scene, b)
        b.received_monotonic_s -= 1
        with self.assertRaisesRegex(RuntimeError, 'fresh'):
            scene_unchanged(scene, b)

    def test_complete_plan_preserves_object_transform_and_passes_real_ik_and_hand_checks(self):
        scene, candidate, w, cfg, clouds = synthetic_scene()
        with patch('piper_app.graspnet.planning.selected_point_clouds', return_value=clouds):
            result = build_motion_plan(scene, candidate, 0, w, cfg)
        stages = {s.name: pose6_to_matrix(s.tcp_pose) for s in result.stages}
        np.testing.assert_allclose(stages['grasp'][:3, :3], stages['place'][:3, :3], atol=1e-8)
        self.assertGreater(stages['clear_up'][2, 3], stages['clear'][2, 3])
        self.assertEqual([s.name for s in result.stages], ['pregrasp','grasp','lift','above_place','place','clear','clear_up','observe'])
        self.assertEqual(result.stages[-1].motion_kind, 'joint')

    def test_occupied_destination_refuses_before_motion(self):
        scene, candidate, w, cfg, clouds = synthetic_scene()
        camera = pose6_to_matrix(json.loads(scene.metadata_json)['execution_capture']['pose']) @ w.T_tcp_camera
        centre = np.r_[w.drop_pose.tcp_pose[:2], .40]
        obstacle = np.random.default_rng(8).uniform(-.01, .01, (200, 3)) + centre
        obstacle_camera = (obstacle-camera[:3, 3]) @ camera[:3, :3]
        with patch('piper_app.graspnet.planning.selected_point_clouds', return_value=(clouds[0], np.r_[clouds[1], obstacle_camera], {})):
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                build_motion_plan(scene, candidate, 0, w, cfg)

    def test_out_of_limit_observe_is_reported_without_clamping(self):
        scene, candidate, w, cfg, clouds = synthetic_scene()
        w.observe.joint_angles[4] = np.deg2rad(71)
        before = w.observe.joint_angles.copy()
        with self.assertRaisesRegex(RuntimeError, 'observe joint 5'):
            build_motion_plan(scene, candidate, 0, w, cfg)
        self.assertEqual(w.observe.joint_angles, before)


class FakeClient:
    def __init__(self, p):
        self.config = NS(tcp_offset=[0.]*6)
        self.pose = list(p.start_pose); self.calls = []; self.width = .08
        self.robot = NS(get_tcp_pose=lambda: NS(msg=self.pose, timestamp=time.time()),
            get_joint_angles=lambda: NS(msg=[0.]*6, timestamp=time.time()), move_l=self.move, move_j=self.return_observe)
        self.plan = p; self.release_blocked = False; self.empty = False; self.fail_at = None

    def move(self, pose):
        name = next(s.name for s in self.plan.stages if np.allclose(s.tcp_pose, pose))
        self.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError('simulated motion failure')
        self.pose = list(pose)

    def return_observe(self, joints):
        self.calls.append('observe'); self.pose = list(self.plan.stages[-1].tcp_pose)

    def get_tcp2flange_pose(self, pose): return pose
    def get_motion_status_code(self): return 0
    def get_arm_status(self): return NS(msg=NS(motion_status=0), timestamp=time.time())
    def enable_and_wait(self, timeout): self.calls.append('enable'); return True
    def set_speed_percent(self, value): self.calls.append('speed')
    def get_gripper_teaching_pendant_param(self, **kwargs): return NS(msg=NS(max_range_config=.08))
    def get_gripper_status(self):
        return NS(timestamp=time.time(), msg=NS(value=self.width, foc_status=NS(driver_enable_status=True, driver_error_status=False)))
    def move_gripper_width(self, width, **kwargs):
        action = 'open' if width else 'close'; self.calls.append(action)
        self.width = (.02 if self.release_blocked and self.calls.count('open') == 2 else .08) if width else (0. if self.empty else .02)


class ExecutionRegressionTests(TestCase):
    def setUp(self):
        self.p = plan(); self.client = FakeClient(self.p)
        self.executor = GraspExecutor(self.client, load_config('configs/graspnet/default.yaml'))
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name)/'attempt.json'

    def run_plan(self, **kwargs):
        options = dict(approach_only=False, dry_run=False, mapping_verified=True, log_path=self.log)
        options.update(kwargs)
        with patch('piper_app.graspnet.execution.time.sleep'):
            self.executor.run(self.p, **options)

    def test_full_sequence_opens_releases_withdraws_rises_and_waits_for_observe(self):
        self.run_plan()
        self.assertEqual(self.client.calls, ['enable','speed','open','pregrasp','grasp','close','lift',
                         'above_place','place','open','clear','clear_up','observe'])
        self.assertEqual(self.executor.state, 'idle')
        self.assertTrue(self.log.exists())

    def test_dry_run_never_accesses_hardware_and_needs_no_alignment(self):
        self.run_plan(dry_run=True, mapping_verified=False)
        self.assertEqual(self.client.calls, [])

    def test_unverified_frame_refuses_full_pick_before_enabling(self):
        with self.assertRaisesRegex(RuntimeError, 'verify'):
            self.run_plan(mapping_verified=False)
        self.assertEqual(self.client.calls, [])

    def test_approach_only_leaves_open_then_can_resume_same_plan(self):
        self.run_plan(approach_only=True, mapping_verified=False)
        self.assertEqual(self.executor.state, 'pregrasp')
        self.assertNotIn('close', self.client.calls)
        self.run_plan()
        self.assertEqual(self.client.calls.count('pregrasp'), 1)

    def test_motion_failure_never_falls_back_or_continues_to_close(self):
        self.client.fail_at = 'grasp'
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            self.run_plan()
        self.assertNotIn('close', self.client.calls)
        self.assertNotIn('lift', self.client.calls)
        self.assertEqual(self.executor.state, 'failed')

    def test_failed_release_prevents_withdrawal_and_return(self):
        self.client.release_blocked = True
        with self.assertRaisesRegex(RuntimeError, 'open command'):
            self.run_plan()
        self.assertNotIn('clear', self.client.calls)
        self.assertNotIn('observe', self.client.calls)

    def test_empty_close_does_not_lift(self):
        self.client.empty = True
        with self.assertRaisesRegex(RuntimeError, 'empty'):
            self.run_plan()
        self.assertNotIn('lift', self.client.calls)

    def test_expired_plan_refuses_before_enable(self):
        self.p = replace(self.p, capture_wall_s=time.time()-200, capture_monotonic_s=time.monotonic()-200)
        with self.assertRaisesRegex(RuntimeError, 'expired'):
            self.run_plan()
        self.assertEqual(self.client.calls, [])

    def test_no_second_execution_can_start(self):
        self.executor._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'running'):
                self.run_plan()
        finally: self.executor._lock.release()


class GuiStateRegressionTests(TestCase):
    def test_reteaching_observe_preserves_saved_local_task_defaults(self):
        from piper_app.keypoints.gui import KeypointCaptureGuiApp
        gui = KeypointCaptureGuiApp.__new__(KeypointCaptureGuiApp)
        gui.output_path_var = NS(get=lambda: 'configs/task/pick_demo_points.yaml')
        gui.robot_var = NS(get=lambda: 'piper'); gui.interface_var = NS(get=lambda: 'socketcan')
        gui.channel_var = NS(get=lambda: 'can0'); gui.bitrate_var = NS(get=lambda: 1000000)
        gui.args = NS(tcp_offset=[0.]*6, task_defaults={'gripper_open_width_m': .0644, 'lift_distance_m': .1})
        gui._records = []
        with patch('piper_app.keypoints.gui.load_keypoint_config', return_value={'task_defaults': {'gripper_open_width_m': 0., 'custom_local_setting': 7}}), \
             patch('piper_app.keypoints.gui.save_keypoint_config', return_value=Path('/tmp/test-points.yaml')) as save:
            gui.save_points()
        saved = save.call_args.args[1]['task_defaults']
        self.assertEqual(saved['gripper_open_width_m'], 0.)
        self.assertEqual(saved['custom_local_setting'], 7)
        self.assertEqual(saved['lift_distance_m'], .1)

    def test_new_inference_cannot_leave_previous_motion_plan_executable(self):
        gui = GraspNetTargetPickGui.__new__(GraspNetTargetPickGui)
        gui._action_running = False; gui._executor = NS(state='idle'); gui._motion_plan = object()
        with patch('piper_app.graspnet.gui.GraspNetTargetPreviewGui.plan_grasp', side_effect=RuntimeError('inference failed')):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                gui.plan_grasp()
        self.assertIsNone(gui._motion_plan)

    def test_mismatched_scene_cannot_reach_tk_or_robot_access(self):
        gui = GraspNetTargetPickGui.__new__(GraspNetTargetPickGui)
        gui._planning = gui._action_running = False
        gui._motion_plan = plan(); gui._grasp_scene = NS(scene_id='new scene')
        gui._grasp_result = NS(scene_id='new scene'); gui._grasp_index = 0
        with self.assertRaisesRegex(RuntimeError, 'displayed scene'):
            gui._start_execution(False)


if __name__ == '__main__':
    main()
