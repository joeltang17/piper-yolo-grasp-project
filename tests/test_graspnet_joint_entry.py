"""Small-start recovery and joint-transfer regressions; never connects hardware."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace as NS
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
from piper_app.calibration.transforms import pose6_to_matrix, matrix_to_pose6
from piper_app.graspnet.client import load_config
from piper_app.graspnet.planning import (PiperKinematics, MotionStage, build_motion_plan,
    orientation_metrics, plan_preference)
from piper_app.graspnet.execution import GraspExecutor
from piper_app.graspnet.swept import check_gripper_paths
from test_graspnet_execution import plan, FakeClient, synthetic_scene


class RecoveryTests(TestCase):
    def setUp(self):
        self.kin = PiperKinematics('piper', [0.]*6)
        self.ec = load_config('configs/graspnet/default.yaml')['execution']
        self.q = np.deg2rad([10, 20, -30, 10, 20, 10])

    def test_measured_start_is_preserved_and_only_violating_joint_moves_inward(self):
        self.q[2] = np.deg2rad(.397); before = self.q.copy()
        lower, upper = self.kin.lower.copy(), self.kin.upper.copy()
        target = self.kin.recovery_target(self.q, self.ec)
        self.assertAlmostEqual(np.degrees(target[2]), -1.)
        np.testing.assert_array_equal(target[[0,1,3,4,5]], self.q[[0,1,3,4,5]])
        np.testing.assert_array_equal(self.q, before)
        np.testing.assert_array_equal(self.kin.lower, lower)
        np.testing.assert_array_equal(self.kin.upper, upper)
        qs, _ = self.kin.joint_samples(self.q, target, recovery=True)
        self.assertTrue(np.all(np.diff([self.q[2]]+[q[2] for q in qs]) < 0))

    def test_lower_overrun_moves_up_and_larger_overrun_is_refused(self):
        self.q[4] = self.kin.lower[4]-np.deg2rad(.5)
        target = self.kin.recovery_target(self.q, self.ec)
        self.assertAlmostEqual(target[4], self.kin.lower[4]+np.deg2rad(1.))
        self.q[4] = self.kin.lower[4]-np.deg2rad(1.01)
        with self.assertRaisesRegex(RuntimeError, 'at most 1'):
            self.kin.recovery_target(self.q, self.ec)

    def test_inside_start_needs_no_recovery_and_new_targets_stay_strict(self):
        self.assertIsNone(self.kin.recovery_target(self.q, self.ec))
        self.q[2] = np.deg2rad(.01)
        with self.assertRaisesRegex(RuntimeError, 'outside SDK limits'):
            self.kin.require_limits(self.q, 'New target')

    def test_invalid_or_unbounded_recovery_settings_are_refused(self):
        for key, value in [('start_recovery_max_overrun_deg', 73), ('start_recovery_inset_deg', 0)]:
            with self.assertRaises(ValueError):
                self.kin.recovery_target(self.q, {**self.ec, key:value})
        with self.assertRaisesRegex(RuntimeError, 'six finite'):
            self.kin.recovery_target([float('nan')]*6, self.ec)


class JointRouteTests(TestCase):
    def test_joint_route_samples_real_fk_curve_instead_of_tcp_chord(self):
        kin = PiperKinematics('piper', [0.]*6)
        a = np.deg2rad([-30, 60, -90, 0, 30, 0]); b = a.copy(); b[0] = np.deg2rad(30)
        qs, poses = kin.joint_samples(a, b)
        middle = poses[len(poses)//2]
        chord = (kin.forward(a)[:3,3]+kin.forward(b)[:3,3])/2
        self.assertGreater(np.linalg.norm(middle[:3,3]-chord), .02)
        for q, t in zip(qs, poses):
            kin.require_limits(q, 'Test sample')
            np.testing.assert_allclose(t, kin.forward(q))
        with self.assertRaisesRegex(RuntimeError, 'outside SDK limits'):
            kin.joint_samples(a, np.deg2rad([151,60,-90,0,30,0]))

    def test_obstacle_on_supplied_joint_curve_blocks_even_when_chord_is_clear(self):
        cfg = load_config('configs/graspnet/default.yaml')
        start = np.eye(4); start[:3,3] = [0,0,.3]
        end = start.copy(); end[0,3] = .2
        middle = start.copy(); middle[:3,3] = [.1,.1,.3]
        obj = np.array([[.8,.8,.8],[.801,.8,.8]])
        obstacle = np.array([.1,.15,.3])+np.random.default_rng(2).uniform(-.002,.002,(100,3))
        matrices = [('pregrasp',end),('grasp',end),('place',end)]
        check_gripper_paths(start, matrices, start, np.array([.8,.03]), obj, obstacle,
                            np.array([0.,0.,0.]), cfg)
        with self.assertRaisesRegex(RuntimeError, 'envelope on pregrasp'):
            check_gripper_paths(start, matrices, start, np.array([.8,.03]), obj, obstacle,
                np.array([0.,0.,0.]), cfg, segment_paths=[[middle,end],None,None])

    def test_complete_plan_uses_joint_pregrasp_and_linear_final_insertion(self):
        scene, row, w, cfg, clouds = synthetic_scene()
        with patch('piper_app.graspnet.planning.selected_point_clouds', return_value=clouds):
            result = build_motion_plan(scene, row, 0, w, cfg)
        self.assertEqual(result.stages[result.pregrasp_index].motion_kind, 'joint')
        self.assertGreater(len(result.stages[result.pregrasp_index].joint_path), 0)
        self.assertEqual(result.stages[result.pregrasp_index+1].motion_kind, 'linear')
        details = json.loads(result.details_json)
        self.assertGreaterEqual(details['minimum_joint_margin_deg'], 0)
        self.assertIn('grasp_j5_deg', details)

    def test_old_angle_caps_do_not_reject_a_reachable_existing_grasp(self):
        scene, row, w, cfg, clouds = synthetic_scene()
        # Rotate the capture wrist 60 degrees while retaining the exact same
        # base-frame grasp/object. The previous 45-degree veto would reject it.
        kin=PiperKinematics(w.robot,w.tcp_offset)
        meta=json.loads(scene.metadata_json); capture=meta['execution_capture']
        old_camera=pose6_to_matrix(capture['pose']) @ w.T_tcp_camera
        q=np.array(capture['joints']); q[5]+=np.deg2rad(60)
        new_start=kin.forward(q); camera=new_start @ w.T_tcp_camera
        rotation_base=old_camera[:3,:3] @ row[4:13].reshape(3,3)
        anchor_base=old_camera[:3,:3] @ row[13:16]+old_camera[:3,3]
        row=row.copy(); row[4:13]=(camera[:3,:3].T @ rotation_base).ravel()
        row[13:16]=(anchor_base-camera[:3,3]) @ camera[:3,:3]
        def reframe(points):
            base=points @ old_camera[:3,:3].T+old_camera[:3,3]
            return (base-camera[:3,3]) @ camera[:3,:3]
        clouds=(reframe(clouds[0]),reframe(clouds[1]),{})
        capture.update(pose=matrix_to_pose6(new_start),joints=q.tolist())
        scene=replace(scene,metadata_json=json.dumps(meta))
        cfg['execution'].update(max_approach_tilt_deg=.001, max_closing_tilt_deg=.001,
                                max_orientation_change_deg=.001)
        with patch('piper_app.graspnet.planning.selected_point_clouds', return_value=clouds):
            result = build_motion_plan(scene, row, 0, w, cfg)
        self.assertGreater(json.loads(result.details_json)['capture_orientation_change_deg'], 45.)

    def test_top_down_is_a_preference_not_an_angle_veto(self):
        down=np.eye(4); down[:3,:3]=np.diag([-1.,1.,-1.])
        side=np.eye(4); side[:3,:3]=np.array([[0,0,1],[0,1,0],[-1,0,0]])
        self.assertAlmostEqual(orientation_metrics(down, side)['approach_tilt_deg'], 90)
        base={'minimum_joint_margin_deg':10,'maximum_abs_j5_deg':30}
        def candidate(t): return NS(details_json=json.dumps({**base,**orientation_metrics(t, side)}))
        self.assertGreater(plan_preference(candidate(down), [.8]), plan_preference(candidate(side), [.8]))
        # Changing posture can reverse the preference: it is not unconditional.
        bad=NS(details_json=json.dumps({**orientation_metrics(down,side),
            'minimum_joint_margin_deg':0,'maximum_abs_j5_deg':70}))
        self.assertLess(plan_preference(bad,[.1]),plan_preference(candidate(side),[.9]))

    def test_complete_top_down_transfer_with_real_ik_and_observed_hand_checks(self):
        scene,row,w,cfg,clouds=synthetic_scene(grasp_joints=np.deg2rad([0,100,-45,0,40,0]),
                                            object_z_bounds=(-.052,.028))
        # Synthetic tabletop fixture: return the object to its original XY.
        # A 1cm extra rise follows the already vertical 9cm withdrawal.
        kin=PiperKinematics(w.robot,w.tcp_offset)
        grasp=kin.forward(np.deg2rad([0,100,-45,0,40,0]))
        w.drop_pose.tcp_pose[:2]=grasp[:2,3].tolist()
        cfg['execution']['release_clearance_m']=.01
        with patch('piper_app.graspnet.planning.selected_point_clouds',return_value=clouds):
            result=build_motion_plan(scene,row,0,w,cfg)
        details=json.loads(result.details_json)
        self.assertGreater(details['approach_tilt_deg'],89.)
        self.assertLess(abs(details['grasp_j5_deg']),70.)


class EntryClient(FakeClient):
    def __init__(self, p):
        super().__init__(p)
        self.joints=list(p.start_joints); self.fail_recovery=False
        self.robot.get_joint_angles=lambda:NS(msg=self.joints,timestamp=time.time())
        self.robot.move_j=self.move_joint
    def move_joint(self, joints):
        stage=next(s for s in self.plan.stages if s.motion_kind=='joint' and np.allclose(s.joint_solution,joints))
        self.calls.append(stage.name)
        if stage.name=='recover_start' and self.fail_recovery:
            raise RuntimeError('recovery failure')
        self.joints=list(joints); self.pose=list(stage.tcp_pose)
    def move(self, pose):
        super().move(pose)
        stage=next(s for s in self.plan.stages if np.allclose(s.tcp_pose,pose))
        self.joints=list(stage.joint_solution)
    def electronic_emergency_stop(self): self.calls.append('stop')


class RecoveryExecutionTests(TestCase):
    def setUp(self):
        original=plan(); start=np.zeros(6); start[2]=np.deg2rad(.397)
        recover=start.copy(); recover[2]=np.deg2rad(-1.)
        entry=MotionStage('recover_start',(.11,0,.35,0,0,0),tuple(recover),'joint')
        pre=replace(original.stages[0],motion_kind='joint')
        self.p=replace(original,start_joints=tuple(start),stages=(entry,pre)+original.stages[1:])
        self.c=EntryClient(self.p)
        self.e=GraspExecutor(self.c,load_config('configs/graspnet/default.yaml'))
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
    def run_plan(self, **kwargs):
        options=dict(approach_only=True,dry_run=False,mapping_verified=False,
                     log_path=Path(self.temp.name)/'log.json')
        options.update(kwargs)
        with patch('piper_app.graspnet.execution.time.sleep'):
            self.e.run(self.p,**options)
    def test_recovery_runs_only_after_live_action_before_pregrasp_and_not_twice(self):
        self.assertEqual(self.c.calls, [])
        self.run_plan(dry_run=True)
        self.assertEqual(self.c.calls, [])
        self.run_plan()
        self.assertEqual(self.c.calls[:5],['enable','speed','open','recover_start','pregrasp'])
        self.assertNotIn('close',self.c.calls)
        self.assertEqual(self.e.state,'pregrasp')
        # Mock fixture uses identical zero joint goals for pregrasp/return;
        # resume stops before that artificial ambiguity matters.
        self.c.fail_at='grasp'
        with self.assertRaisesRegex(RuntimeError,'simulated'):
            self.run_plan(approach_only=False,mapping_verified=True)
        self.assertEqual(self.c.calls.count('recover_start'),1)
    def test_failed_recovery_stops_and_never_sends_pregrasp_or_close(self):
        self.c.fail_recovery=True
        with self.assertRaisesRegex(RuntimeError,'recovery failure'):
            self.run_plan()
        self.assertIn('stop',self.c.calls)
        self.assertNotIn('pregrasp',self.c.calls)
        self.assertNotIn('close',self.c.calls)
    def test_recovery_requires_joint_arrival_even_when_tcp_reports_arrival(self):
        real_move=self.c.move_joint
        def no_joint_arrival(joints):
            old=list(self.c.joints); real_move(joints); self.c.joints=old
        self.c.robot.move_j=no_joint_arrival
        self.e.config['execution']['timeout_s']=.001
        with self.assertRaisesRegex(RuntimeError,'recover_start did not settle'):
            self.run_plan()
        self.assertNotIn('pregrasp',self.c.calls)
        self.assertIn('stop',self.c.calls)
    def test_start_joint_drift_or_larger_overrun_refuses_before_enable(self):
        self.c.joints[2]=np.deg2rad(1.2)
        with self.assertRaisesRegex(RuntimeError,'changed'):
            self.run_plan()
        self.assertEqual(self.c.calls,[])
    def test_zero_recovery_allowance_refuses_even_small_measured_overrun(self):
        self.e.config['execution']['start_recovery_max_overrun_deg']=0
        with self.assertRaisesRegex(RuntimeError,'at most 0'):
            self.run_plan()
        self.assertEqual(self.c.calls,[])


if __name__=='__main__': main()
