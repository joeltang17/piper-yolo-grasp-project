"""Responsive planning and measured-stability regressions. No hardware."""
from dataclasses import replace
import json
from pathlib import Path
import queue
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace as NS
from unittest import TestCase,main
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh
from piper_app.calibration.transforms import pose6_to_matrix,translation_rotation_delta
from piper_app.graspnet.client import load_config
from piper_app.graspnet.planning import (PiperKinematics,prepare_geometry,build_motion_plan,
    telemetry_sample,samples_stable,MotionStage)
from piper_app.graspnet.swept import (hand_radius,hand_hits,nearby,transform_points,
    check_gripper_paths,payload_hits,prepare_scene)
from piper_app.graspnet.execution import GraspExecutor
from piper_app.graspnet.live_gui import GraspNetTargetPickGui
from test_graspnet_execution import plan,FakeClient,synthetic_scene


class KinematicsTests(TestCase):
    def test_analytic_fk_and_jacobian_match_sdk_and_finite_differences(self):
        k=PiperKinematics('piper',[.01,-.01,.103,0,0,0]);rng=np.random.default_rng(42)
        for _ in range(12):
            q=rng.uniform(k.lower+.01,k.upper-.01)
            t,j=k.forward_jacobian(q)
            np.testing.assert_allclose(t,pose6_to_matrix(fk_from_mdh(k.mdh,q.tolist()))@k.offset,atol=1e-12)
            numerical=[]
            for c in range(6):
                dq=np.zeros(6);dq[c]=1e-7
                a,b=k.forward(q+dq),k.forward(q-dq)
                numerical.append(np.r_[(a[:3,3]-b[:3,3])/2e-7,
                    Rotation.from_matrix(a[:3,:3]@b[:3,:3].T).as_rotvec()/2e-7])
            np.testing.assert_allclose(j,np.asarray(numerical).T,atol=2e-8)

    def test_full_pose_residual_derivative_matches_numerical_solver_derivative(self):
        k=PiperKinematics('piper',[0,0,.1,0,0,0]);q=np.array([.2,.8,-1.1,.2,.4,.3])
        target=k.forward(q+np.array([.04,.05,.02,-.1,.1,.2]));checked=[]
        def verify(fun,seed,*,jac,**kwargs):
            np.testing.assert_allclose(jac(seed),approx_derivative(fun,seed,method='3-point'),atol=2e-7)
            checked.append(True)
            return least_squares(fun,seed,jac=jac,**kwargs)
        with patch('piper_app.graspnet.planning.least_squares',side_effect=verify):
            solved=k.inverse(target,q)
        d,a=translation_rotation_delta(k.forward(solved),target)
        self.assertTrue(checked);self.assertLess(d,.002);self.assertLess(a,1.)

    def test_partial_entry_is_only_accepted_after_required_clearance(self):
        k=PiperKinematics.__new__(PiperKinematics)
        def inverse(target,seed):
            if target[0,3]>.06001:raise RuntimeError('limit')
            return np.full(6,target[0,3])
        k.inverse=inverse
        a=np.eye(4);b=a.copy();b[0,3]=.09
        q,t,d=k.reachable_prefix(a,b,np.zeros(6),.045)
        self.assertAlmostEqual(d,.06);self.assertAlmostEqual(t[0,3],.06)
        with self.assertRaisesRegex(RuntimeError,'required 75'):
            k.reachable_prefix(a,b,np.zeros(6),.075)
    def test_new_ik_goals_leave_encoder_reserve_without_changing_sdk_bounds(self):
        k=PiperKinematics('piper',[0,0,.1,0,0,0])
        lower,upper=k.lower.copy(),k.upper.copy()
        goal=np.array([.1,1.,-1.5,0.,k.upper[4]-.000001,0.])
        target=k.forward(goal);q=k.inverse(target,goal)
        self.assertLessEqual(q[4],upper[4]-np.deg2rad(.1)+1e-10)
        self.assertLess(translation_rotation_delta(k.forward(q),target)[1],1.)
        np.testing.assert_array_equal(k.lower,lower);np.testing.assert_array_equal(k.upper,upper)


class SceneCostTests(TestCase):
    def test_spatial_pruning_preserves_all_finger_and_palm_hits(self):
        cfg=load_config('configs/graspnet/default.yaml');rng=np.random.default_rng(7)
        points=rng.uniform(-.3,.3,(12000,3));tree=cKDTree(points)
        for _ in range(20):
            pose=np.eye(4);pose[:3,:3]=Rotation.random(random_state=rng).as_matrix()
            pose[:3,3]=rng.uniform(-.05,.05,3)
            local=nearby(points,tree,pose[:3,3],hand_radius(.1,cfg))
            self.assertEqual(np.count_nonzero(hand_hits(points,pose,.1,cfg)),
                             np.count_nonzero(hand_hits(local,pose,.1,cfg)))
        np.testing.assert_allclose(transform_points(points,pose[:3,:3]),points@pose[:3,:3],atol=1e-15)

    def test_bounded_payload_queries_preserve_collision_classification(self):
        rng=np.random.default_rng(9);tree=cKDTree(rng.uniform(-1,1,(10000,3)))
        queries=np.r_[rng.uniform(-1,1,(500,3)),rng.uniform(3,10,(100,3))]
        np.testing.assert_array_equal(tree.query(queries)[0]<.006,
            tree.query(queries,distance_upper_bound=.006)[0]<.006)
    def test_one_edge_return_cannot_be_counted_as_many_independent_obstacles(self):
        self.assertFalse(payload_hits(cKDTree([[0.,0.,0.]]),np.zeros((100,3))))
        obstacle=np.array([[0,0,0],[.004,0,0],[0,.004,0],[.004,.004,0]])
        self.assertTrue(payload_hits(cKDTree(obstacle),obstacle))
    def test_target_surface_tolerance_is_capped_and_preserves_nearby_obstacles(self):
        cfg=load_config('configs/graspnet/default.yaml')
        obj=np.array([[0.,0.,.1],[0.,0.,.2]])
        scene=np.array([[.0035,0,.1],[.0045,0,.2]])
        start=np.eye(4);start[:3,3]=[1,1,1]
        # Larger hand padding must not silently expand target ownership.
        cfg['execution']['geometry_margin_m']=.02
        other=prepare_scene(start,obj,scene,np.zeros(3),cfg)[1]
        self.assertEqual(len(other),1)
        np.testing.assert_allclose(other[0],scene[1])

    def test_scene_preparation_is_reused_and_cannot_cross_configurations(self):
        scene,row,w,cfg,clouds=synthetic_scene()
        with patch('piper_app.graspnet.planning.selected_point_clouds',return_value=clouds) as source:
            prepared=prepare_geometry(scene,w,cfg)
            first=build_motion_plan(scene,row,0,w,cfg,geometry=prepared)
            second=build_motion_plan(scene,row,0,w,cfg,geometry=prepared)
        self.assertEqual(source.call_count,1)
        self.assertEqual(first.config_signature,second.config_signature)
        cfg['execution']['lift_m']+=.001
        with self.assertRaisesRegex(RuntimeError,'does not belong'):
            build_motion_plan(scene,row,0,w,cfg,geometry=prepared)


class PickupSupportTests(TestCase):
    def fixture(self):
        cfg=load_config('configs/graspnet/default.yaml')
        cfg['execution']['placement_mode']='taught_release'
        rot=np.array([[0,0,1],[0,1,0],[-1,0,0]])
        start=np.eye(4);start[:3,:3]=rot;start[:3,3]=[.1,0,.1]
        grasp=start.copy();grasp[0,3]=.3
        lift=grasp.copy();lift[2,3]+=.03
        xy=np.linspace(-.01,.01,18);x,y=np.meshgrid(xy,xy)
        bottom=np.c_[.3+x.ravel(),y.ravel(),np.full(x.size,-.005)]
        top=bottom.copy();top[:,2]=.2
        obj=np.r_[bottom,top]
        x,y=np.meshgrid(np.linspace(.27,.33,30),np.linspace(-.03,.03,30))
        floor=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
        mount=np.eye(4);mount[:3,3]=[0,.2,-.2]
        return cfg,start,grasp,lift,obj,floor,start@mount
    def check(self,extra=None,transport_down=False,lift_distance=.03):
        cfg,start,g,lift,obj,floor,cam=self.fixture()
        lift[2,3]=g[2,3]+lift_distance
        matrices=[('grasp',g),('lift',lift)]
        if transport_down:matrices.append(('above_place',g))
        matrices.append(('place',lift))
        context=np.r_[floor,obj] if extra is None else np.r_[floor,obj,extra]
        check_gripper_paths(start,matrices,cam,np.array([.8,.03]),obj,context,np.zeros(3),cfg)
    def test_initial_support_proximity_does_not_block_an_upward_lift(self):
        self.check()
    def test_elevated_obstacle_and_transport_floor_contact_still_block(self):
        x,y=np.meshgrid(np.linspace(-.01,.01,18),np.linspace(-.01,.01,18))
        obstacle=np.c_[.3+x.ravel(),y.ravel(),np.full(x.size,.208)]
        with self.assertRaisesRegex(RuntimeError,'Carried object intersects'):
            self.check(extra=obstacle)
        with self.assertRaisesRegex(RuntimeError,'Carried object intersects'):
            self.check(transport_down=True)
    def test_downward_motion_or_inadequate_final_clearance_is_not_a_lift(self):
        with self.assertRaisesRegex(RuntimeError,'pickup support'):
            self.check(lift_distance=-.02)
        with self.assertRaisesRegex(RuntimeError,'clear the measured'):
            self.check(lift_distance=.005)


class ImmediateThread:
    def __init__(self,*,target,**kwargs):self.target=target
    def start(self):self.target()


class PlanningFlowTests(TestCase):
    def run_search(self,valid_index):
        temp=TemporaryDirectory();self.addCleanup(temp.cleanup)
        gui=GraspNetTargetPickGui.__new__(GraspNetTargetPickGui)
        gui._grasp_scene=NS(metadata_json=json.dumps({'execution_capture':{}}))
        rows=np.zeros((40,17));rows[:,0]=np.arange(40)
        gui._grasp_result=NS(candidates=rows,replay_dir=temp.name)
        gui.workspace=NS(robot='piper',tcp_offset=[0.]*6,observe=NS(joint_angles=[0.]*6))
        gui._grasp_client=NS(config={'execution':{'initial_candidates':20}})
        gui._closed=False;gui._grasp_generation=1;gui._motion_results=queue.Queue()
        gui.grasp_status_var=NS(set=lambda value:None)
        attempted=[]
        def build(scene,row,index,*args,**kwargs):
            attempted.append(index)
            if index!=valid_index:raise RuntimeError('no IK')
            return replace(plan(),candidate_index=index)
        with patch('piper_app.graspnet.live_gui.threading.Thread',ImmediateThread), \
             patch('piper_app.graspnet.live_gui.PiperKinematics'), \
             patch('piper_app.graspnet.live_gui.prepare_geometry'), \
             patch('piper_app.graspnet.live_gui.candidate_preference',side_effect=lambda scene,row,*args:-row[0]), \
             patch('piper_app.graspnet.live_gui.build_motion_plan',side_effect=build):
            gui._build_motion_plans(list(range(40)))
        messages=[]
        while not gui._motion_results.empty():messages.append(gui._motion_results.get())
        return attempted,messages
    def test_first_usable_plan_returns_without_comparing_three(self):
        attempted,messages=self.run_search(0)
        self.assertEqual(attempted,[0]);self.assertEqual(messages[-1][2].candidate_index,0)
    def test_second_batch_is_automatic_and_does_not_require_next_grasp_clicks(self):
        attempted,messages=self.run_search(25)
        self.assertEqual(attempted,list(range(26)))
        self.assertTrue(any('automatic fallback' in m[2] for m in messages if m[0]=='planning_progress'))
        self.assertEqual(messages[-1][2].candidate_index,25)


class FakeClock:
    def __init__(self):self.now=time.monotonic()
    def sleep(self,seconds):self.now+=seconds


class ExecutionTimingTests(TestCase):
    def setUp(self):
        self.p=plan();self.c=FakeClient(self.p)
        self.p=replace(self.p,return_view=MotionStage('return_view',self.p.start_pose,(0.,)*6,'joint'))
        self.c.plan=self.p
        self.e=GraspExecutor(self.c,load_config('configs/graspnet/default.yaml'))
        self.clock=FakeClock();self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
    def invoke(self,**kwargs):
        opts=dict(approach_only=True,dry_run=False,mapping_verified=True,log_path=Path(self.temp.name)/'log.json')
        opts.update(kwargs)
        with patch('time.monotonic',side_effect=lambda:self.clock.now),patch('time.sleep',side_effect=self.clock.sleep):
            self.e.run(self.p,**opts)
    def test_target_not_reached_does_not_claim_a_stationary_arm_is_moving(self):
        self.c.get_arm_status=lambda:NS(msg=NS(motion_status=1,arm_status=0),timestamp=time.time())
        self.assertEqual(telemetry_sample(self.c)['motion_status'],1)
        self.invoke();self.assertEqual(self.e.state,'pregrasp')
    def test_fault_and_measured_movement_are_still_rejected(self):
        self.c.get_arm_status=lambda:NS(msg=NS(motion_status=1,arm_status=1),timestamp=time.time())
        with self.assertRaisesRegex(RuntimeError,'arm fault'):telemetry_sample(self.c)
        a={'tcp_pose':[0.]*6,'joints':[0.]*6};b={**a,'joints':[.02]*6}
        self.assertFalse(samples_stable([a,b]))
    def test_inspection_uses_arrival_clock_without_rewriting_original_capture(self):
        self.p=replace(self.p,capture_monotonic_s=self.clock.now-175);self.c.plan=self.p
        original=self.c.move
        def slow_move(pose):self.clock.now+=10;original(pose)
        self.c.robot.move_l=slow_move
        capture=self.p.capture_monotonic_s
        self.invoke();self.clock.now+=10
        self.c.robot.move_l=self.c.move
        self.invoke(approach_only=False)
        self.assertEqual(self.e.state,'idle');self.assertIn('close',self.c.calls)
        self.assertEqual(self.p.capture_monotonic_s,capture)
    def test_expired_inspection_can_cancel_through_explicit_checked_return(self):
        self.invoke();self.clock.now+=61
        with self.assertRaisesRegex(RuntimeError,'Return to View'):
            self.invoke(approach_only=False)
        self.assertNotIn('close',self.c.calls)
        with patch('time.monotonic',side_effect=lambda:self.clock.now),patch('time.sleep',side_effect=self.clock.sleep):
            self.e.return_to_view(self.p,dry_run=False,log_path=Path(self.temp.name)/'return.json')
        self.assertEqual(self.e.state,'idle');self.assertNotIn('close',self.c.calls)
    def test_return_dry_run_never_moves(self):
        self.invoke();before=list(self.c.calls)
        self.e.return_to_view(self.p,dry_run=True,log_path=Path(self.temp.name)/'dry-return.json')
        self.assertEqual(self.c.calls,before);self.assertEqual(self.e.state,'pregrasp')
    def test_failed_opening_during_return_stops_without_sending_arm_return(self):
        self.invoke();before=self.c.calls.count('observe')
        self.c.release_blocked=True
        self.c.electronic_emergency_stop=lambda:self.c.calls.append('stop')
        with patch('time.monotonic',side_effect=lambda:self.clock.now),patch('time.sleep',side_effect=self.clock.sleep):
            with self.assertRaisesRegex(RuntimeError,'open command'):
                self.e.return_to_view(self.p,dry_run=False,log_path=Path(self.temp.name)/'bad-return.json')
        self.assertEqual(self.e.state,'failed');self.assertIn('stop',self.c.calls)
        self.assertEqual(self.c.calls.count('observe'),before)


if __name__=='__main__':main()
