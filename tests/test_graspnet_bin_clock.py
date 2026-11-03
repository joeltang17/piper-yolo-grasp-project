"""Fresh frame acquisition independent of wall clocks and taught bin release."""
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest import TestCase, main
from unittest.mock import patch, Mock
import time
import json
import numpy as np
import pyrealsense2 as rs

from piper_app.camera.d405 import D405RealSenseCamera, D405CameraConfig
from piper_app.graspnet.client import load_config
from piper_app.graspnet.planning import qualify_capture, release_target, PiperKinematics, build_motion_plan
from piper_app.calibration.transforms import pose6_to_matrix, matrix_to_pose6
from piper_app.graspnet.execution import GraspExecutor
from piper_app.graspnet.live_gui import GraspNetTargetPickGui
from test_graspnet_preview import bundle
from test_graspnet_execution import workspace, plan, FakeClient, synthetic_scene


class Frame:
    def __init__(self, number, *, depth=False):
        self.number = number
        self.data = np.full((3,4),300,dtype=np.uint16) if depth else np.zeros((3,4,3),np.uint8)
        intrinsics = NS(width=4,height=3,fx=5.,fy=5.,ppx=2.,ppy=1.,model=rs.distortion.none,coeffs=[0.]*5)
        self.profile = NS(as_video_stream_profile=lambda:NS(get_intrinsics=lambda:intrinsics))
    def get_data(self): return self.data
    def get_frame_number(self): return self.number
    def get_timestamp(self): return -43200000.  # intentionally unrelated to host epoch
    def get_frame_timestamp_domain(self): return 'timestamp_domain.hardware_clock'


class Frames:
    def __init__(self,number): self.color=Frame(number);self.depth=Frame(number,depth=True)
    def get_color_frame(self): return self.color
    def get_depth_frame(self): return self.depth


class FreshCaptureTests(TestCase):
    def camera(self, second=12):
        camera=D405RealSenseCamera(D405CameraConfig())
        camera._serial='test';camera._last_frame_number=9
        camera._align=NS(process=lambda frames:frames)
        sequence=iter([Frames(11),Frames(second)])
        last=Frames(second)
        camera._pipeline=NS(poll_for_frames=Mock(side_effect=[Frames(10),False]),
                            wait_for_frames=Mock(side_effect=lambda **kw:next(sequence,last)))
        return camera

    def test_fresh_read_discards_backlog_and_crosses_new_frame_barrier(self):
        camera=self.camera();result=camera.read_frames(fresh=True)
        self.assertTrue(result.fresh_frame)
        self.assertEqual(result.frame_number,12);self.assertEqual(result.previous_frame_number,11)
        self.assertEqual(result.discarded_frames,1)
        self.assertEqual(camera._pipeline.wait_for_frames.call_count,2)
        self.assertEqual(result.timestamp_ms,-43200000.)

    def test_repeated_camera_frame_is_rejected_despite_recent_host_receipt(self):
        with self.assertRaisesRegex(RuntimeError,'counter did not advance'):
            self.camera(second=11).read_frames(fresh=True)

    def test_slow_color_stream_reuse_waits_for_a_new_color_frame(self):
        camera=self.camera()
        camera._pipeline.wait_for_frames.side_effect=[Frames(11),Frames(11),Frames(12)]
        result=camera.read_frames(fresh=True)
        self.assertEqual(result.frame_number,12)
        self.assertEqual(camera._pipeline.wait_for_frames.call_count,3)

    def test_freeze_uses_new_image_and_its_prediction_instead_of_cached_frame(self):
        gui=GraspNetTargetPickGui.__new__(GraspNetTargetPickGui)
        old=bundle();fresh=bundle();fresh.frame_number=43
        gui._action_running=False;gui._executor=NS(state='idle');gui._paused=False;gui._connected=True
        gui._client=Mock();gui.grasp_status_var=Mock()
        gui._apply_target_category=Mock();gui._camera_bundle=old;gui._live_prediction='old mask'
        gui._camera=NS(read_frames=Mock(return_value=fresh))
        def process(value):
            self.assertIs(value,fresh)
            gui._live_prediction='fresh mask';gui._live_capture=(43,{'proof':'fresh'},None)
        gui.on_camera_bundle=process
        def freeze(parent_self):
            parent_self._paused=True;parent_self._frozen_bundle=parent_self._camera_bundle
            parent_self._frozen_prediction=parent_self._live_prediction
        with patch('piper_app.graspnet.live_gui.telemetry_sample',return_value={}), \
             patch('piper_app.graspnet.gui.GraspNetTargetPreviewGui.toggle_pause',new=freeze):
            gui.toggle_pause()
        gui._camera.read_frames.assert_called_once_with(fresh=True)
        self.assertIs(gui._frozen_bundle,fresh)
        self.assertEqual(gui._frozen_prediction,'fresh mask')
        self.assertEqual(gui._frozen_capture[0],43)

    def test_verified_fresh_frame_accepts_camera_clock_twelve_hours_wrong(self):
        b=bundle();b.fresh_frame=True;b.previous_frame_number=41
        b.read_started_monotonic_s=10.01;b.received_monotonic_s=10.05
        b.timestamp_ms=-43200000.;b.timestamp_domain='timestamp_domain.hardware_clock'
        before=dict(monotonic_s=10.,wall_s=100.,tcp_pose=[0.]*6,joints=[0.]*6)
        after=dict(monotonic_s=10.1,wall_s=100.1,tcp_pose=[0.]*6,joints=[0.]*6)
        proof=qualify_capture(b,before,after,workspace(),load_config('configs/graspnet/default.yaml'))
        self.assertEqual(proof['monotonic_s'],10.1)

    def test_cached_frame_refused_even_with_perfect_clock(self):
        b=bundle();b.timestamp_ms=100050.;b.timestamp_domain='timestamp_domain.global_time'
        b.received_monotonic_s=10.05
        before=dict(monotonic_s=10.,wall_s=100.,tcp_pose=[0.]*6,joints=[0.]*6)
        after=dict(monotonic_s=10.1,wall_s=100.1,tcp_pose=[0.]*6,joints=[0.]*6)
        with self.assertRaisesRegex(RuntimeError,'newly acquired'):
            qualify_capture(b,before,after,workspace(),load_config('configs/graspnet/default.yaml'))


class BinReleaseTests(TestCase):
    def test_full_taught_height_and_orientation_used_without_destination_plane(self):
        w=workspace();cfg=load_config('configs/graspnet/default.yaml')
        self.assertEqual(cfg['execution']['placement_mode'],'taught_release')
        w.drop_pose.tcp_pose=[.28,-.13,.34,-2.9,1.1,2.8]
        with patch('piper_app.graspnet.planning.destination_geometry',side_effect=AssertionError('no bin tabletop check')):
            pose,detail=release_target(np.eye(4),None,None,None,None,w,cfg)
        np.testing.assert_allclose(pose,pose6_to_matrix(w.drop_pose.tcp_pose))
        self.assertEqual(detail['placement_mode'],'taught_release')
        self.assertAlmostEqual(pose[2,3],.34)

    def test_bin_release_does_not_reject_container_contents_as_table_obstacles(self):
        w=workspace();cfg=load_config('configs/graspnet/default.yaml')
        rubbish=np.full((100,3),.2)
        pose,_=release_target(np.eye(4),rubbish,rubbish,np.array([0,0,0]),0,w,cfg)
        np.testing.assert_allclose(pose,pose6_to_matrix(w.drop_pose.tcp_pose))

    def test_invalid_release_is_not_accepted(self):
        w=workspace();cfg=load_config('configs/graspnet/default.yaml');w.drop_pose.tcp_pose[2]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'release pose'):
            release_target(np.eye(4),None,None,None,None,w,cfg)

    def test_complete_bin_plan_uses_taught_release_and_release_frame_for_withdrawal(self):
        scene,candidate,w,cfg,clouds=synthetic_scene()
        cfg['execution']['placement_mode']='taught_release'
        # This synthetic bin release is a nearby reachable pose above the table.
        base_camera=pose6_to_matrix(json.loads(scene.metadata_json)['execution_capture']['pose'])@w.T_tcp_camera
        from piper_app.graspnet.planning import piper_pad_tcp
        grasp=base_camera@piper_pad_tcp(candidate,cfg['pad_length_m'])
        taught=grasp.copy();taught[:3,3]+=[.08,0,.02]
        w.drop_pose.tcp_pose=matrix_to_pose6(taught)
        with patch('piper_app.graspnet.planning.selected_point_clouds',return_value=clouds), \
             patch('piper_app.graspnet.planning.destination_geometry',side_effect=AssertionError('no destination tabletop')):
            result=build_motion_plan(scene,candidate,0,w,cfg)
        poses={s.name:pose6_to_matrix(s.tcp_pose) for s in result.stages}
        np.testing.assert_allclose(poses['place'],taught,atol=1e-8)
        np.testing.assert_allclose(poses['clear'][:3,3]-taught[:3,3],-taught[:3,2]*.09,atol=1e-8)

    def test_air_drop_is_not_blocked_by_an_infinite_pickup_table_plane(self):
        from piper_app.graspnet.swept import check_gripper_paths
        cfg=load_config('configs/graspnet/default.yaml')
        start=np.eye(4);grasp=start.copy();grasp[2,3]=.6
        lift=grasp.copy();lift[2,3]=.7
        place=grasp.copy();place[0,3]=.3;place[2,3]=.3
        above=place.copy();above[2,3]=.4
        obj=np.array([[0.,0.,.6],[.002,0.,.602]])
        camera=start.copy();camera[2,3]=10  # keep nominal camera body out of this geometry fixture
        matrices=[('grasp',grasp),('lift',lift),('above_place',above),('place',place)]
        with patch('piper_app.graspnet.swept.hand_hits',side_effect=lambda p,*a:np.zeros(len(p),bool)):
            check_gripper_paths(start,matrices,camera,np.r_[.8,.03,.02,.03,np.eye(3).ravel(),[0,0,.6],-1],
                                obj,np.empty((0,3)),np.array([0.,0.,.5]),cfg)
            cfg['execution']['placement_mode']='tabletop'
            with self.assertRaisesRegex(RuntimeError,'support plane'):
                check_gripper_paths(start,matrices,camera,np.r_[.8,.03,.02,.03,np.eye(3).ravel(),[0,0,.6],-1],
                                    obj,np.empty((0,3)),np.array([0.,0.,.5]),cfg)


class PlanAgeTests(TestCase):
    def test_old_plan_without_monotonic_capture_age_cannot_execute(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        p=replace(plan(),capture_monotonic_s=None)
        client=FakeClient(p);executor=GraspExecutor(client,load_config('configs/graspnet/default.yaml'))
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError,'steady-clock'):
                executor.run(p,approach_only=True,dry_run=False,mapping_verified=False,log_path=Path(folder)/'log.json')
        self.assertEqual(client.calls,[])

    def test_wall_clock_difference_does_not_expire_a_new_monotonic_plan(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        p=replace(plan(),capture_wall_s=time.time()-43200)
        client=FakeClient(p);executor=GraspExecutor(client,load_config('configs/graspnet/default.yaml'))
        with TemporaryDirectory() as folder, patch('piper_app.graspnet.execution.time.sleep'):
            executor.run(p,approach_only=True,dry_run=False,mapping_verified=False,log_path=Path(folder)/'log.json')
        self.assertEqual(executor.state,'pregrasp')


if __name__=='__main__': main()
