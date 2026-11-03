"""A top-face pickup correction must preserve a separately taught release pose."""
from types import SimpleNamespace
from unittest import TestCase, main
import numpy as np

from piper_app.pick_demo.task import build_pick_plan

class PickHeightTests(TestCase):
    def test_block_height_correction_does_not_shift_captured_release(self):
        release = [0.400, -0.018, 0.018, 3.0, 1.2, -3.0]
        workspace = SimpleNamespace(
            observe=SimpleNamespace(tcp_pose=[0.13, 0.0, 0.16, 3.0, 1.1, 3.1]),
            drop_pose=SimpleNamespace(tcp_pose=list(release)),
            task_defaults={
                "pick_point_offset_m": [0.0, 0.0, -0.04225],
                "pregrasp_offset_m": [0.0, 0.0, 0.08],
                "descend_distance_m": 0.08,
                "lift_distance_m": 0.10,
                "drop_offset_m": [0.0, 0.0, 0.08],
            },
        )
        plan = build_pick_plan(
            selected_pixel=(320, 240), camera_point_m=(0.0, 0.0, 0.3),
            base_point_m=np.array([0.42, 0.12, 0.053]), workspace=workspace,
        )
        np.testing.assert_allclose(plan.grasp_pose[:3], [0.42, 0.12, 0.01075])
        np.testing.assert_allclose(plan.pregrasp_pose[:3], [0.42, 0.12, 0.09075])
        np.testing.assert_allclose(plan.lift_pose[:3], [0.42, 0.12, 0.11075])
        self.assertEqual(plan.drop_pose, release)
        self.assertEqual(workspace.drop_pose.tcp_pose, release)
        np.testing.assert_allclose(plan.drop_prepose[:3], [0.400, -0.018, 0.098])

if __name__ == "__main__":
    main()
