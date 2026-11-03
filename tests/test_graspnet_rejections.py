"""Geometry regressions for false placement refusals; no hardware."""
import numpy as np
from unittest import TestCase, main
from unittest.mock import Mock

from piper_app.graspnet.planning import destination_geometry, PiperKinematics


class DestinationGeometryTests(TestCase):
    def setUp(self):
        x,y=np.meshgrid(np.linspace(-.02,.02,20),np.linspace(-.01,.01,20))
        self.target=np.c_[x.ravel(),y.ravel(),np.full(x.size,.04)]
        self.source=self.target+[.2,0,0]
        self.table=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
        self.plane=np.array([0.,0.,0.])

    def evaluate(self, extra, source=None):
        return destination_geometry(np.r_[self.table,extra],self.source if source is None else source,
                                    self.target,self.plane,.004)

    def test_points_in_old_circle_corners_outside_real_footprint_do_not_block(self):
        # This point is inside the old ~37mm-radius circle but outside the
        # actual 40x20mm footprint and configured 4mm clearance.
        noise=np.repeat([[.030,.020,.025]],100,axis=0)
        self.assertLess(np.linalg.norm(noise[0,:2]),np.sqrt(.02**2+.01**2)+.015)
        result=self.evaluate(noise)
        self.assertEqual(result['destination_obstacle_pixels'],0)
        self.assertGreaterEqual(result['destination_support_pixels'],50)

    def test_obstacle_inside_destination_still_blocks(self):
        obstacle=np.repeat([[0.,0.,.025]],100,axis=0)
        self.assertEqual(self.evaluate(obstacle)['destination_obstacle_pixels'],100)

    def test_selected_object_does_not_block_its_own_destination(self):
        result=self.evaluate(self.target,source=self.target)
        self.assertEqual(result['destination_obstacle_pixels'],0)
        self.assertEqual(result['destination_selected_object_pixels_excluded'],len(self.target))

    def test_margin_still_reserves_clearance_outside_object_boundary(self):
        obstacle=np.repeat([[.022,0.,.025]],50,axis=0)
        self.assertEqual(self.evaluate(obstacle)['destination_obstacle_pixels'],50)

    def test_degenerate_footprint_is_not_accepted(self):
        target=np.c_[np.zeros(20),np.linspace(0,.02,20),np.full(20,.04)]
        with self.assertRaisesRegex(RuntimeError,'footprint'):
            destination_geometry(self.table,self.source,target,self.plane,.004)


class IkMessageTests(TestCase):
    def test_solution_jump_is_reported_without_claiming_detected_singularity(self):
        kin=PiperKinematics.__new__(PiperKinematics)
        kin.inverse=Mock(return_value=np.full(6,np.deg2rad(21)))
        start=np.eye(4); end=start.copy();end[0,3]=.01
        with self.assertRaisesRegex(RuntimeError,'discontinuous') as caught:
            kin.check_segment(start,end,np.zeros(6))
        self.assertNotIn('singularity',str(caught.exception))


if __name__=='__main__':
    main()
