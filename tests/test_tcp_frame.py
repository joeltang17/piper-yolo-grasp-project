"""Calibration and execution must report the same configured tool frame."""
from math import pi
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from piper_app.robot.client import PiperRobotClient
from piper_app.robot.factory import PiperConnectionConfig


class HostOnlySdk:
    def __init__(self):
        self.offset = [0.0] * 6
        self.speed_commands = []

    def connect(self):
        pass

    def disconnect(self):
        pass

    def set_tcp_offset(self, value):
        self.offset = list(value)

    def set_speed_percent(self, value):
        self.speed_commands.append(value)

    def get_tcp_pose(self):
        # A flange at (300, 0, 250) mm, rotated 90 degrees about Z.
        x, y, z = self.offset[:3]
        return SimpleNamespace(msg=[.300 - y, x, .250 + z, 0.0, 0.0, pi / 2])


class TcpFrameTests(TestCase):
    def test_read_only_and_motion_connections_agree_on_tool_frame(self):
        sdk_instances = []
        observed = []
        for configure in (False, True):
            sdk = HostOnlySdk()
            sdk_instances.append(sdk)
            client = PiperRobotClient(PiperConnectionConfig(tcp_offset=[.030, 0, .100, 0, 0, 0]))
            with patch.object(client, 'probe_firmware', return_value=('default', {})), \
                    patch('piper_app.robot.client.AgxArmFactory.create_arm', return_value=sdk), \
                    patch('piper_app.robot.client.time.sleep'):
                client.connect(configure_robot=configure, init_gripper=False)
                observed.append(client.get_tcp_pose())
                client.disconnect()
        expected = [.300, .030, .350, 0.0, 0.0, pi / 2]
        self.assertEqual(observed[0], expected)
        self.assertEqual(observed[1], expected)
        self.assertEqual(sdk_instances[0].speed_commands, [])
        self.assertEqual(sdk_instances[1].speed_commands, [10])


if __name__ == '__main__':
    main()
