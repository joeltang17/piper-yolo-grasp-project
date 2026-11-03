"""Device-family and explicit-serial selection without opening hardware."""
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from piper_app.camera.d405 import D405CameraConfig, D405RealSenseCamera


class CameraSelectionTests(TestCase):
    def select(self, devices, serial="auto"):
        fake_rs = SimpleNamespace(camera_info=SimpleNamespace(name="name", serial_number="serial"))
        context = SimpleNamespace(query_devices=lambda: devices)
        camera = D405RealSenseCamera(D405CameraConfig(serial=serial))
        with patch("piper_app.camera.d405._import_rs", return_value=fake_rs):
            return camera._select_device(context)

    @staticmethod
    def device(name, serial):
        return SimpleNamespace(get_info=lambda field: {"name": name, "serial": serial}[field])

    def test_supported_camera_names(self):
        for name in ("Intel RealSense D405", "Intel RealSense D435", "Intel RealSense D435I", "Intel RealSense D435i"):
            with self.subTest(name=name):
                device = self.device(name, "123")
                self.assertIs(self.select([device]), device)

    def test_explicit_serial_selects_correct_camera(self):
        first = self.device("Intel RealSense D405", "first")
        wanted = self.device("Intel RealSense D435I", "wanted")
        self.assertIs(self.select([first, wanted], "wanted"), wanted)

    def test_unknown_explicit_serial_does_not_fall_back(self):
        device = self.device("Intel RealSense D435I", "123")
        with self.assertRaisesRegex(RuntimeError, "missing.*not found"):
            self.select([device], "missing")

    def test_unsupported_camera_is_not_selected(self):
        self.assertIsNone(self.select([self.device("Intel RealSense D415", "123")]))

    def test_no_devices(self):
        self.assertIsNone(self.select([]))
