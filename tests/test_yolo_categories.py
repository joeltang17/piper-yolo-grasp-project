"""Perception/target selection regressions; no GUI window, camera or robot."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import numpy as np

from piper_app.perception.yolo import (
    YoloConfig, YoloDetection, YoloDetector, YoloPrediction,
    _instance_masks, render_detections,
)
from piper_app.yolo_target_pick_demo.gui import YoloTargetPickDemoGuiApp


class Array:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def tolist(self):
        return self.values.tolist()


class Boxes:
    def __init__(self, boxes, classes):
        self.xyxy = Array(boxes)
        self.cls = Array(classes)
        self.conf = Array([0.9] * len(boxes))
        self.count = len(boxes)

    def __len__(self):
        return self.count


class PromptModel:
    def __init__(self, _path):
        self.names = {0: "bottle", 1: "mouse"}
        self.model = SimpleNamespace(model=[SimpleNamespace(is_fused=False)])
        self.set_calls = []

    def set_classes(self, classes):
        self.set_calls.append(list(classes))
        self.names = dict(enumerate(classes))


class FixedModel:
    def __init__(self, _path):
        self.names = {0: "bottle", 1: "mouse"}
        self.model = SimpleNamespace(model=[SimpleNamespace()])


class FusedModel(PromptModel):
    def __init__(self, path):
        super().__init__(path)
        self.model.model[-1].is_fused = True


class YoloCategoryTests(TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.weights = Path(self.temp.name) / "renamed.pt"
        self.weights.touch()

    def detector(self):
        return YoloDetector(YoloConfig(weights_path=str(self.weights), device="cpu"))

    def test_typed_custom_category_becomes_prompt_and_switches_back(self):
        detector = self.detector()
        with patch("piper_app.perception.yolo._import_yolo_runtime", return_value=PromptModel):
            detector.set_target_label("wooden block")
        self.assertEqual(detector.class_names, ["wooden block"])
        self.assertTrue(detector.supports_text_prompts)
        model = detector._model
        self.assertEqual(model.set_calls, [["wooden block"]])
        detector.set_target_label("wooden block")
        self.assertEqual(len(model.set_calls), 1)
        detector.set_target_label("bottle")
        self.assertEqual(detector.class_names, ["bottle"])
        detector.set_target_label("")
        self.assertEqual(detector.class_names, ["bottle", "mouse"])

    def test_fixed_weights_accept_known_class_but_reject_unknown(self):
        detector = self.detector()
        with patch("piper_app.perception.yolo._import_yolo_runtime", return_value=FixedModel):
            detector.set_target_label("Bottle")
        self.assertFalse(detector.supports_text_prompts)
        with self.assertRaisesRegex(RuntimeError, "fixed-vocabulary"):
            detector.set_target_label("wooden block")
        self.assertEqual(detector.class_names, ["bottle", "mouse"])

    def test_failed_backend_prompt_update_discards_partial_model(self):
        detector = self.detector()
        with patch("piper_app.perception.yolo._import_yolo_runtime", return_value=PromptModel):
            detector.set_target_label("bottle")
        detector._model.set_classes = Mock(side_effect=RuntimeError("encoder failed"))
        with self.assertRaisesRegex(RuntimeError, "encoder failed"):
            detector.set_target_label("wooden block")
        self.assertIsNone(detector._model)
        self.assertEqual(detector._target_label, "bottle")

    def test_fused_weights_do_not_advertise_custom_prompts(self):
        detector = self.detector()
        with patch("piper_app.perception.yolo._import_yolo_runtime", return_value=FusedModel):
            detector.set_target_label("bottle")
        self.assertFalse(detector.supports_text_prompts)
        self.assertEqual(detector._model.set_calls, [])
        with self.assertRaisesRegex(RuntimeError, "fixed-vocabulary"):
            detector.set_target_label("wooden block")

    def test_masks_are_associated_with_correct_boxes_at_original_resolution(self):
        detector = self.detector()
        rgb = np.zeros((80, 160, 3), dtype=np.uint8)
        masks = np.zeros((2, 80, 160), dtype=np.float32)
        masks[0, 10:30, 20:50] = 1
        masks[1, 45:65, 100:130] = 1
        result = SimpleNamespace(
            boxes=Boxes([[20, 10, 50, 30], [100, 45, 130, 65]], [0, 1]),
            names={0: "bottle", 1: "block"},
            masks=SimpleNamespace(data=Array(masks)), plot=lambda: rgb.copy(),
        )
        model = SimpleNamespace(predict=Mock(return_value=[result]))
        detector._model = model
        detector._device_label = "cpu"
        prediction = detector.predict(rgb)
        self.assertEqual([item.label for item in prediction.detections], ["bottle", "block"])
        self.assertTrue(prediction.detections[0].mask[15, 25])
        self.assertFalse(prediction.detections[1].mask[15, 25])
        self.assertTrue(prediction.detections[1].mask[50, 110])
        self.assertEqual(prediction.detections[0].mask.dtype, np.dtype(bool))
        self.assertTrue(model.predict.call_args.kwargs["retina_masks"])
        drawn = render_detections(rgb, [prediction.detections[1]])
        np.testing.assert_array_equal(drawn[15, 25], [0, 0, 0])
        self.assertGreater(int(drawn[50, 110].sum()), 0)
        self.assertEqual(int(rgb.sum()), 0)

    def test_letterboxed_masks_use_original_coordinate_polygons(self):
        result = SimpleNamespace(masks=SimpleNamespace(
            data=Array(np.ones((1, 64, 64))),
            xy=[np.array([[50, 25], [80, 25], [80, 45], [50, 45]])],
        ))
        mask = _instance_masks(result, (100, 200))[0]
        self.assertEqual(mask.shape, (100, 200))
        self.assertTrue(mask[30, 60])
        self.assertFalse(mask[60, 30])

    def test_box_only_weights_remain_supported(self):
        detector = self.detector()
        rgb = np.zeros((30, 40, 3), dtype=np.uint8)
        result = SimpleNamespace(
            boxes=Boxes([[1, 2, 20, 25]], [0]), names={0: "bottle"},
            masks=None, plot=lambda: rgb.copy(),
        )
        detector._model = SimpleNamespace(predict=lambda *args, **kwargs: [result])
        self.assertIsNone(detector.predict(rgb).detections[0].mask)

    def test_box_mask_count_mismatch_is_an_error(self):
        detector = self.detector()
        rgb = np.zeros((30, 40, 3), dtype=np.uint8)
        result = SimpleNamespace(
            boxes=Boxes([[1, 2, 20, 25]], [0]), names={0: "bottle"},
            masks=SimpleNamespace(data=Array(np.empty((0, 30, 40)))), plot=lambda: rgb.copy(),
        )
        detector._model = SimpleNamespace(predict=lambda *args, **kwargs: [result])
        with self.assertRaisesRegex(RuntimeError, "different numbers"):
            detector.predict(rgb)


class TargetDemoTests(TestCase):
    def app(self):
        app = YoloTargetPickDemoGuiApp.__new__(YoloTargetPickDemoGuiApp)
        app._configured_target_label = "bottle"
        app._current_target_label = lambda: "wooden block"
        app._yolo_detector = SimpleNamespace(
            set_target_label=Mock(), predict=Mock(), supports_text_prompts=True,
            suggested_class_names=["bottle", "wooden block"],
        )
        app.selected_plan = object()
        app._selected_pixel = (10, 20)
        app._selected_detection = YoloDetection("bottle", 0.9, (1, 2, 30, 40))
        app._selected_detection_text = "bottle"
        app._live_prediction = object()
        app._frozen_prediction = object()
        app._action_running = False
        app._paused = True
        app._frozen_bundle = SimpleNamespace(color_rgb=np.zeros((20, 40, 3), dtype=np.uint8))
        app._available_yolo_labels = []
        app.refresh_ui = Mock()
        return app

    def test_failed_prompt_cannot_reuse_previous_target_or_prediction(self):
        app = self.app()
        app._yolo_detector.set_target_label.side_effect = RuntimeError("bad prompt")
        app._on_target_object_input()
        self.assertIsNone(app.selected_plan)
        self.assertIsNone(app._selected_detection)
        self.assertIsNone(app._current_prediction())
        self.assertIsNone(app._live_prediction)
        self.assertEqual(app._configured_target_label, "bottle")

    def test_paused_prompt_change_reinfers_frozen_image(self):
        app = self.app()
        prediction = YoloPrediction(app._frozen_bundle.color_rgb, [], "cpu")
        app._yolo_detector.predict.return_value = prediction
        app._auto_select_target_in_current_view = Mock(return_value=False)
        app._on_target_object_input()
        self.assertIs(app._yolo_detector.predict.call_args.args[0], app._frozen_bundle.color_rgb)
        self.assertIs(app._frozen_prediction, prediction)
        self.assertEqual(app._configured_target_label, "wooden block")
        self.assertIsNone(app.selected_plan)

    def test_paused_without_prediction_never_falls_back_to_live_results(self):
        app = self.app()
        app._frozen_prediction = None
        self.assertIsNone(app._current_prediction())

    def test_pausing_immediately_after_applying_category_infers_current_image(self):
        app = self.app()
        app._paused = False
        app._configured_target_label = "wooden block"
        app._live_prediction = None
        app._camera_bundle = app._frozen_bundle
        prediction = YoloPrediction(app._camera_bundle.color_rgb, [], "cpu")
        app._yolo_detector.predict.return_value = prediction
        app._auto_select_target_in_current_view = Mock(return_value=False)
        app.toggle_pause()
        self.assertTrue(app._paused)
        self.assertIs(app._frozen_prediction, prediction)
        self.assertIs(app._yolo_detector.predict.call_args.args[0], app._camera_bundle.color_rgb)

    def test_category_change_and_unpause_are_blocked_during_robot_action(self):
        app = self.app()
        app._action_running = True
        app.target_object_var = SimpleNamespace(set=Mock())
        original_plan = app.selected_plan
        app._on_target_object_input()
        app.toggle_pause()
        self.assertTrue(app._paused)
        self.assertIs(app.selected_plan, original_plan)
        app._yolo_detector.set_target_label.assert_not_called()


if __name__ == "__main__":
    main()
