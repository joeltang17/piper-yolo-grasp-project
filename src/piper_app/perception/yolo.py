from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from piper_app.config import repo_root


@dataclass
class YoloConfig:
    enabled: bool = False
    weights_path: str = "third_party/yolo/新松-检测/yolo11m.pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 50
    device: str = "auto"


@dataclass
class YoloDetection:
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    # Boolean mask in the original RGB/depth-aligned image coordinates.
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class YoloPrediction:
    annotated_rgb: np.ndarray
    detections: list[YoloDetection]
    device_label: str


def render_detections(image_rgb: np.ndarray, detections: list[YoloDetection], *, show_masks: bool = True) -> np.ndarray:
    rgb = image_rgb.copy()
    if show_masks:
        color = np.array([34, 197, 94], dtype=np.float32)
        for detection in detections:
            if detection.mask is not None:
                if detection.mask.shape != rgb.shape[:2]:
                    raise ValueError("Detection mask must match the original image dimensions.")
                rgb[detection.mask] = (0.65 * rgb[detection.mask] + 0.35 * color).astype(np.uint8)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        draw.rectangle((x1, y1, x2, y2), outline="#22c55e", width=3)
        text = f"{detection.label} {detection.confidence:.2f}"
        text_xy = (max(0, x1), max(0, y1 - 18))
        draw.rectangle(draw.textbbox(text_xy, text), fill="#22c55e")
        draw.text(text_xy, text, fill="black")
    return np.asarray(image)


def _instance_masks(result, image_shape: tuple[int, int]) -> list[np.ndarray]:
    masks = getattr(result, "masks", None)
    if masks is None:
        return []
    data = masks.data.detach().cpu().numpy()
    if data.ndim != 3:
        raise RuntimeError("YOLO returned an invalid instance-mask array.")
    if data.shape[1:] == image_shape:
        return [np.array(mask > 0.5, dtype=bool, copy=True) for mask in data]

    # Older runtimes may return letterboxed masks. Their xy polygons are already
    # mapped to the original image; stretching the letterbox would shift pixels.
    polygons = masks.xy
    if len(polygons) != len(data):
        raise RuntimeError("YOLO mask polygons do not match the instance count.")
    output = []
    for polygon in polygons:
        mask = np.zeros(image_shape, dtype=np.uint8)
        polygon = np.asarray(polygon)
        if polygon.size:
            cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        output.append(mask.astype(bool))
    return output


def _vendor_root() -> Path:
    return repo_root() / "third_party" / "yolo" / "新松-检测"


def _import_yolo_runtime():
    try:
        from ultralytics import YOLO
        return YOLO
    except ModuleNotFoundError:
        vendor_root = _vendor_root()
        if str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))
        try:
            from ultralytics import YOLO
            return YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError("Ultralytics is not installed. Install the project dependencies first.") from exc


def _model_class_names(model) -> list[str]:
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


class YoloDetector:
    def __init__(self, config: YoloConfig):
        self.config = config
        self._model = None
        self._torch = None
        self._device_label = "uninitialized"
        self._class_names: list[str] = []
        self._default_class_names: list[str] = []
        self._supports_text_prompts = False
        self._target_label = ""

    @property
    def device_label(self) -> str:
        return self._device_label

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def suggested_class_names(self) -> list[str]:
        return list(dict.fromkeys(self._default_class_names + self._class_names))

    @property
    def supports_text_prompts(self) -> bool:
        return self._supports_text_prompts

    def open(self) -> None:
        if self._model is not None:
            return
        weights_path = Path(self.config.weights_path)
        if not weights_path.is_absolute():
            weights_path = repo_root() / weights_path
        weights_path = weights_path.resolve()
        if not weights_path.exists():
            raise RuntimeError(f"YOLO weights not found: {weights_path}")
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("Torch is not installed. Install the project dependencies first.") from exc

        model = _import_yolo_runtime()(str(weights_path))
        underlying = getattr(model, "model", None)
        layers = getattr(underlying, "model", None)
        head = layers[-1] if layers is not None else None
        promptable = (
            callable(getattr(model, "set_classes", None))
            and not hasattr(head, "lrpc")
            and not bool(getattr(head, "is_fused", False))
        )
        default_names = _model_class_names(model)
        if promptable and type(model).__name__ == "YOLOE":
            from ultralytics.utils import ROOT, YAML
            coco_names = YAML.load(ROOT / "cfg/datasets/coco.yaml")["names"]
            default_names = [str(coco_names[index]) for index in sorted(coco_names)]
        if promptable:
            model.set_classes([self._target_label] if self._target_label else default_names)
        elif self._target_label:
            self._validate_fixed_label(self._target_label, default_names)

        # Publish the loaded model only after prompt configuration succeeds.
        self._torch = torch
        self._model = model
        self._supports_text_prompts = promptable
        self._default_class_names = default_names
        self._class_names = _model_class_names(model)
        self._device_label = self._resolve_device()

    @staticmethod
    def _validate_fixed_label(label: str, names: list[str]) -> None:
        if label.casefold() not in {name.casefold() for name in names}:
            raise RuntimeError(
                f"'{label}' is not a class in these fixed-vocabulary weights. "
                "Use a promptable YOLOE *-seg.pt checkpoint for custom categories."
            )

    def set_target_label(self, label: str) -> None:
        label = str(label).strip()
        previous = self._target_label
        if self._model is None:
            self._target_label = label
            try:
                self.open()
            except Exception:
                self._target_label = previous
                raise
            return
        if label == previous:
            return
        if self._supports_text_prompts:
            classes = [label] if label else self._default_class_names
            try:
                self._model.set_classes(classes)
            except Exception:
                # A failed backend update must not leave a partially changed
                # model available to a later prediction.
                self.close()
                raise
            self._class_names = _model_class_names(self._model)
        elif label:
            self._validate_fixed_label(label, self._class_names)
        self._target_label = label

    def predict(self, image_rgb: np.ndarray) -> YoloPrediction:
        if self._model is None:
            self.open()
        results = self._model.predict(
            np.ascontiguousarray(image_rgb[:, :, ::-1]),
            imgsz=int(self.config.imgsz), conf=float(self.config.conf),
            iou=float(self.config.iou), max_det=int(self.config.max_det),
            device=self._device_label, verbose=False, retina_masks=True,
        )
        result = results[0]
        annotated_rgb = result.plot()[:, :, ::-1].copy()
        detections: list[YoloDetection] = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            masks = _instance_masks(result, image_rgb.shape[:2])
            if getattr(result, "masks", None) is not None and len(masks) != len(boxes):
                raise RuntimeError("YOLO returned different numbers of boxes and instance masks.")
            names = getattr(result, "names", {}) or {}
            for index, (xyxy, conf, cls_idx) in enumerate(zip(
                boxes.xyxy.detach().cpu().tolist(),
                boxes.conf.detach().cpu().tolist(),
                boxes.cls.detach().cpu().tolist(),
            )):
                cls_id = int(cls_idx)
                label = str(names.get(cls_id, cls_id)) if isinstance(names, dict) else str(names[cls_id])
                detections.append(YoloDetection(
                    label=label, confidence=float(conf),
                    bbox_xyxy=tuple(int(round(value)) for value in xyxy),
                    mask=masks[index] if masks else None,
                ))
        return YoloPrediction(annotated_rgb, detections, self._device_label)

    def close(self) -> None:
        self._model = None
        self._torch = None
        self._device_label = "uninitialized"
        self._class_names = []
        self._default_class_names = []
        self._supports_text_prompts = False

    def _resolve_device(self) -> str:
        requested = str(self.config.device).strip().lower()
        if requested and requested != "auto":
            return requested
        if self._torch is not None and self._torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
