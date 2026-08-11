"""record.py --yolo=true 용 실시간 YOLO 바코드 검출기.

robot.get_observation()이 반환하는 카메라별 (H, W, 3) uint8 RGB numpy 배열에
학습된 YOLOv8 바코드 모델(BarcodeDetection/yolov8_barcode_v6)을 프레임마다 돌려서,
카메라별로 바코드가 지금 잡히는지 여부/confidence를 알려준다.

--yolo=false(기본값)면 이 모듈은 아예 임포트되지 않고 ultralytics도 필요 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# BarcodeDetection 저장소(runs/detect/barcode/yolov8_barcode_v6)에서 검증된 최신 가중치.
# 그리퍼/바구니 오탐은 없지만, right_wrist 외 카메라의 실질 검출력은 아직 약함 — 자세한
# 내용은 BarcodeDetection 리포의 Notion 문서("v4 -> v6" 항목) 참고.
DEFAULT_YOLO_MODEL_PATH = str(Path(__file__).resolve().parents[1] / "assets" / "yolo" / "barcode_v6.pt")


@dataclass
class BarcodeDetection:
    camera_key: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


class BarcodeDetector:
    """카메라 프레임에서 YOLO로 바코드를 검출하는 얇은 래퍼."""

    def __init__(
        self,
        model_path: str = DEFAULT_YOLO_MODEL_PATH,
        conf: float = 0.4,
        camera_keys: list[str] | None = None,
    ):
        from ultralytics import YOLO  # 지연 임포트: --yolo=false면 ultralytics 없어도 동작해야 함

        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO 바코드 모델을 찾을 수 없음: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        # None이면 obs에 들어있는 이미지 키(ndim==3인 배열) 전부를 대상으로 함
        self.camera_keys = camera_keys

    def image_keys(self, obs: dict) -> list[str]:
        if self.camera_keys is not None:
            return self.camera_keys
        return [k for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim == 3]

    def detect(self, obs: dict) -> list[BarcodeDetection]:
        """obs(robot.get_observation() 결과)에서 카메라 이미지 키들만 골라 검출."""
        detections: list[BarcodeDetection] = []
        for key in self.image_keys(obs):
            frame = obs.get(key)
            if not isinstance(frame, np.ndarray) or frame.ndim != 3:
                continue
            # MuJoCo 렌더(env.render_images)는 RGB, ultralytics는 BGR 기준이라 채널을 뒤집어준다.
            frame_bgr = np.ascontiguousarray(frame[:, :, ::-1])
            result = self.model.predict(source=frame_bgr, conf=self.conf, verbose=False)[0]
            for box, conf_t in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True):
                detections.append(
                    BarcodeDetection(camera_key=key, confidence=float(conf_t), box_xyxy=tuple(box))
                )
        return detections

    def format_status(self, detections: list[BarcodeDetection], obs: dict) -> str:
        """카메라별 검출 여부를 한 줄로 요약 (콘솔 로그용)."""
        best_by_cam: dict[str, float] = {}
        for d in detections:
            best_by_cam[d.camera_key] = max(best_by_cam.get(d.camera_key, 0.0), d.confidence)
        parts = []
        for key in self.image_keys(obs):
            conf = best_by_cam.get(key)
            mark = f"✅{conf:.2f}" if conf else "❌"
            parts.append(f"{key}={mark}")
        return " | ".join(parts)
