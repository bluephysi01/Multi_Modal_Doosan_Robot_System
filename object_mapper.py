"""Mapping utilities between detected point-cloud instances and dialogue labels."""

from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class LabeledObject:
    """Container for a detected object enriched with UI-friendly metadata."""

    label: str
    center_xyz: Tuple[float, float, float]
    pixel_uv: Tuple[int, int]
    yaw_deg: float
    grip_length: float
    raw: dict

    def to_prompt_line(self) -> str:
        x, y, z = self.center_xyz
        px, py = self.pixel_uv
        return (
            f"{self.label}: pos=({x:.3f},{y:.3f},{z:.3f})m, yaw={self.yaw_deg:.1f}deg, "
            f"span={self.grip_length*1000:.0f}mm, pixel=({px},{py})"
        )


@dataclass
class SnapshotPayload:
    """Frozen view of the workspace plus metadata for the realtime conversation."""

    labeled_objects: List[LabeledObject]
    original_frame: np.ndarray  # AI에 전송할 원본 이미지 (어노테이션 없음)
    annotated_frame: np.ndarray  # 사용자에게 보여줄 이미지 (바운딩 박스 포함)
    summary_text: str
    captured_at: float

    def to_base64(self, mime: str = "image/png") -> Tuple[str, str]:
        """Return (data_url, mime) tuple for websocket payloads. 원본 이미지 사용."""
        if mime == "image/jpeg":
            ext = ".jpg"
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            success, buffer = cv2.imencode(ext, self.original_frame, encode_param)
        else:
            ext = ".png"
            success, buffer = cv2.imencode(ext, self.original_frame)

        if not success:
            raise RuntimeError("Failed to encode original snapshot")
        image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        return f"data:{mime};base64,{image_b64}", mime

    def find_object(self, label: str) -> Optional[LabeledObject]:
        for obj in self.labeled_objects:
            if obj.label == label:
                return obj
        return None


class ObjectMapper:
    """Assigns stable labels to detections and parses operator commands."""

    LABEL_PATTERN = re.compile(r"OBJ[-_ ]?(\d{1,2})", re.IGNORECASE)

    def __init__(self) -> None:
        self._last_snapshot: Optional[SnapshotPayload] = None

    def build_snapshot(self, color_frame_bgr: np.ndarray, instances: Sequence[dict]) -> Optional[SnapshotPayload]:
        if color_frame_bgr is None or len(instances) == 0:
            return None

        labeled: List[LabeledObject] = []
        for idx, inst in enumerate(instances, start=1):
            label = f"OBJ-{idx:02d}"
            pixel_u = int(inst.get("pixel_u", -1))
            pixel_v = int(inst.get("pixel_v", -1))
            yaw_rad = float(inst.get("yaw", 0.0))
            obj = LabeledObject(
                label=label,
                center_xyz=(float(inst.get("x", 0.0)), float(inst.get("y", 0.0)), float(inst.get("z", 0.0))),
                pixel_uv=(pixel_u, pixel_v),
                yaw_deg=float(np.degrees(yaw_rad)),
                grip_length=float(inst.get("length", 0.0)),
                raw=inst,
            )
            labeled.append(obj)

        # AI 전송용 이미지: 작은 번호 라벨만 추가
        original = self._annotate_minimal(color_frame_bgr.copy(), labeled)
        # 사용자 표시용: 전체 바운딩 박스 포함
        annotated = self._annotate_frame(color_frame_bgr.copy(), labeled)

        # 간단한 요약
        obj_labels = ", ".join([obj.label for obj in labeled])
        summary_text = (
            f"📸 테이블 스냅샷\n"
            f"감지된 물체: {len(labeled)}개 ({obj_labels})\n\n"
            f"이미지를 보고 각 물체가 무엇인지 간단히 말하세요.\n"
            f"형식: 'OBJ-01은 [물체이름]입니다. OBJ-02는 [물체이름]입니다.' 식으로 나열하고,\n"
            f"마지막에 '어떤 물체를 집을까요?'라고 물으세요."
        )

        snapshot = SnapshotPayload(
            labeled_objects=labeled,
            original_frame=original,
            annotated_frame=annotated,
            summary_text=summary_text,
            captured_at=time.time(),
        )
        self._last_snapshot = snapshot
        return snapshot

    def resolve_label(self, transcript: str, snapshot: Optional[SnapshotPayload] = None) -> Optional[LabeledObject]:
        snap = snapshot or self._last_snapshot
        if snap is None or not transcript:
            return None

        # PICK_LABEL: 키워드가 있는 경우에만 라벨 추출
        if "PICK_LABEL:" not in transcript:
            return None

        # PICK_LABEL: 이후의 OBJ-XX 패턴만 찾기
        pick_label_part = transcript.split("PICK_LABEL:")[-1]
        match = self.LABEL_PATTERN.search(pick_label_part)
        if not match:
            return None
        idx = int(match.group(1))
        label = f"OBJ-{idx:02d}"
        return snap.find_object(label)

    def _annotate_minimal(self, frame: np.ndarray, labeled: Sequence[LabeledObject]) -> np.ndarray:
        """AI 전송용: 작은 번호만 표시 (바운딩 박스 없음)"""
        if frame is None:
            return frame

        h, w = frame.shape[:2]

        for idx, obj in enumerate(labeled, start=1):
            u, v = obj.pixel_uv
            if u < 0 or v < 0 or u >= w or v >= h:
                continue

            # 작은 숫자만 표시 (반 크기)
            number = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            radius = 12

            # 배경 원
            cv2.circle(frame, (u, v), radius, (0, 0, 0), -1)  # 검은 배경
            cv2.circle(frame, (u, v), radius, (0, 255, 0), 2)  # 초록 테두리

            # 숫자
            (text_w, text_h), _ = cv2.getTextSize(number, font, font_scale, thickness)
            text_x = u - text_w // 2
            text_y = v + text_h // 2
            cv2.putText(frame, number, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        return frame

    def _annotate_frame(self, frame: np.ndarray, labeled: Sequence[LabeledObject]) -> np.ndarray:
        if frame is None:
            return frame

        h, w = frame.shape[:2]

        for obj in labeled:
            u, v = obj.pixel_uv
            if u < 0 or v < 0 or u >= w or v >= h:
                continue

            # 바운딩 박스 그리기
            raw = obj.raw
            if "bbox_min_u" in raw and "bbox_max_u" in raw:
                bbox_min_u = int(raw["bbox_min_u"])
                bbox_max_u = int(raw["bbox_max_u"])
                bbox_min_v = int(raw["bbox_min_v"])
                bbox_max_v = int(raw["bbox_max_v"])

                cv2.rectangle(
                    frame,
                    (bbox_min_u, bbox_min_v),
                    (bbox_max_u, bbox_max_v),
                    (0, 255, 0),
                    2
                )

            # 중심점 원 그리기
            cv2.circle(frame, (u, v), 8, (0, 255, 255), -1)
            cv2.circle(frame, (u, v), 8, (0, 0, 255), 2)

            # 라벨 배경 그리기
            label_text = obj.label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            if "bbox_min_v" in raw:
                label_x = int(raw["bbox_min_u"])
                label_y = int(raw["bbox_min_v"]) - 10
            else:
                label_x = u + 8
                label_y = v - 8

            # 배경 사각형
            cv2.rectangle(
                frame,
                (label_x, label_y - text_h - baseline),
                (label_x + text_w, label_y + baseline),
                (0, 0, 0),
                -1
            )

            # 텍스트
            cv2.putText(
                frame,
                label_text,
                (label_x, label_y),
                font,
                font_scale,
                (0, 255, 255),
                thickness,
            )

        return frame

    @staticmethod
    def format_decision_message(obj: LabeledObject) -> str:
        return (
            f"Picked {obj.label} at pos=({obj.center_xyz[0]:.3f},{obj.center_xyz[1]:.3f},{obj.center_xyz[2]:.3f})m "
            f"with yaw={obj.yaw_deg:.1f}deg"
        )
