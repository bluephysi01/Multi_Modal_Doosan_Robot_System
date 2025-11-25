"""Shared helpers reused by multimodal pick/place and YOLO pose tracking."""

from __future__ import annotations

import time
from typing import Optional

import cv2


def safe_imshow(window_name: str, image, retry: int = 3) -> bool:
    """Update an OpenCV window with basic retry logic (Wayland friendly)."""
    for attempt in range(retry):
        try:
            cv2.imshow(window_name, image)
            cv2.waitKey(1)
            return True
        except Exception as exc:  # pragma: no cover - UI guard
            if attempt == retry - 1:
                print(f"⚠️  윈도우 업데이트 실패 ({window_name}): {exc}")
                return False
            time.sleep(0.01)
    return False


def wait_for_camera_stream(controller, *, max_wait_sec: float = 10.0, poll: float = 0.1) -> bool:
    """Block until the controller reports at least one camera frame."""
    print("카메라 데이터 수신 대기 중...", end="", flush=True)
    waited = 0.0
    while controller.latest_cv_color is None and waited < max_wait_sec:
        time.sleep(poll)
        waited += poll
        if int(waited / poll) % int(1.0 / poll) == 0:
            print(".", end="", flush=True)

    if controller.latest_cv_color is None:
        print("\n\n❌ 카메라 데이터 수신 실패!")
        return False

    print(" ✓ 완료!\n")
    return True

