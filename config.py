"""Central configuration for the integrated multi-modal + YOLO pose workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


@dataclass
class PathsConfig:
    snapshot_dir: Path = field(default_factory=lambda: BASE_DIR / "snapshots")


@dataclass
class RobotMotionConfig:
    robot_id: str = "dsr01"
    robot_model: str = "e0509"
    velocity: float = 60.0
    acceleration: float = 60.0
    home_posj_deg: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        90.0,
        0.0,
        90.0,
        0.0,
    )
    target_posj_deg: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        90.0,
        -90.0,
        90.0,
        0.0,
    )
    basket_posj_deg: tuple[float, float, float, float, float, float] = (
        -50.0,
        30.0,
        90.0,
        0.0,
        60.0,
        -50.0,
    )
    gripper_offset_y: float = -60.0
    gripper_offset_z: float = -150.0
    yaw_offset_deg: float = -100.0


@dataclass
class VisionDetectionConfig:
    detection_interval: float = 0.2
    tracker_max_age: int = 1.0
    tracker_dist_thresh: float = 0.05
    snapshot_wait: float = 1.0
    gui_update_interval: float = 0.2


@dataclass
class RealtimeConfig:
    model: str = "gpt-realtime-mini"
    operator_audio_seconds: int = 5
    max_turns: int = 3
    play_audio: bool = True


@dataclass
class YoloPoseConfig:
    model_path: str = "yolo11n-pose.pt"
    device: str = "cuda"
    scale_x: float = 0.5
    scale_y: float = 0.5
    deadzone_pixels: float = 40.0
    idle_seconds_before_exit: float = 3.0
    control_period: float = 0.5
    stale_timeout: float = 1.0
    vel_lin: float = 60.0
    vel_ang: float = 60.0
    acc_lin: float = 60.0
    acc_ang: float = 60.0
    exit_lift_dz_mm: float = 200.0
    exit_wait_after_lift: float = 3.0
    exit_open_stroke: int = 0
    exit_close_stroke: int = 700
    exit_wait_after_close: float = 5.0
    overall_timeout: float = 120.0
    start_posj_deg: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        90.0,
        -90.0,
        90.0,
        0.0,
    )
    start_lift_dz_mm: float = 200.0
    rest_wait_after_pose: float = 3.0
    target_iou_thresh: float = 0.2
    target_miss_frames: int = 30


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    robot: RobotMotionConfig = field(default_factory=RobotMotionConfig)
    vision: VisionDetectionConfig = field(default_factory=VisionDetectionConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    yolo: YoloPoseConfig = field(default_factory=YoloPoseConfig)


APP_CONFIG = AppConfig()
