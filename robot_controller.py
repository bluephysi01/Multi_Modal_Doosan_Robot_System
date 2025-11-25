#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Modal Robot Controller
음성명령 → 물체 인식 → 집기 → 홈 → 타겟위치 → 그리퍼 개방 → 홈
"""

import cv2
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import time
import math
from typing import Optional

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import DR_init

from vision_module import extract_instances_from_pcd, project_point
from config import APP_CONFIG

# ==============================================================================
# 설정 상수 (config 중앙화)
# ==============================================================================

ROBOT_ID = APP_CONFIG.robot.robot_id
ROBOT_MODEL = APP_CONFIG.robot.robot_model
VELOCITY, ACC = APP_CONFIG.robot.velocity, APP_CONFIG.robot.acceleration

HOME_POSJ_DEG = np.array(APP_CONFIG.robot.home_posj_deg, dtype=np.float32)
TARGET_POSJ_DEG = np.array(APP_CONFIG.robot.target_posj_deg, dtype=np.float32)

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


# ==============================================================================
# 메인 로봇 컨트롤러 노드
# ==============================================================================

class MultiModalRobotController(Node):
    def __init__(self, config=APP_CONFIG):
        super().__init__("multi_modal_robot_controller")
        self.config = config

        self.bridge = CvBridge()

        # 카메라 관련
        self.intrinsics = None
        self.latest_cv_color = None
        self.latest_cv_depth_mm = None

        # QoS 프로파일 설정 (RealSense 카메라 호환)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ROS 토픽 구독
        self.color_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/color/image_raw', qos_profile=qos_profile
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw', qos_profile=qos_profile
        )
        self.info_sub = message_filters.Subscriber(
            self, CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', qos_profile=qos_profile
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

        # DSR 로봇 모듈 로드 및 초기화
        try:
            # sys.modules에서 이미 import된 모듈 확인
            import sys
            if 'DSR_ROBOT2' not in sys.modules:
                import DSR_ROBOT2
            else:
                DSR_ROBOT2 = sys.modules['DSR_ROBOT2']

            from DSR_ROBOT2 import (
                wait, get_current_posj, movej, movel, amovej,
                set_robot_mode, ROBOT_MODE_AUTONOMOUS, get_current_posx
            )
            from DR_common2 import posj, posx

            self._wait = wait
            self._get_current_posj = get_current_posj
            self._get_current_posx = get_current_posx
            self._movej = movej
            self._movel = movel
            self._amovej = amovej
            self._posj = posj
            self._posx = posx
            self._set_robot_mode = set_robot_mode

            # 로봇 모드를 AUTONOMOUS로 설정
            try:
                set_robot_mode(ROBOT_MODE_AUTONOMOUS)
                self.get_logger().info("로봇 모드: AUTONOMOUS")
            except Exception as e:
                self.get_logger().warn(f"로봇 모드 설정 실패: {e}")

        except ImportError as e:
            self.get_logger().error(f"DSR_ROBOT2 임포트 실패: {e}")
            raise

        # 그리퍼 초기화
        self.gripper = None
        try:
            from dsr_example.simple.gripper_drl_controller import GripperController
            self.gripper = GripperController(node=self, namespace=ROBOT_ID)
            wait(2)
            if not self.gripper.initialize():
                self.get_logger().error("Gripper initialization failed!")
                raise Exception("Gripper initialization failed")
            self.get_logger().info("그리퍼 활성화 완료")
            self.gripper.move(0)  # 열기
        except Exception as e:
            self.get_logger().error(f"그리퍼 초기화 실패: {e}")
            rclpy.shutdown()

        # 상태 변수
        self.detected_objects = []
        self.is_robot_moving = False
        self.should_exit = False

        self.home_posj_deg = np.array(self.config.robot.home_posj_deg, dtype=np.float32)
        self.target_posj_deg = np.array(self.config.robot.target_posj_deg, dtype=np.float32)
        self.basket_posj_deg = np.array(self.config.robot.basket_posj_deg, dtype=np.float32)
        self.get_logger().info("Multi-Modal 시스템 초기화 완료")

    def synced_callback(self, color_msg, depth_msg, info_msg):
        try:
            self.latest_cv_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.latest_cv_depth_mm = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

            if not hasattr(self, '_data_received_count'):
                self._data_received_count = 0
            if self._data_received_count < 3:
                self.get_logger().info(f"카메라 데이터 수신: {self.latest_cv_color.shape}")
                self._data_received_count += 1

        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge 오류: {e}")
            return

        if self.intrinsics is None:
            self.intrinsics = rs.intrinsics()
            self.intrinsics.width = info_msg.width
            self.intrinsics.height = info_msg.height
            self.intrinsics.ppx = info_msg.k[2]
            self.intrinsics.ppy = info_msg.k[5]
            self.intrinsics.fx = info_msg.k[0]
            self.intrinsics.fy = info_msg.k[4]

            if info_msg.distortion_model == 'plumb_bob' or info_msg.distortion_model == 'rational_polynomial':
                self.intrinsics.model = rs.distortion.brown_conrady
            else:
                self.intrinsics.model = rs.distortion.none

            self.intrinsics.coeffs = list(info_msg.d)
            self.get_logger().info("카메라 intrinsics 수신 완료")

    def detect_objects(self):
        """물체 감지 (vision_module 사용)"""
        if self.latest_cv_depth_mm is None:
            self.get_logger().error("깊이 데이터 없음")
            return []
        if self.intrinsics is None:
            self.get_logger().error("카메라 intrinsics 없음")
            return []
        if self.latest_cv_color is None:
            self.get_logger().error("컬러 데이터 없음")
            return []

        self.get_logger().info("물체 감지 시작...")

        depth_scale = 0.001
        h, w = self.latest_cv_depth_mm.shape

        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        zs = self.latest_cv_depth_mm.astype(np.float32) * depth_scale

        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.ppx, self.intrinsics.ppy

        xs = (xs - cx) * zs / fx
        ys = (ys - cy) * zs / fy
        vtx = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)

        valid_mask = (zs.flatten() > 0.15) & (zs.flatten() < 0.40)
        valid_count = np.sum(valid_mask)
        self.get_logger().info(f"유효 포인트: {valid_count}")

        if valid_count < 1000:
            self.get_logger().warn(f"유효 포인트 부족: {valid_count}")
            return []

        vtx_valid = vtx[valid_mask]

        # vision_module의 extract_instances_from_pcd 사용
        filtered_points, labels, instances, ground_points = extract_instances_from_pcd(
            vtx_valid,
            self.latest_cv_color,
            valid_mask,
            fx, fy, cx, cy,
            depth_axis='z',
            voxel_size=0.005,
            dbscan_eps=0.04,
            dbscan_min_samples=20,
            global_min_pts=50
        )

        self.get_logger().info(f"감지된 물체: {len(instances)}개")

        if len(instances) == 0:
            return []

        # 중심점으로부터 거리 계산 및 픽셀 좌표 계산
        center_x, center_y = 0, 0
        for inst in instances:
            dist = np.sqrt((inst["x"] - center_x)**2 + (inst["y"] - center_y)**2)
            inst["distance_from_center"] = dist

            if inst["z"] > 0:
                inst["pixel_u"] = int(inst["x"] * fx / inst["z"] + cx)
                inst["pixel_v"] = int(inst["y"] * fy / inst["z"] + cy)
            else:
                inst["pixel_u"] = -1
                inst["pixel_v"] = -1

            # 바운딩 박스 계산
            cluster_pts = filtered_points[labels == inst["id"]]
            if len(cluster_pts) > 0:
                valid_pts = cluster_pts[cluster_pts[:, 2] > 0]
                if len(valid_pts) > 0:
                    pixels_u = (valid_pts[:, 0] * fx / valid_pts[:, 2] + cx).astype(int)
                    pixels_v = (valid_pts[:, 1] * fy / valid_pts[:, 2] + cy).astype(int)

                    pixels_u = np.clip(pixels_u, 0, w - 1)
                    pixels_v = np.clip(pixels_v, 0, h - 1)

                    inst["bbox_min_u"] = int(np.min(pixels_u))
                    inst["bbox_max_u"] = int(np.max(pixels_u))
                    inst["bbox_min_v"] = int(np.min(pixels_v))
                    inst["bbox_max_v"] = int(np.max(pixels_v))
                else:
                    inst["bbox_min_u"] = inst["pixel_u"] - 30
                    inst["bbox_max_u"] = inst["pixel_u"] + 30
                    inst["bbox_min_v"] = inst["pixel_v"] - 30
                    inst["bbox_max_v"] = inst["pixel_v"] + 30
            else:
                inst["bbox_min_u"] = inst["pixel_u"] - 30
                inst["bbox_max_u"] = inst["pixel_u"] + 30
                inst["bbox_min_v"] = inst["pixel_v"] - 30
                inst["bbox_max_v"] = inst["pixel_v"] + 30

        instances_sorted = sorted(instances, key=lambda x: x["distance_from_center"])
        return instances_sorted

    def pick_object(self, obj):
        """물체 집기"""
        DR_TOOL = 1
        DR_MV_MOD_REL = 1
        cfg = self.config.robot

        cam_x, cam_y, cam_z = obj['x'], obj['y'], obj['z']
        yaw = obj['yaw']
        length = obj['length']

        try:
            # 그리퍼 열기
            self.get_logger().info("그리퍼 열기...")
            self.gripper.move(0)
            self._wait(2.0)

            self.is_robot_moving = True

            # 물체 위치 계산
            target_x = cam_x * 1000
            target_y = cam_y * 1000 + cfg.gripper_offset_y
            target_z = cam_z * 1000 + cfg.gripper_offset_z
            yaw_deg = math.degrees(yaw) + cfg.yaw_offset_deg

            self.get_logger().info(f"목표: X={target_x:.1f}, Y={target_y:.1f}, Z={target_z:.1f}")

            # 물체 위치로 이동
            target_pos = self._posx(target_x, target_y, target_z, 0, 0, 0)
            self._movel(target_pos, vel=VELOCITY, acc=ACC, ref=DR_TOOL, mod=DR_MV_MOD_REL)
            self._wait(1.0)

            # Joint 6 회전
            current_joints = self._get_current_posj()
            target_joints = self._posj(current_joints[0], current_joints[1], current_joints[2],
                           current_joints[3], current_joints[4], yaw_deg)
            self._movej(target_joints, vel=VELOCITY, acc=ACC)
            self._wait(1.5)

            # Z축 조정
            self._movel(self._posx(0,0,50,0,0,0), vel=VELOCITY, acc=ACC, ref=DR_TOOL, mod=DR_MV_MOD_REL)
            self._wait(1.0)

            # 그리퍼 닫기
            gripper_position = 680 if length < 0.03 else 550
            self.get_logger().info(f"그리퍼 닫기: {gripper_position}")
            self.gripper.move(gripper_position)
            self._wait(3.0)

            # 홈으로 복귀
            self.get_logger().info("홈 위치로 복귀")
            self.move_home()

            self.is_robot_moving = False
            return True

        except Exception as e:
            self.get_logger().error(f"물체 집기 실패: {e}")
            self.is_robot_moving = False
            return False

    def place_object_at_target(self, *, open_gripper: bool = True, return_home: bool = True):
        """타겟 위치로 이동 (옵션: 놓기/홈 복귀)"""
        try:
            self.is_robot_moving = True

            self.get_logger().info("타겟 위치로 이동 중...")
            target_pos = self._posj(*self.target_posj_deg)
            self._movej(target_pos, vel=VELOCITY, acc=ACC)
            self._wait(3.0)

            if open_gripper:
                # 그리퍼 열기 (물체 놓기)
                self.get_logger().info("그리퍼 열기 (물체 놓기)...")
                self.gripper.move(0)
                self.get_logger().info("2초 대기 중...")
                self._wait(2.0)  # 그리퍼 열고 2초 대기

            if return_home:
                # 홈으로 복귀
                self.get_logger().info("홈 위치로 복귀")
                self.move_home()

            self.is_robot_moving = False
            self.get_logger().info("✅ 홈 위치 복귀 완료")
            return True

        except Exception as e:
            self.get_logger().error(f"물체 놓기 실패: {e}")
            self.is_robot_moving = False
            return False

    def place_object_in_basket(self):
        """바구니 위치로 이동해 물체를 놓고 홈으로 복귀"""
        try:
            self.is_robot_moving = True
            wait_time = getattr(self.config.yolo, "rest_wait_after_pose", 2.0)

            self.get_logger().info("바구니 위치로 이동 중...")
            basket_pos = self._posj(*self.basket_posj_deg)
            self._movej(basket_pos, vel=VELOCITY, acc=ACC)
            self._wait(3.0)

            if wait_time > 0:
                self._wait(wait_time)

            self.get_logger().info("바구니 위치에서 그리퍼 개방")
            self.gripper.move(0)
            if wait_time > 0:
                self._wait(wait_time)

            self.get_logger().info("바구니 전달 완료 → 홈 복귀")
            self.move_home()

            self.is_robot_moving = False
            return True
        except Exception as e:
            self.get_logger().error(f"바구니 전달 실패: {e}")
            self.is_robot_moving = False
            return False

    def move_home(self):
        home_pos = self._posj(*self.home_posj_deg)
        self._movej(home_pos, vel=VELOCITY, acc=ACC)
        self._wait(3.0)

    def terminate_gripper(self):
        if self.gripper:
            self.gripper.terminate()
