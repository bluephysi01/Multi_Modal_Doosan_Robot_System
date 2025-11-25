#!/usr/bin/env python3

import rclpy
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
import cv2
from ultralytics import YOLO
import numpy as np
from dsr_msgs2.srv import MoveLine, GetCurrentPosx, DrlStart, MoveJoint
import threading

from config import APP_CONFIG

# DRL 스크립트 베이스 (사용자 제공 로직과 동일)
DRL_BASE_CODE = """
g_slaveid = 0
flag = 0
def modbus_set_slaveid(slaveid):
    global g_slaveid
    g_slaveid = slaveid
def modbus_fc06(address, value):
    global g_slaveid
    data = (g_slaveid).to_bytes(1, byteorder='big')
    data += (6).to_bytes(1, byteorder='big')
    data += (address).to_bytes(2, byteorder='big')
    data += (value).to_bytes(2, byteorder='big')
    return modbus_send_make(data)
def modbus_fc16(startaddress, cnt, valuelist):
    global g_slaveid
    data = (g_slaveid).to_bytes(1, byteorder='big')
    data += (16).to_bytes(1, byteorder='big')
    data += (startaddress).to_bytes(2, byteorder='big')
    data += (cnt).to_bytes(2, byteorder='big')
    data += (2 * cnt).to_bytes(1, byteorder='big')
    for i in range(0, cnt):
        data += (valuelist[i]).to_bytes(2, byteorder='big')
    return modbus_send_make(data)
def recv_check():
    size, val = flange_serial_read(0.1)
    if size > 0:
        return True, val
    else:
        tp_log("CRC Check Fail")
        return False, val
def gripper_move(stroke):
    flange_serial_write(modbus_fc16(282, 2, [stroke, 0]))
    wait(1.0) # 물리적 동작 시간을 충분히 기다려줍니다.

# ---- init serial & torque/current ----
while True:
    flange_serial_open(
        baudrate=57600,
        bytesize=DR_EIGHTBITS,
        parity=DR_PARITY_NONE,
        stopbits=DR_STOPBITS_ONE,
    )

    modbus_set_slaveid(1)

    # 256(40257) Torque enable
    # 275(40276) Goal Current
    # 282(40283) Goal Position

    flange_serial_write(modbus_fc06(256, 1))   # torque enable
    flag, val = recv_check()

    flange_serial_write(modbus_fc06(275, 400)) # goal current
    flag, val = recv_check()

    if flag is True:
        break

    flange_serial_close()
"""


class YoloHandRobotControl(Node):
    def __init__(self, config=APP_CONFIG, realtime_client=None):
        super().__init__('yolo_hand_robot_control')
        self.config = config
        self.realtime_client = realtime_client
        ycfg = self.config.yolo

        # YOLO 모델 로드
        self.get_logger().info('YOLO 모델 로딩 중...')
        self.model = YOLO(ycfg.model_path)
        self.model.to(ycfg.device)
        self.get_logger().info('YOLO 모델 로딩 완료!')

        # 이미지 구독자 생성
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        # 결과 이미지 발행자 생성
        self.publisher = self.create_publisher(
            Image,
            '/yolo_pose/image',
            10
        )

        # 오른손 끝 좌표 발행자 생성
        self.hand_publisher = self.create_publisher(
            Point,
            '/right_wrist_position',
            10
        )

        # 두산 로봇 서비스 클라이언트 생성
        self.move_line_client = self.create_client(MoveLine, '/dsr01/motion/move_line')
        self.get_current_posx_client = self.create_client(GetCurrentPosx, '/dsr01/aux_control/get_current_posx')
        self.move_joint_client = self.create_client(MoveJoint, '/dsr01/motion/move_joint')

        # 서비스 대기
        while not self.move_line_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('두산 로봇 MoveLine 서비스 대기 중...')

        while not self.get_current_posx_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('두산 로봇 GetCurrentPosx 서비스 대기 중...')

        while not self.move_joint_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('두산 로봇 MoveJoint 서비스 대기 중...')

        self.get_logger().info('두산 로봇 서비스 연결 완료!')

        # DRL 스크립트 서비스 (그리퍼 제어용)
        self.drl_client = self.create_client(DrlStart, '/dsr01/drl/drl_start')
        while not self.drl_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('두산 로봇 DrlStart 서비스 대기 중...')

        # 이미지 크기
        self.image_width = 640
        self.image_height = 480

        # Eye-in-hand Visual Servoing 파라미터
        self.target_pixel_x = self.image_width / 2
        self.target_pixel_y = self.image_height / 2
        self.scale_x = ycfg.scale_x
        self.scale_y = ycfg.scale_y

        # 제어 데드존 (픽셀)
        self.deadzone_pixels = ycfg.deadzone_pixels

        # 로봇 제어 파라미터
        self.vel_lin = ycfg.vel_lin
        self.vel_ang = ycfg.vel_ang
        self.acc_lin = ycfg.acc_lin
        self.acc_ang = ycfg.acc_ang

        # 손목 키포인트 인덱스 (COCO format)
        self.RIGHT_WRIST_INDEX = 10  # 오른손 손목

        # 이동 제어 스레드 락
        self.move_lock = threading.Lock()
        self.last_move_time = time.time()  # 마지막 이동 명령 시간
        self.last_stop_time = 0.0  # 마지막 stop 전송 시간

        # 최신 명령만 적용하기 위한 버퍼 (stale drop)
        self.latest_error_x = None
        self.latest_error_y = None
        self.latest_error_time = 0.0
        self.stale_timeout = ycfg.stale_timeout  # s, 이 시간 이상 지난 명령은 무시
        self.control_period = ycfg.control_period  # s, 명령 적용 주기
        self.control_timer = self.create_timer(self.control_period, self._control_timer_cb)

        # FPS 계산을 위한 변수
        self.frame_count = 0
        self.start_time = self.get_clock().now()

        # 종료 절차 플래그
        self.exit_started = False
        self.exit_complete_event = threading.Event()

        # 외부 제어 플래그
        self.tracking_enabled = False
        self.should_exit = False
        self.active_target_bbox = None  # 최근 추적 대상 bbox (x1,y1,x2,y2)
        self.target_miss_count = 0
        self.awaiting_consent = False
        self.last_color_frame = None
        self.last_annotated_frame = None

    def enable_tracking(self):
        self.get_logger().info("YOLO 포즈 트래킹 활성화")
        # 현재 위치 기준 +Z 리프트 후 트래킹 시작
        lift = self.config.yolo.start_lift_dz_mm
        if lift and lift != 0.0:
            self.get_logger().info(f"초기 트래킹 위치: TOOL +Z {lift}mm 이동")
            self.move_tool_relative_blocking(0.0, 0.0, lift)
        self.tracking_enabled = True
        self.exit_started = False
        self.exit_complete_event.clear()
        self.last_move_time = time.time()
        self.latest_error_x = None
        self.latest_error_y = None
        self._reset_tracking_target()
        self.awaiting_consent = False

    def disable_tracking(self):
        self.get_logger().info("YOLO 포즈 트래킹 비활성화")
        self.tracking_enabled = False
        self.latest_error_x = None
        self.latest_error_y = None
        self._reset_tracking_target()
        self.awaiting_consent = False

    def wait_for_exit_sequence(self, timeout: float = None) -> bool:
        return self.exit_complete_event.wait(timeout)

    def close_gripper(self, stroke: int = 700):
        """그리퍼 닫기 (기본 700)"""
        self._run_gripper_script(stroke, action_label=f"그리퍼 닫기({stroke})")

    def open_gripper(self, stroke: int = 0):
        """그리퍼 완전 개방 (기본 0)"""
        self._run_gripper_script(stroke, action_label=f"그리퍼 개방({stroke})")

    def move_home_blocking(self, joints=None):
        """지정 조인트 포즈로 이동 후 완료까지 대기."""
        if joints is None:
            joints = list(self.config.robot.home_posj_deg)
        if not self.move_joint_client.service_is_ready():
            self.get_logger().warn('MoveJoint 서비스 준비 안됨: 홈 복귀 스킵')
            return
        req = MoveJoint.Request()
        req.pos = [float(x) for x in joints]
        req.vel = 50.0
        req.acc = 50.0
        future = self.move_joint_client.call_async(req)
        try:
            future.result()
        except Exception as e:
            self.get_logger().warn(f"홈 복귀 대기 실패/타임아웃: {e}")

    def start_exit_sequence(self):
        """
        데드존에서 정지 후:
        +Z 이동 → 그리퍼 개방 → 기다림 → 그리퍼 닫기 → 홈 복귀 → 완료 알림
        """
        if self.exit_started:
            return
        self.exit_started = True
        self.tracking_enabled = False
        self.exit_complete_event.clear()
        self.latest_error_x = None
        self.latest_error_y = None
        self._reset_tracking_target()
        self.awaiting_consent = False

        def _seq():
            try:
                ycfg = self.config.yolo
                # 1) TOOL +Z 이동
                self.get_logger().info("데드존 정지 → +Z 이동 시작 (동기 MoveLine)")
                moved = self.move_tool_relative_blocking(0.0, 0.0, ycfg.exit_lift_dz_mm)
                if moved:
                    self.get_logger().info(f"+Z 상승 완료 ({ycfg.exit_lift_dz_mm}mm)")
                else:
                    self.get_logger().warn("TOOL +Z 이동 실패 또는 시간 초과 → 그대로 진행")

                # 2) 이동 후 대기
                time.sleep(ycfg.exit_wait_after_lift)

                # 3) 그리퍼 완전 개방
                self.get_logger().info("그리퍼 완전 개방")
                opened = self._run_gripper_script(ycfg.exit_open_stroke, action_label="종료 시퀀스 개방")
                if not opened:
                    self.get_logger().warn("그리퍼 개방 명령 실패/타임아웃")

                # 4) 대기
                time.sleep(2.0)

                # 5) 그리퍼 닫기
                self.get_logger().info("그리퍼 닫기")
                closed = self._run_gripper_script(ycfg.exit_close_stroke, action_label="종료 시퀀스 닫기")
                if closed:
                    self.get_logger().info("그리퍼 닫기 완료")
                else:
                    self.get_logger().error("그리퍼 닫기 명령 실패/타임아웃 → 재시도")
                    retry = self._run_gripper_script(ycfg.exit_close_stroke, action_label="종료 시퀀스 닫기(재시도)")
                    if retry:
                        self.get_logger().info("재시도 성공")
                    else:
                        self.get_logger().error("재시도 실패 → 닫기 단계 건너뜀")

                # 6) 대기
                time.sleep(ycfg.exit_wait_after_close)

                # 7) 전달 후 목표 포즈로 이동 후 대기
                self.get_logger().info("posj(0,0,90,-90,90,0) 이동 후 대기")
                self.move_home_blocking(joints=list(ycfg.start_posj_deg))
                time.sleep(ycfg.rest_wait_after_pose)

                # 8) 물체 감지 위치(홈)로 이동
                self.get_logger().info("홈 위치로 이동 (동기 MoveJoint)")
                self.move_home_blocking(joints=list(self.config.robot.home_posj_deg))
                self.get_logger().info("최종 위치 도달 – 대기 상태로 유지")
            except Exception as e:
                self.get_logger().error(f"종료 시퀀스 실패: {e}")
            finally:
                self.exit_started = False
                self.exit_complete_event.set()

        threading.Thread(target=_seq, daemon=True).start()

    def ros_to_cv2(self, img_msg):
        """ROS Image 메시지를 OpenCV 이미지로 변환"""
        if img_msg.encoding == 'rgb8':
            channels = 3
        elif img_msg.encoding == 'bgr8':
            channels = 3
        elif img_msg.encoding == 'mono8':
            channels = 1
        else:
            self.get_logger().error(f'지원하지 않는 인코딩: {img_msg.encoding}')
            return None

        dtype = np.uint8
        img_array = np.frombuffer(img_msg.data, dtype=dtype)
        img_array = img_array.reshape((img_msg.height, img_msg.width, channels))

        if img_msg.encoding == 'rgb8':
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        return img_array

    def cv2_to_ros(self, cv_image, header, encoding='bgr8'):
        """OpenCV 이미지를 ROS Image 메시지로 변환"""
        msg = Image()
        msg.header = header
        msg.height = cv_image.shape[0]
        msg.width = cv_image.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = cv_image.shape[1] * cv_image.shape[2]
        msg.data = cv_image.tobytes()
        return msg

    def get_current_tcp_pose(self):
        """현재 TCP 위치 가져오기 (캐시 사용)"""
        return None

    def calculate_tcp_movement(self, pixel_error_x, pixel_error_y):
        """
        이미지 픽셀 오차를 TCP 좌표계 이동량으로 변환 (Eye-in-hand Visual Servoing)
        """
        dx = pixel_error_x * self.scale_x
        dy = pixel_error_y * self.scale_y
        dz = 0.0  # Z 방향은 고정 (카메라와 손 사이 거리 유지)

        return dx, dy, dz

    def move_robot_relative(self, dx, dy, dz):
        """
        로봇 TCP를 TOOL 좌표계 기준으로 상대 이동
        """
        if not self.tracking_enabled:
            return
        current_time = time.time()

        # 명령 전송 주기 제한 (0.2초에 한 번만) - 큐 누적 방지
        if current_time - self.last_move_time < 1.0:
            return

        if not self.move_lock.acquire(blocking=False):
            # 이미 이동 중이면 스킵
            return

        try:
            request = MoveLine.Request()
            # TOOL 좌표계 기준 상대 이동
            request.pos = [
                float(dx),
                float(dy),
                float(dz),
                0.0,  # Rx 변화 없음
                0.0,  # Ry 변화 없음
                0.0   # Rz 변화 없음
            ]
            request.vel = [self.vel_lin, self.vel_ang]
            request.acc = [self.acc_lin, self.acc_ang]
            request.time = 0.0
            request.radius = 0.0
            request.ref = 1      # TOOL 좌표계 (TCP 기준)
            request.mode = 1     # 상대 좌표 (REL)
            request.blend_type = 2
            request.sync_type = 1  # 비동기(amovel) 모드로 실행

            # 비동기 호출 (결과를 기다리지 않음)
            future = self.move_line_client.call_async(request)
            self.last_move_time = current_time

        finally:
            self.move_lock.release()

    def move_tool_relative_blocking(self, dx, dy, dz, timeout: float = 10.0) -> bool:
        """TOOL 기준 상대 이동을 동기 방식으로 전송 후 완료까지 대기."""
        request = MoveLine.Request()
        request.pos = [float(dx), float(dy), float(dz), 0.0, 0.0, 0.0]
        request.vel = [self.vel_lin, self.vel_ang]
        request.acc = [self.acc_lin, self.acc_ang]
        request.time = 0.0
        request.radius = 0.0
        request.ref = 1
        request.mode = 1
        request.blend_type = 2
        request.sync_type = 0  # 동기 모드

        future = self.move_line_client.call_async(request)
        return self._wait_future(future, action_label=f"MoveLine dx={dx},dy={dy},dz={dz}", timeout=timeout)

    def _run_gripper_script(self, stroke: int, *, action_label: str = "gripper", timeout: float = 10.0) -> bool:
        """DrlStart를 동기적으로 호출해 그리퍼를 제어한다."""
        if not self.drl_client.service_is_ready():
            self.get_logger().warn(f"DrlStart 서비스 준비 안됨: {action_label} 스킵")
            return False
        script = f"{DRL_BASE_CODE}\n\ngripper_move({stroke})"
        req = DrlStart.Request()
        req.robot_system = 0
        req.code = script
        future = self.drl_client.call_async(req)
        return self._wait_future(future, action_label=f"{action_label} (stroke={stroke})", timeout=timeout)

    def _wait_future(self, future, *, action_label: str, timeout: float) -> bool:
        """rclpy future를 timeout 안에 기다린다."""
        start = time.time()
        while time.time() - start < timeout:
            if future.done():
                try:
                    _ = future.result()
                    return True
                except Exception as e:
                    self.get_logger().error(f"{action_label} 실패: {e}")
                    return False
            time.sleep(0.05)
        self.get_logger().error(f"{action_label} 타임아웃({timeout}s)")
        return False

    def _control_timer_cb(self):
        """주기적으로 최신 명령만 적용하고 오래된 명령은 무시"""
        try:
            if self.exit_started or not self.tracking_enabled:
                return
            now = (self.get_clock().now().nanoseconds) / 1e9
            # 최신 명령이 없거나 오래되었으면 무시
            if self.latest_error_x is None or (now - self.latest_error_time) > self.stale_timeout:
                return

            # 데드존 체크
            mag = float(np.hypot(self.latest_error_x, self.latest_error_y))
            if mag <= self.deadzone_pixels:
                return

            # 이동량 계산
            dx, dy, dz = self.calculate_tcp_movement(self.latest_error_x, self.latest_error_y)

            # 상대 이동 전송
            self.move_robot_relative(dx, dy, dz)
        except Exception as e:
            self.get_logger().error(f'control timer error: {e}')

    def image_callback(self, msg):
        if self.exit_started or not self.tracking_enabled:
            return
        try:
            # ROS Image 메시지를 OpenCV 이미지로 변환
            cv_image = self.ros_to_cv2(msg)
            if cv_image is None:
                return
            self.last_color_frame = cv_image.copy()

            # 이미지 크기 업데이트
            self.image_height, self.image_width = cv_image.shape[:2]
            self.target_pixel_x = self.image_width / 2
            self.target_pixel_y = self.image_height / 2

            # YOLO 포즈 추정 수행
            results = self.model(cv_image, verbose=False, half=True)

            # 결과 이미지 생성
            annotated_image = results[0].plot()

            # 이미지 중심에 목표 위치 표시 (초록색 십자)
            center_x, center_y = int(self.target_pixel_x), int(self.target_pixel_y)
            cv2.drawMarker(annotated_image, (center_x, center_y), (0, 255, 0),
                          cv2.MARKER_CROSS, 30, 3)
            cv2.circle(annotated_image, (center_x, center_y), int(self.deadzone_pixels),
                      (0, 255, 0), 2)
            self.last_annotated_frame = annotated_image.copy()

            # 키포인트 추출
            det = results[0]
            boxes_obj = getattr(det, "boxes", None)
            keypoints_obj = det.keypoints

            if keypoints_obj is not None and len(keypoints_obj.xy) > 0 and boxes_obj is not None and len(boxes_obj) > 0:
                boxes_xyxy = boxes_obj.xyxy.cpu().numpy()
                kp_array = keypoints_obj.xy.cpu().numpy()
                target_idx = self._select_target_index(boxes_xyxy)

                if target_idx is None:
                    # 타겟을 유지할 수 없음 → 대기
                    if self._increment_target_miss():
                        return  # 종료 시퀀스 시작됨
                    status_text = "TARGET LOST"
                    status_color = (0, 0, 255)
                    cv2.putText(
                        annotated_image,
                        status_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        status_color,
                        2
                    )
                else:
                    keypoints = kp_array[target_idx]
                    selected_box = boxes_xyxy[target_idx].astype(int)

                    # 선택된 사람 bbox 시각화
                    cv2.rectangle(
                        annotated_image,
                        (selected_box[0], selected_box[1]),
                        (selected_box[2], selected_box[3]),
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        annotated_image,
                        "TARGET",
                        (selected_box[0], max(0, selected_box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                    # 오른손 키포인트 처리 (TCP 제어)
                    if len(keypoints) > self.RIGHT_WRIST_INDEX:
                        right_wrist = keypoints[self.RIGHT_WRIST_INDEX]
                        wrist_x, wrist_y = float(right_wrist[0]), float(right_wrist[1])

                        # 키포인트가 유효한지 확인 (0,0이 아닌 경우)
                        if wrist_x > 0 and wrist_y > 0:
                            # 오른손 끝 위치 표시 (빨간색 원)
                            cv2.circle(annotated_image, (int(wrist_x), int(wrist_y)), 10, (0, 0, 255), -1)

                            # 목표 위치와의 오차 계산
                            error_x = wrist_x - self.target_pixel_x
                            error_y = wrist_y - self.target_pixel_y
                            error_magnitude = np.sqrt(error_x**2 + error_y**2)

                            # 오차 벡터 표시
                            cv2.arrowedLine(annotated_image,
                                           (int(wrist_x), int(wrist_y)),
                                           (center_x, center_y),
                                           (255, 0, 0), 3)

                            # 정보 표시
                            cv2.putText(
                                annotated_image,
                                f'Wrist: ({int(wrist_x)}, {int(wrist_y)})',
                                (int(wrist_x) + 15, int(wrist_y) - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                2
                            )

                            cv2.putText(
                                annotated_image,
                                f'Error: ({error_x:.1f}, {error_y:.1f}) px | Mag: {error_magnitude:.1f} px',
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 0, 0),
                                2
                            )

                            # 오른손 끝 좌표 발행
                            hand_msg = Point()
                            hand_msg.x = wrist_x
                            hand_msg.y = wrist_y
                            hand_msg.z = error_magnitude
                            self.hand_publisher.publish(hand_msg)

                            # 최신 명령 버퍼에 저장 (stale drop은 타이머에서 처리)
                            self.latest_error_x = error_x
                            self.latest_error_y = error_y
                            self.latest_error_time = (self.get_clock().now().nanoseconds) / 1e9

                            # 데드존 외부면 움직임 예정 상태만 표시 (실제 전송은 타이머에서)
                            if error_magnitude > self.deadzone_pixels:
                                status_text = "MOVE PENDING"
                                status_color = (0, 165, 255)
                            else:
                                status_text = "IN DEADZONE"
                                status_color = (0, 255, 0)

                                # 데드존에 3초 이상 머무르면 종료 시퀀스
                                idle_time = time.time() - self.last_move_time
                                if idle_time >= self.config.yolo.idle_seconds_before_exit:
                                    self.get_logger().info("데드존에서 정지 → 전달 허가 절차 시작")
                                    self._trigger_handoff_sequence()

                            num_people = len(keypoints_obj.xy)
                            cv2.putText(
                                annotated_image,
                                f'People: {num_people} | Status: {status_text}',
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                status_color,
                                2
                            )
                    else:
                        # 키포인트가 부족하면 타겟을 잃은 것으로 간주
                        if self._increment_target_miss():
                            return
            else:
                # 키포인트가 없으면 대기 모드
                self._increment_target_miss()
                status_text = "WAITING - NO PERSON"
                status_color = (200, 200, 200)
                cv2.putText(
                    annotated_image,
                        status_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        status_color,
                        2
                    )

                # exit 시퀀스가 시작되었으면 더 이상 퍼블리시/처리하지 않음
                if self.exit_started:
                    return

            # FPS 계산 및 표시
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                current_time = self.get_clock().now()
                elapsed_time = (current_time - self.start_time).nanoseconds / 1e9
                fps = self.frame_count / elapsed_time
                self.get_logger().info(f'FPS: {fps:.2f}')

            # OpenCV 이미지를 ROS Image 메시지로 변환
            output_msg = self.cv2_to_ros(annotated_image, msg.header)

            # 결과 발행
            self.publisher.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f'오류 발생: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _reset_tracking_target(self):
        self.active_target_bbox = None
        self.target_miss_count = 0

    @staticmethod
    def _bbox_iou(box_a, box_b):
        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])
        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _select_target_index(self, boxes_xyxy):
        if len(boxes_xyxy) == 0:
            return None
        if self.active_target_bbox is None:
            areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
            idx = int(np.argmax(areas))
            self.active_target_bbox = boxes_xyxy[idx].copy()
            self.target_miss_count = 0
            return idx

        ious = np.array([self._bbox_iou(self.active_target_bbox, box) for box in boxes_xyxy])
        best_idx = int(np.argmax(ious))
        if ious[best_idx] >= self.config.yolo.target_iou_thresh:
            self.active_target_bbox = boxes_xyxy[best_idx].copy()
            self.target_miss_count = 0
            return best_idx
        return None

    def _increment_target_miss(self):
        """현재 타겟을 놓쳤을 때 카운트. True를 반환하면 종료 시퀀스를 시작한 것."""
        if self.active_target_bbox is None:
            return False
        self.target_miss_count += 1
        if self.target_miss_count >= self.config.yolo.target_miss_frames:
            self._handle_target_lost()
            return True
        return False

    def _handle_target_lost(self):
        self.get_logger().warn("타겟 이탈 감지 → 종료 시퀀스 진행")
        self.active_target_bbox = None
        self.target_miss_count = 0
        self.awaiting_consent = False
        if not self.exit_started:
            self.start_exit_sequence()

    def _trigger_handoff_sequence(self):
        if self.awaiting_consent or self.exit_started:
            return
        self.awaiting_consent = True
        self.tracking_enabled = False
        self.last_move_time = time.time()

        def _workflow():
            try:
                frame = self.last_annotated_frame if self.last_annotated_frame is not None else self.last_color_frame
                if frame is None:
                    self.get_logger().warn("스냅샷 프레임이 없어 바로 전달 시퀀스를 진행합니다.")
                    self.start_exit_sequence()
                    return

                consent = self._request_handoff_consent(frame)
                if consent:
                    self.get_logger().info("연산자 동의 확인 → 전달 시퀀스 시작")
                    self.start_exit_sequence()
                else:
                    self.get_logger().info("연산자가 거절하여 대기 상태로 유지합니다.")
                    self.tracking_enabled = True
            finally:
                self.awaiting_consent = False
                self.last_move_time = time.time()

        threading.Thread(target=_workflow, daemon=True).start()

    def _request_handoff_consent(self, frame_bgr: np.ndarray) -> bool:
        if self.realtime_client is None:
            self.get_logger().warn("Realtime 클라이언트 없음 → 자동으로 전달 시퀀스를 진행합니다.")
            return True
        try:
            return self.realtime_client.request_handoff_consent(frame_bgr)
        except Exception as exc:
            self.get_logger().error(f"Realtime consent 요청 실패: {exc} → 기본 승인")
            return True


def main(args=None):
    rclpy.init(args=args)
    node = YoloHandRobotControl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.should_exit = True
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
