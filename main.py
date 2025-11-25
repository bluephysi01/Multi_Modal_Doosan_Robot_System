#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 멀티모달(물체 인식/음성) + YOLO 포즈 트래킹 메인
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
import threading

import DR_init

from robot_controller import MultiModalRobotController, ROBOT_ID
from object_mapper import ObjectMapper
from realtime_client import RealtimeImageVoiceClient, test_microphone
from vision_module import Tracker, project_point, get_3d_bbox
from yolo_hand_robot_control import YoloHandRobotControl
from common_utils import safe_imshow, wait_for_camera_stream
from config import APP_CONFIG


def run_yolo_pose_phase(yolo_node: Optional[YoloHandRobotControl]) -> None:
    """타겟 위치 이동 후 YOLO 포즈 트래킹 단계."""
    if yolo_node is None:
        return

    print("\n🎯 타겟 도달 → YOLO 포즈 트래킹 활성화")
    yolo_node.enable_tracking()
    completed = yolo_node.wait_for_exit_sequence(timeout=APP_CONFIG.yolo.overall_timeout)
    if completed:
        print("✅ 포즈 기반 데드존 처리 완료 (그리퍼 개방 및 홈 복귀 확인)")
    else:
        print("⚠️  포즈 트래킹 타임아웃 – 다음 사이클로 넘어갑니다.")
        yolo_node.disable_tracking()


def run_multimodal_loop(controller, mapper, realtime_client, yolo_node, args):
    """멀티모달 자동 루프 실행 (YOLO 포즈 트래킹 포함)"""
    if realtime_client is None:
        print("❌ Realtime 클라이언트가 없습니다. --disable-multimodal 옵션을 해제해주세요.")
        return
    main_window = "Real-time Object Tracking"
    snapshot_window = "Snapshot - Annotated"
    ai_window = "Snapshot - Original (AI)"

    # 윈도우 생성 상태 추적
    windows_created = False

    try:
        # 모든 윈도우를 미리 생성 (한 번만)
        cv2.namedWindow(main_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(main_window, 1280, 720)
        cv2.moveWindow(main_window, 0, 0)

        cv2.namedWindow(snapshot_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(snapshot_window, 640, 480)
        cv2.moveWindow(snapshot_window, 0, 750)

        cv2.namedWindow(ai_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ai_window, 640, 480)
        cv2.moveWindow(ai_window, 650, 750)

        windows_created = True
        print("✅ GUI 윈도우 생성 완료")
    except Exception as e:
        print(f"⚠️  윈도우 생성 오류: {e}")
        return

    print("\n📸 실시간 물체 추적 시작. Ctrl+C 또는 ESC로 종료합니다.")

    # Tracker 초기화
    tracker = Tracker(max_age=APP_CONFIG.vision.tracker_max_age, dist_thresh=APP_CONFIG.vision.tracker_dist_thresh)
    tracked_objects = {}
    last_detection_time = 0.0
    detection_interval = APP_CONFIG.vision.detection_interval

    # 상태 변수
    waiting_for_snapshot = True
    cycle_in_progress = False
    gui_update_active = False

    def reset_tracking_state():
        nonlocal tracker, tracked_objects
        tracker = Tracker(max_age=APP_CONFIG.vision.tracker_max_age,
                          dist_thresh=APP_CONFIG.vision.tracker_dist_thresh)
        tracked_objects = {}
        controller.detected_objects = []
        if hasattr(controller, '_tracking_logged'):
            delattr(controller, '_tracking_logged')
        if hasattr(controller, '_bbox_error_logged'):
            delattr(controller, '_bbox_error_logged')

    # 단일 스레드 GUI: 메인 루프에서만 cv2 이벤트를 처리한다.
    def pump_gui():
        """윈도우 이벤트 최소 처리"""
        cv2.waitKey(1)

    # 더미 이미지로 윈도우 초기화
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_img, "Waiting for snapshot...", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    safe_imshow(snapshot_window, dummy_img)
    safe_imshow(ai_window, dummy_img)

    # 메인 윈도우에도 초기 이미지 표시
    if controller.latest_cv_color is not None:
        safe_imshow(main_window, controller.latest_cv_color)

    # 윈도우가 완전히 생성되도록 대기
    for _ in range(10):
        cv2.waitKey(10)

    while rclpy.ok() and not controller.should_exit:
        # 실시간 물체 감지 및 추적
        current_time = time.time()
        if (not controller.is_robot_moving and
            controller.latest_cv_color is not None and
            controller.latest_cv_depth_mm is not None and
            controller.intrinsics is not None and
            (current_time - last_detection_time) >= detection_interval):

            try:
                detections = controller.detect_objects()
                if len(detections) > 0:
                    tracked_objects = tracker.update(detections)
                    controller.detected_objects = list(tracked_objects.values())
                    # 디버깅: 추적 중인 객체 확인
                    if len(tracked_objects) > 0 and not hasattr(controller, '_tracking_logged'):
                        print(f"✅ 추적 중: {len(tracked_objects)}개 객체")
                        controller._tracking_logged = True
                else:
                    # 감지된 물체가 없으면 이전 트래킹 정보도 즉시 초기화하여 잔상 제거
                    tracked_objects = {}
                    tracker.objects.clear()
                    controller.detected_objects = []
                    if hasattr(controller, '_tracking_logged'):
                        delattr(controller, '_tracking_logged')
                last_detection_time = current_time
            except Exception as e:
                # 감지 에러는 무시하고 계속 진행
                if "Connection reset" in str(e):
                    print(f"⚠️  카메라 연결 오류 (무시): {e}")
                last_detection_time = current_time

        # 실시간 트래킹 화면 표시 (항상 업데이트)
        if controller.latest_cv_color is not None:
            display_frame = controller.latest_cv_color.copy()
            h, w, _ = display_frame.shape

            # 추적 중인 물체에 6D 박스 2D 프로젝션(ID 라벨 1번부터)만 표시
            if tracked_objects and controller.intrinsics is not None:
                fx, fy = controller.intrinsics.fx, controller.intrinsics.fy
                cx, cy = controller.intrinsics.ppx, controller.intrinsics.ppy

                for oid, obj in tracked_objects.items():
                    display_id = oid + 1
                    Xo, Yo, Zo = obj["x"], obj["y"], obj["z"]
                    lx = obj.get("lx", obj.get("length", 0.05))
                    ly = obj.get("ly", obj.get("length", 0.05))

                    try:
                        box = get_3d_bbox(Xo, Yo, Zo, obj["yaw"], lx, ly)
                        pts = []
                        for p in box:
                            pr = project_point(p, fx, fy, cx, cy)
                            if pr is not None:
                                pts.append(pr)

                        if len(pts) == 4:
                            # 3D 박스 프로젝션 그리기
                            for i in range(4):
                                cv2.line(display_frame, pts[i], pts[(i+1)%4], (0, 255, 0), 2)

                            # ID 라벨
                            label_pos = (pts[0][0], max(10, pts[0][1] - 10))
                            cv2.putText(
                                display_frame, f"ID-{display_id:02d}", label_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                            )

                            # 중심점 표시
                            if Zo > 0:
                                center_u = int(Xo * fx / Zo + cx)
                                center_v = int(Yo * fy / Zo + cy)
                                if 0 <= center_u < w and 0 <= center_v < h:
                                    cv2.circle(display_frame, (center_u, center_v), 4, (0, 255, 255), -1)
                    except Exception as e:
                        if not hasattr(controller, '_bbox_error_logged'):
                            print(f"⚠️  바운딩 박스 그리기 오류: {e}")
                            controller._bbox_error_logged = True

            # 화면 중심점 표시
            cv2.circle(display_frame, (w // 2, h // 2), 5, (0, 0, 255), -1)
            cv2.line(display_frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 0, 255), 2)
            cv2.line(display_frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 0, 255), 2)

            # 상태 표시
            if controller.is_robot_moving:
                status_text = "ROBOT MOVING..."
                color = (0, 0, 255)
            elif cycle_in_progress:
                status_text = "AI PROCESSING..."
                color = (255, 165, 0)
            elif waiting_for_snapshot:
                status_text = "TRACKING - Ready for snapshot"
                color = (255, 255, 0)
            else:
                status_text = "Ready"
                color = (0, 255, 0)

            cv2.rectangle(display_frame, (5, 5), (500, 95), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (5, 5), (500, 95), color, 2)
            cv2.putText(display_frame, status_text, (15, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # 추적 중인 객체 수
            if tracked_objects:
                obj_text = f"Tracking: {len(tracked_objects)} objects"
                cv2.putText(display_frame, obj_text, (15, 75),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No objects detected", (15, 75),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            safe_imshow(main_window, display_frame)

        # GUI 이벤트 처리 (더 짧은 주기로)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("\n✓ ESC 키로 종료합니다...")
            controller.should_exit = True
            break

        # 스냅샷 촬영 대기 중일 때
        if waiting_for_snapshot and not cycle_in_progress and not controller.is_robot_moving:
            if len(tracked_objects) == 0:
                time.sleep(0.1)
                continue

            print("\n🔄 AI에게 스냅샷 촬영 허가 요청 중...")
            cycle_in_progress = True
            waiting_for_snapshot = False

            try:
                pump_gui()
                # AI가 "스냅샷을 찍을까요?" 물어봄
                should_take_snapshot = realtime_client.ask_snapshot_permission()

                if not should_take_snapshot:
                    print("⚠️  스냅샷 촬영 취소됨")
                    waiting_for_snapshot = True
                    cycle_in_progress = False
                    time.sleep(APP_CONFIG.vision.snapshot_wait)
                    continue

                print("\n📸 스냅샷 촬영 중...")

                # 스냅샷 직전에 다시 감지해 최신 위치만 반영
                try:
                    detections_now = controller.detect_objects()
                    if len(detections_now) == 0:
                        print("⚠️  스냅샷 시점에 감지된 물체가 없습니다.")
                        waiting_for_snapshot = True
                        cycle_in_progress = False
                        time.sleep(APP_CONFIG.vision.snapshot_wait)
                        continue
                    tracked_objects = tracker.update(detections_now)
                    controller.detected_objects = list(tracked_objects.values())
                except Exception as detect_exc:
                    print(f"⚠️  스냅샷 직전 재감지 실패: {detect_exc}")
                    waiting_for_snapshot = True
                    cycle_in_progress = False
                    time.sleep(APP_CONFIG.vision.snapshot_wait)
                    continue

                # 현재 추적 중인 물체를 스냅샷으로 변환
                color_frame = controller.latest_cv_color.copy()
                detections = []
                fx, fy = controller.intrinsics.fx, controller.intrinsics.fy
                cx, cy = controller.intrinsics.ppx, controller.intrinsics.ppy

                for oid, obj in tracked_objects.items():
                    obj_copy = obj.copy()
                    dist = np.sqrt(obj_copy["x"]**2 + obj_copy["y"]**2)
                    obj_copy["distance_from_center"] = dist

                    # 픽셀 좌표 계산
                    if "pixel_u" not in obj_copy and obj_copy["z"] > 0:
                        obj_copy["pixel_u"] = int(obj_copy["x"] * fx / obj_copy["z"] + cx)
                        obj_copy["pixel_v"] = int(obj_copy["y"] * fy / obj_copy["z"] + cy)

                    # bbox
                    if "bbox_min_u" not in obj_copy and "pixel_u" in obj_copy:
                        obj_copy["bbox_min_u"] = obj_copy["pixel_u"] - 50
                        obj_copy["bbox_max_u"] = obj_copy["pixel_u"] + 50
                        obj_copy["bbox_min_v"] = obj_copy["pixel_v"] - 50
                        obj_copy["bbox_max_v"] = obj_copy["pixel_v"] + 50

                    # length 필드 통일
                    if "length" not in obj_copy and "ly" in obj_copy:
                        obj_copy["length"] = obj_copy["ly"]

                    detections.append(obj_copy)

                detections = sorted(detections, key=lambda x: x["distance_from_center"])

                # 감지된 객체 정보 출력
                print("\n" + "=" * 80)
                print(f"🔍 스냅샷에 포함된 물체: {len(detections)}개")
                print("=" * 80)
                for idx, obj in enumerate(detections, start=1):
                    print(f"  📦 OBJ-{idx:02d}:")
                    print(f"     위치: ({obj['x']:.3f}, {obj['y']:.3f}, {obj['z']:.3f})m")
                    print(f"     회전: {np.degrees(obj['yaw']):.1f}°")
                    print(f"     크기: {obj.get('length', obj.get('ly', 0))*1000:.0f}mm")
                print("=" * 80 + "\n")

                # 스냅샷 생성
                snapshot = mapper.build_snapshot(color_frame, detections)
                if snapshot is None:
                    print("⚠️  스냅샷 생성 실패")
                    waiting_for_snapshot = True
                    cycle_in_progress = False
                    continue

                # 스냅샷 저장 및 표시 (이미 생성된 윈도우에 업데이트)
                safe_imshow(snapshot_window, snapshot.annotated_frame)
                safe_imshow(ai_window, snapshot.original_frame)

                # 스냅샷 저장
                snapshot_dir = APP_CONFIG.paths.snapshot_dir
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                timestamp = int(time.time())

                original_path = snapshot_dir / f"snapshot_original_{timestamp}.jpg"
                cv2.imwrite(str(original_path), snapshot.original_frame)

                annotated_path = snapshot_dir / f"snapshot_annotated_{timestamp}.jpg"
                cv2.imwrite(str(annotated_path), snapshot.annotated_frame)

                print(f"💾 스냅샷 저장: {original_path.name}, {annotated_path.name}")

                # AI에게 물체 분석 및 선택 요청
                try:
                    pump_gui()
                    decision = realtime_client.request_pick_label(
                        mapper=mapper,
                        snapshot=snapshot,
                        max_turns=args.realtime_turns,
                    )
                except Exception as exc:
                    print(f"❌ Realtime API 통신 실패: {exc}")
                    waiting_for_snapshot = True
                    cycle_in_progress = False
                    continue

                if not decision.label:
                    print("⚠️  유효한 PICK_LABEL을 받지 못했습니다.")
                    waiting_for_snapshot = True
                    cycle_in_progress = False
                    continue

                target = snapshot.find_object(decision.label)
                if target is None:
                    print(f"⚠️  {decision.label}에 해당하는 물체를 찾지 못했습니다.")
                    waiting_for_snapshot = True
                    cycle_in_progress = False
                    continue

                print(f"\n✅ 선택된 대상: {target.label}")

                # 물체 집기
                success = controller.pick_object(target.raw)
                if success:
                    print(f"\n✅ {ObjectMapper.format_decision_message(target)}")

                    delivery_choice = "person"
                    picked_description = decision.transcript.strip() if decision and decision.transcript else target.label
                    if realtime_client is not None:
                        try:
                            delivery_choice = realtime_client.request_delivery_destination(picked_description)
                            print(f"🗣️ 전달 대상 선택: {delivery_choice}")
                        except Exception as exc:
                            print(f"⚠️  전달 대상 질의 실패: {exc}. 기본값(person) 사용")

                    if delivery_choice == "basket":
                        controller.place_object_in_basket()
                    else:
                        controller.place_object_at_target(open_gripper=False, return_home=False)
                        if not args.disable_yolo_pose:
                            run_yolo_pose_phase(yolo_node)

                    print("✅ 사이클 완료! 다음 사이클 대기 중...\n")
                else:
                    print("⚠️  물체 집기에 실패했습니다.")

                # 다음 사이클 준비
                waiting_for_snapshot = True
                cycle_in_progress = False
                reset_tracking_state()

                # 짧은 대기 후 바로 다음 사이클로
                print("⏳ 다음 사이클 준비 중...\n")
                time.sleep(1.0)

            except KeyboardInterrupt:
                print("\n⚠️  사이클 중 Ctrl+C 감지")
                controller.should_exit = True
                break
            except Exception as e:
                print(f"❌ 사이클 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                waiting_for_snapshot = True
                cycle_in_progress = False
                reset_tracking_state()

                time.sleep(1.0)

    # 윈도우 명시적으로 닫기
    print("GUI 윈도우 정리 중...")
    try:
        # GUI 스레드가 있다면 먼저 정지
        if 'gui_update_active' in locals():
            gui_update_active = False
        if 'gui_thread' in locals() and gui_thread.is_alive():
            gui_thread.join(timeout=2.0)

        # 윈도우 닫기
        if windows_created:
            cv2.destroyWindow(main_window)
            cv2.destroyWindow(snapshot_window)
            cv2.destroyWindow(ai_window)
            for _ in range(5):
                cv2.waitKey(1)
    except Exception as e:
        print(f"⚠️  윈도우 정리 중 오류 (무시됨): {e}")
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Robot Control System (with YOLO pose)")

    parser.add_argument("--disable-multimodal", action="store_true", help="Realtime 연동 비활성화")
    parser.add_argument("--text-input-only", action="store_true", help="마이크 대신 텍스트 입력")
    parser.add_argument("--play-audio-responses", action="store_true", help="AI 음성 응답 재생")
    parser.add_argument("--realtime-model", type=str, default=APP_CONFIG.realtime.model, help="Realtime API 모델")
    parser.add_argument("--realtime-turns", type=int, default=APP_CONFIG.realtime.max_turns, help="한 명령 당 허용 턴 수")
    parser.add_argument("--operator-audio-seconds", type=int, default=APP_CONFIG.realtime.operator_audio_seconds, help="명령 녹음 시간(초)")
    parser.add_argument("--realtime-debug-events", action="store_true", help="Realtime 이벤트 로그 출력")
    parser.add_argument("--skip-mic-test", action="store_true", help="마이크 사전 테스트 건너뛰기")
    parser.add_argument("--disable-yolo-pose", dest="disable_yolo_pose", action="store_true", help="YOLO 포즈 트래킹 비활성화")

    args = parser.parse_args()

    # 마이크 테스트
    if (not args.text_input_only and
        not args.disable_multimodal and
        not args.skip_mic_test):
        if not test_microphone():
            print("\n❌ 마이크가 정상적으로 작동하지 않습니다.")
            print("   --skip-mic-test 또는 --text-input-only 옵션으로 건너뛸 수 있습니다.")
            return

    # ROS2 초기화
    rclpy.init(args=None)

    # DSR 노드 생성
    dsr_node = rclpy.create_node("dsr_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    # DSR_ROBOT2 라이브러리 import
    try:
        from DSR_ROBOT2 import wait, get_current_posj, movej, movel
        from DR_common2 import posj, posx
    except ImportError as e:
        print(f"❌ DSR_ROBOT2 라이브러리 임포트 실패: {e}")
        dsr_node.destroy_node()
        rclpy.shutdown()
        return

    # 컨트롤러 생성
    controller = MultiModalRobotController(APP_CONFIG)

    # ObjectMapper 생성
    mapper = ObjectMapper()

    # Realtime 클라이언트 생성
    realtime_client = None
    if not args.disable_multimodal:
        try:
            realtime_client = RealtimeImageVoiceClient(
                model=args.realtime_model,
                operator_audio_seconds=args.operator_audio_seconds,
                use_microphone=not args.text_input_only,
                play_audio=args.play_audio_responses,
                debug_events=args.realtime_debug_events,
            )
            print("✅ Realtime multimodal 클라이언트 준비 완료")
        except Exception as exc:
            print(f"❌ Realtime 클라이언트 초기화 실패: {exc}")
            args.disable_multimodal = True

    # YOLO 포즈 노드 생성 (필요 시)
    yolo_node = None
    if not args.disable_yolo_pose:
        try:
            yolo_node = YoloHandRobotControl(APP_CONFIG, realtime_client=realtime_client)
            print("✅ YOLO 포즈 트래커 준비 완료")
        except Exception as exc:
            print(f"⚠️  YOLO 포즈 트래커 초기화 실패: {exc}")
            args.disable_yolo_pose = True
            yolo_node = None

    # 카메라 데이터 대기
    # 멀티스레드 executor로 카메라 콜백을 백그라운드에서 계속 돌림
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(controller)
    executor.add_node(dsr_node)
    if yolo_node:
        executor.add_node(yolo_node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    if not wait_for_camera_stream(controller):
        controller.terminate_gripper()
        controller.destroy_node()
        if yolo_node:
            yolo_node.destroy_node()
        dsr_node.destroy_node()
        executor.shutdown()
        spin_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        rclpy.shutdown()
        return

    try:
        run_multimodal_loop(controller, mapper, realtime_client, yolo_node, args)
    except KeyboardInterrupt:
        print("\n\nCtrl+C로 종료...")
    finally:
        print("\n프로그램 종료 중...")
        controller.terminate_gripper()
        cv2.destroyAllWindows()
        controller.destroy_node()
        if yolo_node:
            yolo_node.destroy_node()
        dsr_node.destroy_node()
        executor.shutdown()
        spin_thread.join(timeout=1.0)
        if rclpy.ok():
            rclpy.shutdown()
        print("✓ 종료 완료")


if __name__ == "__main__":
    main()
