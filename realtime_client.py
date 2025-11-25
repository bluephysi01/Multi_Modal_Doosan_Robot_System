"""Realtime multimodal client that reuses the GPT realtime API workflow."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import sounddevice as sd
from websockets import connect

from object_mapper import ObjectMapper, SnapshotPayload
from system_prompt import (
    MULTI_MODAL_SYSTEM_PROMPT,
    SNAPSHOT_REQUEST_PROMPT,
    HANDOFF_PROMPT,
    DELIVERY_PROMPT,
)
from config import APP_CONFIG

sd.default.device = "default"


def test_microphone(duration: int = 2, fs: int = 24000) -> bool:
    """마이크 테스트"""

    print("\n🎤 마이크 테스트 중...")
    try:
        devices = sd.query_devices()
        print("\n📋 사용 가능한 오디오 장치:")
        for idx, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:
                print(f"  [{idx}] {device['name']} (입력 채널: {device['max_input_channels']})")

        default_input = sd.query_devices(kind="input")
        print(f"\n✅ 현재 기본 입력 장치: {default_input['name']}")

        print(f"🎤 {duration}초 테스트 녹음 시작...")
        recording = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype="int16")
        sd.wait()

        audio_max = int(np.max(np.abs(recording)))
        audio_mean = float(np.mean(np.abs(recording)))
        print("\n📊 녹음 데이터 분석:")
        print(f"   - 최대 진폭: {audio_max}")
        print(f"   - 평균 진폭: {audio_mean:.2f}")

        if audio_max < 100:
            print("⚠️  경고: 마이크 입력이 너무 작습니다! 마이크 설정을 확인하세요.")
            return False

        print("✅ 마이크가 정상적으로 작동합니다!\n")
        return True
    except Exception as exc:
        print(f"❌ 마이크 테스트 실패: {exc}")
        return False


@dataclass
class PickDecision:
    label: Optional[str]
    transcript: str
    raw_response: str


def _load_env_file() -> None:
    """Load environment variables from nearby .env files if available."""
    candidates = [
        Path(__file__).resolve().parent / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


class RealtimeImageVoiceClient:
    """Handles the realtime websocket session for voice + image instructions."""

    def __init__(
        self,
        model: Optional[str] = None,
        instructions: str = MULTI_MODAL_SYSTEM_PROMPT,
        operator_audio_seconds: Optional[int] = None,
        use_microphone: bool = True,
        play_audio: Optional[bool] = None,
        debug_events: bool = False,
    ) -> None:
        cfg = APP_CONFIG.realtime
        _load_env_file()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Create a .env or export the variable.")

        model_name = model or cfg.model
        self.url = f"wss://api.openai.com/v1/realtime?model={model_name}"
        self.instructions = instructions.strip()
        self.operator_audio_seconds = operator_audio_seconds or cfg.operator_audio_seconds
        self.use_microphone = use_microphone
        self.play_audio = cfg.play_audio if play_audio is None else play_audio
        self.debug_events = debug_events
        self.modalities: List[str] = ["text"]
        if self.play_audio:
            self.modalities.append("audio")

        self._headers = [
            ("Authorization", f"Bearer {self.api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]

    def ask_snapshot_permission(self) -> bool:
        """스냅샷 촬영 허가 요청"""
        return asyncio.run(self._ask_snapshot_permission())

    def request_pick_label(
        self,
        mapper: ObjectMapper,
        snapshot: SnapshotPayload,
        max_turns: Optional[int] = None,
    ) -> PickDecision:
        """Blocking helper that spins an asyncio loop until a label is extracted."""

        return asyncio.run(self._request_pick_label(mapper, snapshot, max_turns or APP_CONFIG.realtime.max_turns))

    def request_handoff_consent(self, frame_bgr: np.ndarray) -> bool:
        """Compliment the operator and ask for permission to hand over the object."""
        return asyncio.run(self._request_handoff_consent(frame_bgr))

    def request_delivery_destination(self, item_description: str) -> str:
        """Ask operator where to deliver the object. Returns 'person' or 'basket'."""
        return asyncio.run(self._request_delivery_destination(item_description))

    async def _ask_snapshot_permission(self) -> bool:
        """스냅샷 촬영 허가를 AI에게 물어봄"""
        print("[Realtime] Connecting to OpenAI realtime API...")

        # 임시로 SNAPSHOT_REQUEST_PROMPT를 사용
        temp_instructions = self.instructions
        self.instructions = SNAPSHOT_REQUEST_PROMPT

        try:
            async with connect(self.url, additional_headers=self._headers) as ws:
                print("[Realtime] WebSocket connected.")
                await self._clear_conversation(ws)
                await self._send_session_update(ws)

                # AI가 먼저 "스냅샷을 찍을까요?" 물어봄
                await self._request_response(ws)
                print("\n🤖 AI: ", end="", flush=True)
                await self._drain_response(ws, stream_print=True)

                # 사용자 응답 받기
                if not await self._send_operator_turn(ws, 0):
                    return False

                await self._request_response(ws)
                print("\n🤖 AI: ", end="", flush=True)
                transcript = await self._drain_response(ws, stream_print=True)

                # TAKE_SNAPSHOT: YES 확인
                if "TAKE_SNAPSHOT:" in transcript and "YES" in transcript:
                    return True
                return False
        finally:
            self.instructions = temp_instructions

    async def _request_pick_label(
        self,
        mapper: ObjectMapper,
        snapshot: SnapshotPayload,
        max_turns: int,
    ) -> PickDecision:
        print("[Realtime] Connecting to OpenAI realtime API...")
        async with connect(self.url, additional_headers=self._headers) as ws:
            print("[Realtime] WebSocket connected.")
            await self._clear_conversation(ws)
            await self._send_session_update(ws)

            # 이미지를 user 메시지로 전송
            await self._send_image_message(ws, snapshot)

            # AI 응답 요청
            await self._request_response(ws)
            print("\n🤖 AI: ", end="", flush=True)
            initial_ack = await self._drain_response(ws, stream_print=True)

            transcript_cache = ""
            for turn in range(max_turns):
                if not await self._send_operator_turn(ws, turn):
                    continue
                await self._request_response(ws)
                print("\n🤖 AI: ", end="", flush=True)
                transcript_cache = await self._drain_response(ws, stream_print=True)
                labeled_obj = mapper.resolve_label(transcript_cache, snapshot)
                if labeled_obj:
                    return PickDecision(label=labeled_obj.label, transcript=transcript_cache, raw_response=transcript_cache)

            return PickDecision(label=None, transcript=transcript_cache, raw_response=transcript_cache)

    async def _request_handoff_consent(self, frame_bgr: np.ndarray) -> bool:
        temp_instructions = self.instructions
        self.instructions = HANDOFF_PROMPT
        try:
            async with connect(self.url, additional_headers=self._headers) as ws:
                print("[Realtime] WebSocket connected for handoff consent.")
                await self._clear_conversation(ws)
                await self._send_session_update(ws)

                await self._send_person_image_message(ws, frame_bgr)

                # AI compliments + asks question
                await self._request_response(ws)
                print("\n🤖 AI: ", end="", flush=True)
                await self._drain_response(ws, stream_print=True)

                # Operator response (voice/text)
                if not await self._send_operator_turn(ws, 0):
                    print("[Realtime] Operator response missing; default to NO")
                    return False

                await self._request_response(ws)
                print("\n🤖 AI: ", end="", flush=True)
                transcript = await self._drain_response(ws, stream_print=True)
                return "CONSENT: YES" in transcript.upper()
        finally:
            self.instructions = temp_instructions

    async def _request_delivery_destination(self, item_description: str) -> str:
        temp_instructions = self.instructions
        self.instructions = DELIVERY_PROMPT
        try:
            async with connect(self.url, additional_headers=self._headers) as ws:
                print("[Realtime] WebSocket connected for delivery selection.")
                await self._clear_conversation(ws)
                await self._send_session_update(ws)

                await self._send_delivery_prompt(ws, item_description)

                # AI asks the question
                await self._request_response(ws)
                print("\n🤖 AI: ", end="", flush=True)
                await self._drain_response(ws, stream_print=True)

                # Operator response
                if not await self._send_operator_turn(ws, 0):
                    print("[Realtime] Operator response missing; default to PERSON")
                    return "person"

                await self._request_response(ws)
                print("\n🤖 AI: ", end="", flush=True)
                transcript = await self._drain_response(ws, stream_print=True)
                upper = transcript.upper()
                if "DESTINATION: BASKET" in upper:
                    return "basket"
                if "DESTINATION: PERSON" in upper:
                    return "person"
                return "person"
        finally:
            self.instructions = temp_instructions

    async def _clear_conversation(self, ws) -> None:
        await ws.send(json.dumps({"type": "conversation.clear"}))
        print("[Realtime] Conversation cleared.")

    async def _send_session_update(self, ws) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "turn_detection": None,
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "modalities": self.modalities,
                        "instructions": self.instructions,
                    },
                }
            )
        )

    async def _send_image_message(self, ws, snapshot: SnapshotPayload) -> None:
        """이미지를 user 메시지로 전송"""
        data_url, mime = snapshot.to_base64(mime="image/jpeg")
        image_size_kb = len(data_url) / 1024
        print(f"[Realtime] 이미지 전송: {snapshot.original_frame.shape}, {image_size_kb:.1f}KB")

        prompt_text = (
            f"이미지를 보면 물체 위에 번호(1, 2, 3 등)가 표시되어 있습니다.\n"
            f"총 {len(snapshot.labeled_objects)}개의 물체가 감지되었습니다.\n\n"
            f"각 번호에 해당하는 물체가 무엇인지 간단히 말해주세요:\n"
            f"형식: 'OBJ-01은 [물체이름]입니다. OBJ-02는 [물체이름]입니다.'\n"
            f"마지막에 '어떤 물체를 집을까요?'라고 물으세요."
        )

        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt_text},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    },
                }
            )
        )
        print(f"[Realtime] 프롬프트: {prompt_text[:80]}...")

    async def _send_person_image_message(self, ws, frame_bgr: np.ndarray) -> None:
        data_url = _frame_to_data_url(frame_bgr)
        prompt_text = (
            "이미지를 보고 상대방의 옷차림을 한 문장으로 칭찬한 뒤, "
            "'물건을 드릴까요?'라고 꼭 물어보세요."
        )
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt_text},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    },
                }
            )
        )
        print("[Realtime] 전송된 이미지로 의상 칭찬 및 전달 여부 질문")

    async def _send_delivery_prompt(self, ws, item_description: str) -> None:
        prompt_text = (
            f"방금 선택된 물체 설명: {item_description}\n"
            f"위 물체를 어디로 가져다 드릴지 물어보세요. "
            f"1번은 사람(직접 전달), 2번은 바구니로 전달입니다."
        )
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt_text},
                        ],
                    },
                }
            )
        )
        print("[Realtime] 전달 위치 질문 전송")

    async def _request_response(self, ws) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {"modalities": self.modalities},
                }
            )
        )

    async def _drain_response(self, ws, *, stream_print: bool = False) -> str:
        transcript_parts: List[str] = []
        audio_chunks: List[str] = []
        printed = False
        while True:
            data = json.loads(await ws.recv())
            event_type = data.get("type")
            if self.debug_events:
                print(f"[Realtime:event] {event_type}")

            if event_type == "response.audio_transcript.delta":
                delta = data.get("delta", "")
                transcript_parts.append(delta)
                if stream_print and delta:
                    print(delta, end="", flush=True)
                    printed = True
            elif event_type in ("response.output_text.delta", "response.text.delta"):
                delta = data.get("delta", "")
                transcript_parts.append(delta)
                if stream_print and delta:
                    print(delta, end="", flush=True)
                    printed = True
            elif event_type == "response.audio.delta":
                audio_chunks.append(data.get("delta", ""))
            elif event_type == "response.audio.done" and self.play_audio and audio_chunks:
                await self._play_audio(audio_chunks)
                audio_chunks = []
            elif event_type == "response.done":
                if stream_print and printed:
                    print()
                return "".join(transcript_parts)
            elif event_type == "response.error":
                raise RuntimeError(data.get("error", {}).get("message", "Realtime response error"))

    async def _send_operator_turn(self, ws, turn: int) -> bool:
        if self.use_microphone:
            audio_b64 = await self._record_audio(self.operator_audio_seconds)
            if not audio_b64:
                return False
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            return True

        text = input(f"Operator command #{turn+1}: ").strip()
        if not text:
            return False
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
        )
        return True

    async def _record_audio(self, duration: int, fs: int = 24000) -> Optional[str]:
        try:
            print(f"\n🎙️  Recording command for {duration}s... (Ctrl+C to skip)")
            recording = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype="int16")
            for i in range(duration):
                await asyncio.sleep(1)
                print(f"  {i + 1}/{duration}s", end="\r", flush=True)
            sd.wait()
            audio_b64 = base64.b64encode(recording.tobytes()).decode("utf-8")
            print("\n✅ Audio captured\n")
            return audio_b64
        except Exception as exc:
            print(f"[WARN] Failed to capture audio: {exc}")
            return None

    async def _play_audio(self, chunks: Sequence[str], fs: int = 24000) -> None:
        joined = "".join(chunks)
        audio_bytes = base64.b64decode(joined)
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
        sd.play(audio, samplerate=fs)
        sd.wait()


def _frame_to_data_url(frame_bgr: np.ndarray, quality: int = 90) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", frame_bgr, encode_param)
    if not success:
        raise RuntimeError("Failed to encode frame for realtime message")
    image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_b64}"
