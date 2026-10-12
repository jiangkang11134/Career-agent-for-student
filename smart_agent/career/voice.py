"""ASR/TTS adapter contracts for voice interviews."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

VOICE_SCHEMA_VERSION = "1.0"


class SpeechKind(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"


class SpeechPlaybackState(str, Enum):
    QUEUED = "queued"
    PLAYING = "playing"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class SpeechEvent:
    event_id: str
    kind: SpeechKind
    text: str
    stream_id: str
    sequence: int
    created_at: float = field(default_factory=time.time)
    schema_version: str = VOICE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.event_id or "").strip() or not str(self.stream_id or "").strip():
            raise ValueError("SpeechEvent IDs are required")
        if not str(self.text or "").strip():
            raise ValueError("SpeechEvent text is required")
        object.__setattr__(self, "kind", SpeechKind(self.kind))
        if self.sequence < 0:
            raise ValueError("SpeechEvent sequence cannot be negative")
        if self.schema_version != VOICE_SCHEMA_VERSION:
            raise ValueError("Unsupported SpeechEvent schema version")

    @classmethod
    def create(cls, text: str, *, stream_id: str, sequence: int, final: bool = False) -> "SpeechEvent":
        return cls(
            event_id=f"speech-{uuid.uuid4().hex[:16]}",
            kind=SpeechKind.FINAL if final else SpeechKind.PARTIAL,
            text=text,
            stream_id=stream_id,
            sequence=sequence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind.value,
            "text": self.text,
            "stream_id": self.stream_id,
            "sequence": self.sequence,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class SpeechPlayback:
    playback_id: str
    text: str
    state: SpeechPlaybackState = SpeechPlaybackState.QUEUED
    started_at: float | None = None
    finished_at: float | None = None
    audio_base64: str = ""
    media_type: str = "audio/mpeg"

    def __post_init__(self) -> None:
        if not str(self.playback_id or "").strip() or not str(self.text or "").strip():
            raise ValueError("SpeechPlayback identity and text are required")
        object.__setattr__(self, "state", SpeechPlaybackState(self.state))
        object.__setattr__(self, "audio_base64", str(self.audio_base64 or ""))
        object.__setattr__(self, "media_type", str(self.media_type or "audio/mpeg").strip())

    def to_dict(self, *, include_audio: bool = True) -> dict[str, Any]:
        return {
            "playback_id": self.playback_id,
            "text": self.text,
            "state": self.state.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "audio_base64": self.audio_base64 if include_audio else "",
            "media_type": self.media_type,
        }


@dataclass(frozen=True, slots=True)
class VoiceRecordingRef:
    recording_id: str
    session_id: str
    path: str
    media_type: str
    content_sha256: str
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        for name in ("recording_id", "session_id", "path", "media_type"):
            if not str(getattr(self, name) or "").strip():
                raise ValueError(f"{name} is required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_sha256):
            raise ValueError("content_sha256 must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "session_id": self.session_id,
            "path": self.path,
            "media_type": self.media_type,
            "content_sha256": self.content_sha256,
            "created_at": self.created_at,
        }


class VoiceRecordingStore:
    """Explicit-consent recording store kept outside text interview sessions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def save(
        self,
        audio: bytes,
        *,
        session_id: str,
        media_type: str = "audio/webm",
        consent: bool = False,
    ) -> VoiceRecordingRef:
        if not consent:
            raise PermissionError("Explicit consent is required before saving a voice recording")
        if not audio:
            raise ValueError("audio is required")
        normalized_session = str(session_id or "").strip()
        if not normalized_session:
            raise ValueError("session_id is required")
        normalized_media = str(media_type or "audio/webm").strip().lower()
        suffix = ".webm" if normalized_media == "audio/webm" else ".bin"
        digest = hashlib.sha256(audio).hexdigest()
        recording_id = f"recording-{digest[:20]}"
        target = self.root / normalized_session / f"{recording_id}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(audio)
        return VoiceRecordingRef(
            recording_id=recording_id,
            session_id=normalized_session,
            path=str(target),
            media_type=normalized_media,
            content_sha256=digest,
        )


def _post_json(
    request: urllib.request.Request,
    *,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Voice provider response must be a JSON object")
            return payload
        except (TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            if retry_backoff_seconds > 0:
                time.sleep(retry_backoff_seconds * (attempt + 1))
    raise RuntimeError(f"Voice provider request failed: {last_error}")


class SpeechToTextAdapter(Protocol):
    name: str

    def start(self, *, stream_id: str, media_type: str = "audio/webm") -> None: ...

    def push_audio(self, audio: bytes) -> tuple[SpeechEvent, ...]: ...

    def stop(self) -> SpeechEvent | None: ...

    def cancel(self, *, stream_id: str) -> None: ...


class TextToSpeechAdapter(Protocol):
    name: str

    def synthesize(self, text: str) -> SpeechPlayback: ...

    def cancel(self, playback_id: str) -> None: ...


class VoiceInterviewGateway(Protocol):
    """Connects provider adapters to Interviewer Agent without owning its state."""

    asr: SpeechToTextAdapter
    tts: TextToSpeechAdapter

    def accept_final(self, event: SpeechEvent) -> None: ...

    def speak(self, text: str) -> SpeechPlayback: ...


@dataclass(frozen=True, slots=True)
class TurnDetectionResult:
    should_finalize: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_finalize": self.should_finalize,
            "reason": self.reason,
        }


class TurnDetector(Protocol):
    name: str

    def observe(
        self,
        *,
        partial_text: str,
        silence_ms: int = 0,
        elapsed_ms: int = 0,
        audio_bytes: int = 0,
    ) -> TurnDetectionResult: ...


class RuleBasedTurnDetector:
    """Small deterministic turn detector for WebSocket/HTTP voice relay."""

    name = "rule-based-turn-detector-v1"

    def __init__(
        self,
        *,
        silence_finalize_ms: int = 900,
        punctuation_silence_ms: int = 250,
        max_turn_ms: int = 45_000,
        min_text_chars: int = 4,
    ) -> None:
        if silence_finalize_ms < 0 or punctuation_silence_ms < 0 or max_turn_ms < 1000 or min_text_chars < 1:
            raise ValueError("Invalid turn detector thresholds")
        self.silence_finalize_ms = silence_finalize_ms
        self.punctuation_silence_ms = punctuation_silence_ms
        self.max_turn_ms = max_turn_ms
        self.min_text_chars = min_text_chars

    def observe(
        self,
        *,
        partial_text: str,
        silence_ms: int = 0,
        elapsed_ms: int = 0,
        audio_bytes: int = 0,
    ) -> TurnDetectionResult:
        text = str(partial_text or "").strip()
        if not text:
            return TurnDetectionResult(False, "")
        if elapsed_ms >= self.max_turn_ms:
            return TurnDetectionResult(True, "max_turn_duration")
        if len(text) >= self.min_text_chars and silence_ms >= self.silence_finalize_ms:
            return TurnDetectionResult(True, "silence")
        if (
            len(text) >= self.min_text_chars
            and silence_ms >= self.punctuation_silence_ms
            and text.endswith((".", "?", "!", "。", "？", "！"))
        ):
            return TurnDetectionResult(True, "punctuation_and_pause")
        return TurnDetectionResult(False, "")


class BufferedSpeechToTextAdapter:
    """Deterministic adapter used for offline tests and provider fallbacks."""

    name = "buffered-text"

    def __init__(self) -> None:
        self.stream_id = ""
        self._events: list[SpeechEvent] = []
        self._cancelled = False

    def start(self, *, stream_id: str, media_type: str = "audio/webm") -> None:
        if not str(stream_id or "").strip():
            raise ValueError("stream_id is required")
        self.stream_id = stream_id
        self.media_type = str(media_type or "audio/webm").strip().lower()
        self._events = []
        self._cancelled = False

    def push_audio(self, audio: bytes) -> tuple[SpeechEvent, ...]:
        if not self.stream_id or self._cancelled or not audio:
            return ()
        text = audio.decode("utf-8", errors="ignore").strip()
        if not text:
            return ()
        event = SpeechEvent.create(text, stream_id=self.stream_id, sequence=len(self._events), final=False)
        if self._events and self._events[-1].text == event.text:
            return ()
        self._events.append(event)
        return (event,)

    def stop(self) -> SpeechEvent | None:
        if self._cancelled or not self._events:
            return None
        event = SpeechEvent.create(
            self._events[-1].text,
            stream_id=self.stream_id,
            sequence=len(self._events),
            final=True,
        )
        self._events.append(event)
        return event

    def cancel(self, *, stream_id: str) -> None:
        if stream_id == self.stream_id:
            self._cancelled = True


class BufferedTextToSpeechAdapter:
    name = "buffered-tts"

    def __init__(self) -> None:
        self.playbacks: dict[str, SpeechPlayback] = {}

    def synthesize(self, text: str) -> SpeechPlayback:
        if not str(text or "").strip():
            raise ValueError("TTS text is required")
        playback = SpeechPlayback(playback_id=f"playback-{uuid.uuid4().hex[:16]}", text=text)
        self.playbacks[playback.playback_id] = playback
        return playback

    def mark_playing(self, playback_id: str) -> SpeechPlayback:
        current = self.playbacks[playback_id]
        updated = SpeechPlayback(
            playback_id=current.playback_id,
            text=current.text,
            state=SpeechPlaybackState.PLAYING,
            started_at=time.time(),
            audio_base64=current.audio_base64,
            media_type=current.media_type,
        )
        self.playbacks[playback_id] = updated
        return updated

    def complete(self, playback_id: str) -> SpeechPlayback:
        current = self.playbacks[playback_id]
        updated = SpeechPlayback(
            playback_id=current.playback_id,
            text=current.text,
            state=SpeechPlaybackState.COMPLETED,
            started_at=current.started_at,
            finished_at=time.time(),
            audio_base64=current.audio_base64,
            media_type=current.media_type,
        )
        self.playbacks[playback_id] = updated
        return updated

    def cancel(self, playback_id: str) -> None:
        current = self.playbacks.get(playback_id)
        if current is None:
            return
        self.playbacks[playback_id] = SpeechPlayback(
            playback_id=current.playback_id,
            text=current.text,
            state=SpeechPlaybackState.CANCELLED,
            started_at=current.started_at,
            finished_at=time.time(),
            audio_base64=current.audio_base64,
            media_type=current.media_type,
        )


class InterviewVoiceBridge:
    """I/O bridge that never owns Interview Session, RAG, policy, or memory."""

    def __init__(
        self,
        *,
        asr: SpeechToTextAdapter,
        tts: TextToSpeechAdapter,
        submit_final,
    ) -> None:
        self.asr = asr
        self.tts = tts
        self.submit_final = submit_final
        self.partial_by_stream: dict[str, SpeechEvent] = {}

    def accept(self, event: SpeechEvent) -> bool:
        if event.kind is SpeechKind.PARTIAL:
            current = self.partial_by_stream.get(event.stream_id)
            if current is None or event.sequence > current.sequence:
                self.partial_by_stream[event.stream_id] = event
            return False
        self.partial_by_stream.pop(event.stream_id, None)
        self.submit_final(event.text)
        return True

    def interrupt(self, *, stream_id: str, playback_id: str | None = None) -> None:
        self.asr.cancel(stream_id=stream_id)
        if playback_id:
            self.tts.cancel(playback_id)

    def speak(self, text: str) -> SpeechPlayback:
        return self.tts.synthesize(text)


@dataclass(slots=True)
class VoiceStreamState:
    interview_id: str
    stream_id: str
    bridge: InterviewVoiceBridge
    submitted: dict[str, Any] = field(default_factory=dict)
    playback_id: str = ""
    media_type: str = "audio/webm"
    last_partial_text: str = ""
    last_final_reason: str = ""
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not str(self.interview_id or "").strip() or not str(self.stream_id or "").strip():
            raise ValueError("interview_id and stream_id are required")


@dataclass(frozen=True, slots=True)
class VoiceInputResult:
    events: tuple[SpeechEvent, ...]
    accepted_final: bool
    recordings: tuple[VoiceRecordingRef, ...] = ()
    answer_result: Any = None
    final_reason: str = ""
    turn_detection: TurnDetectionResult = field(default_factory=lambda: TurnDetectionResult(False, ""))

    def as_legacy_tuple(self) -> tuple[tuple[SpeechEvent, ...], bool, tuple[VoiceRecordingRef, ...]]:
        return self.events, self.accepted_final, self.recordings


class VoiceSessionRuntime:
    """Pipeline-style voice runtime for one-user interview streams.

    It owns ASR/TTS stream state only. Interview questions, answer evaluation,
    RAG, report writing, and memory remain outside this runtime.
    """

    def __init__(
        self,
        *,
        recording_store: VoiceRecordingStore | None = None,
        turn_detector: TurnDetector | None = None,
    ) -> None:
        self.recording_store = recording_store
        self.turn_detector = turn_detector or RuleBasedTurnDetector()
        self._streams: dict[tuple[str, str], VoiceStreamState] = {}

    def start(
        self,
        *,
        interview_id: str,
        stream_id: str,
        prompt_text: str,
        asr: SpeechToTextAdapter,
        tts: TextToSpeechAdapter,
        submit_final,
        media_type: str = "audio/webm",
    ) -> SpeechPlayback:
        key = self._key(interview_id, stream_id)
        submitted: dict[str, Any] = {}

        def capture_submission(text: str) -> None:
            submitted["answer_result"] = submit_final(text)

        bridge = InterviewVoiceBridge(asr=asr, tts=tts, submit_final=capture_submission)
        normalized_media_type = str(media_type or "audio/webm").strip().lower()
        bridge.asr.start(stream_id=stream_id, media_type=normalized_media_type)
        playback = bridge.speak(prompt_text)
        self._streams[key] = VoiceStreamState(
            interview_id=interview_id,
            stream_id=stream_id,
            bridge=bridge,
            submitted=submitted,
            playback_id=playback.playback_id,
            media_type=normalized_media_type,
        )
        return playback

    def push_audio(
        self,
        *,
        interview_id: str,
        stream_id: str,
        audio: bytes,
        save_recording: bool = False,
        media_type: str = "audio/webm",
        consent: bool = False,
    ) -> tuple[tuple[SpeechEvent, ...], bool, tuple[VoiceRecordingRef, ...]]:
        return self.push_audio_detailed(
            interview_id=interview_id,
            stream_id=stream_id,
            audio=audio,
            save_recording=save_recording,
            media_type=media_type,
            consent=consent,
        ).as_legacy_tuple()

    def push_audio_detailed(
        self,
        *,
        interview_id: str,
        stream_id: str,
        audio: bytes,
        save_recording: bool = False,
        media_type: str = "audio/webm",
        consent: bool = False,
        silence_ms: int = 0,
        elapsed_ms: int = 0,
        auto_finalize: bool = True,
    ) -> VoiceInputResult:
        state = self._get(interview_id, stream_id)
        recordings: tuple[VoiceRecordingRef, ...] = ()
        if save_recording:
            if self.recording_store is None:
                raise RuntimeError("Voice recording store is not configured")
            recordings = (
                self.recording_store.save(
                    audio,
                    session_id=interview_id,
                    media_type=media_type,
                    consent=consent,
                ),
            )
        events = state.bridge.asr.push_audio(audio)
        accepted_final = False
        for event in events:
            accepted_final = state.bridge.accept(event) or accepted_final
            state.updated_at = time.time()
            if event.kind is SpeechKind.PARTIAL:
                state.last_partial_text = event.text
        detection = self.turn_detector.observe(
            partial_text=state.last_partial_text,
            silence_ms=max(0, int(silence_ms)),
            elapsed_ms=max(0, int(elapsed_ms)),
            audio_bytes=len(audio),
        )
        answer_result = None
        final_reason = ""
        if auto_finalize and detection.should_finalize and not accepted_final:
            final_event, submitted, answer_result = self.stop(
                interview_id=interview_id,
                stream_id=stream_id,
                reason=detection.reason,
            )
            accepted_final = submitted
            final_reason = detection.reason
            if final_event is not None:
                events = (*events, final_event)
        return VoiceInputResult(
            events=events,
            accepted_final=accepted_final,
            recordings=recordings,
            answer_result=answer_result,
            final_reason=final_reason,
            turn_detection=detection,
        )

    def stop(
        self,
        *,
        interview_id: str,
        stream_id: str,
        reason: str = "manual_stop",
    ) -> tuple[SpeechEvent | None, bool, Any]:
        state = self._get(interview_id, stream_id)
        event = state.bridge.asr.stop()
        submitted = False
        if event is not None:
            submitted = state.bridge.accept(event)
            state.last_final_reason = str(reason or "manual_stop")
            state.updated_at = time.time()
        return event, submitted, state.submitted.get("answer_result")

    def speak(self, *, interview_id: str, stream_id: str, text: str) -> SpeechPlayback:
        state = self._get(interview_id, stream_id)
        playback = state.bridge.speak(text)
        state.playback_id = playback.playback_id
        state.bridge.asr.start(stream_id=stream_id, media_type=state.media_type)
        return playback

    def interrupt(self, *, interview_id: str, stream_id: str, playback_id: str | None = None) -> str | None:
        key = self._key(interview_id, stream_id)
        state = self._streams.pop(key, None)
        if state is None:
            raise KeyError("Unknown voice stream")
        selected_playback_id = str(playback_id or state.playback_id or "").strip() or None
        state.bridge.interrupt(stream_id=stream_id, playback_id=selected_playback_id)
        return selected_playback_id

    def close(self, *, interview_id: str, stream_id: str) -> None:
        self._streams.pop(self._key(interview_id, stream_id), None)

    def _get(self, interview_id: str, stream_id: str) -> VoiceStreamState:
        state = self._streams.get(self._key(interview_id, stream_id))
        if state is None:
            raise KeyError("Unknown voice stream")
        return state

    @staticmethod
    def _key(interview_id: str, stream_id: str) -> tuple[str, str]:
        normalized_interview = str(interview_id or "").strip()
        normalized_stream = str(stream_id or "").strip()
        if not normalized_interview or not normalized_stream:
            raise ValueError("interview_id and stream_id are required")
        return normalized_interview, normalized_stream


class HttpSpeechToTextAdapter:
    """Small HTTP adapter for provider-specific ASR gateways.

    The gateway must return ``{"text": "..."}``; provider authentication and
    endpoint selection remain configuration concerns outside Interview Runtime.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.1,
    ) -> None:
        self.name = "http-asr"
        self.model = str(model or "").strip()
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.api_key = str(api_key or "").strip()
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.stream_id = ""
        self.sequence = 0
        self.last_text = ""
        self.media_type = "audio/webm"
        if not self.base_url or not self.api_key or not self.model:
            raise ValueError("ASR base_url, api_key, and model are required")

    def start(self, *, stream_id: str, media_type: str = "audio/webm") -> None:
        if not str(stream_id or "").strip():
            raise ValueError("stream_id is required")
        self.stream_id = stream_id
        self.sequence = 0
        self.last_text = ""
        self.media_type = str(media_type or "audio/webm").strip().lower()

    def push_audio(self, audio: bytes) -> tuple[SpeechEvent, ...]:
        if not self.stream_id or not audio:
            return ()
        request = urllib.request.Request(
            f"{self.base_url}/speech-to-text",
            data=json.dumps(
                {
                    "model": self.model,
                    "media_type": self.media_type,
                    "audio_base64": base64.b64encode(audio).decode("ascii"),
                }
            ).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        payload = _post_json(
            request,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
        )
        text = str(payload.get("text") or "").strip()
        if not text:
            return ()
        if text == self.last_text:
            return ()
        self.last_text = text
        event = SpeechEvent.create(text, stream_id=self.stream_id, sequence=self.sequence, final=False)
        self.sequence += 1
        return (event,)

    def stop(self) -> SpeechEvent | None:
        if not self.stream_id or not self.last_text:
            return None
        return SpeechEvent.create(self.last_text, stream_id=self.stream_id, sequence=self.sequence, final=True)

    def cancel(self, *, stream_id: str) -> None:
        if stream_id == self.stream_id:
            self.stream_id = ""


class HttpTextToSpeechAdapter:
    """HTTP TTS adapter accepting ``{"audio_base64": "..."}`` responses."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.1,
        voice: str = "",
        style: str = "",
    ) -> None:
        self.name = "http-tts"
        self.model = str(model or "").strip()
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.api_key = str(api_key or "").strip()
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.voice = str(voice or "").strip()
        self.style = str(style or "").strip()
        self._active: dict[str, SpeechPlayback] = {}
        if not self.base_url or not self.api_key or not self.model:
            raise ValueError("TTS base_url, api_key, and model are required")

    def synthesize(self, text: str) -> SpeechPlayback:
        if not str(text or "").strip():
            raise ValueError("TTS text is required")
        request_payload = {"model": self.model, "text": text}
        if self.voice:
            request_payload["voice"] = self.voice
        if self.style:
            request_payload["style"] = self.style
        request = urllib.request.Request(
            f"{self.base_url}/text-to-speech",
            data=json.dumps(request_payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        payload = _post_json(
            request,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
        )
        audio_base64 = str(payload.get("audio_base64") or "").strip()
        if not audio_base64:
            raise ValueError("TTS provider returned no audio_base64 payload")
        playback = SpeechPlayback(
            playback_id=f"playback-{uuid.uuid4().hex[:16]}",
            text=text,
            audio_base64=audio_base64,
            media_type=str(payload.get("media_type") or "audio/mpeg"),
        )
        self._active[playback.playback_id] = playback
        return playback

    def cancel(self, playback_id: str) -> None:
        current = self._active.get(playback_id)
        if current is not None:
            self._active[playback_id] = replace_playback(current, SpeechPlaybackState.CANCELLED)


class CosyVoiceTextToSpeechAdapter:
    """Local CosyVoice HTTP adapter kept as an experimental fallback."""

    name = "cosyvoice-tts"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:9881",
        model: str = "cosyvoice3-0.5b",
        api_key: str = "",
        timeout_seconds: float = 30.0,
        max_retries: int = 0,
        retry_backoff_seconds: float = 0.1,
        voice: str = "",
        style: str = "",
    ) -> None:
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.model = str(model or "cosyvoice3-0.5b").strip()
        self.api_key = str(api_key or "").strip()
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.voice = str(voice or "").strip()
        self.style = str(style or "").strip()
        self._active: dict[str, SpeechPlayback] = {}
        if not self.base_url or not self.model:
            raise ValueError("CosyVoice TTS base_url and model are required")

    def synthesize(self, text: str) -> SpeechPlayback:
        normalized = str(text or "").strip()
        if not normalized:
            raise ValueError("TTS text is required")
        request_payload = {"model": self.model, "text": normalized}
        if self.voice:
            request_payload["voice"] = self.voice
        if self.style:
            request_payload["style"] = self.style
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/text-to-speech",
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        payload = _post_json(
            request,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
        )
        audio_base64 = str(payload.get("audio_base64") or "").strip()
        if not audio_base64:
            raise ValueError("CosyVoice returned no audio_base64 payload")
        playback = SpeechPlayback(
            playback_id=f"playback-{uuid.uuid4().hex[:16]}",
            text=normalized,
            audio_base64=audio_base64,
            media_type=str(payload.get("media_type") or "audio/wav"),
        )
        self._active[playback.playback_id] = playback
        return playback

    def cancel(self, playback_id: str) -> None:
        current = self._active.get(playback_id)
        if current is not None:
            self._active[playback_id] = replace_playback(current, SpeechPlaybackState.CANCELLED)


class VolcengineTextToSpeechAdapter:
    """Volcengine/ByteDance OpenSpeech TTS adapter for production-quality Mandarin voices."""

    name = "volcengine-tts"

    def __init__(
        self,
        *,
        app_id: str,
        access_token: str,
        voice_type: str,
        base_url: str = "https://openspeech.bytedance.com/api/v1/tts",
        encoding: str = "mp3",
        speed_ratio: float = 1.0,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.1,
    ) -> None:
        self.app_id = str(app_id or "").strip()
        self.access_token = str(access_token or "").strip()
        self.voice_type = str(voice_type or "").strip()
        self.base_url = str(base_url or "").strip()
        self.encoding = str(encoding or "mp3").strip().lower()
        self.speed_ratio = float(speed_ratio)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self._active: dict[str, SpeechPlayback] = {}
        if not self.app_id or not self.access_token or not self.voice_type:
            raise ValueError("Volcengine TTS app_id, access token, and voice_type are required")
        if self.encoding not in {"mp3", "wav", "pcm", "ogg_opus"}:
            raise ValueError("Volcengine TTS encoding must be mp3, wav, pcm, or ogg_opus")

    def synthesize(self, text: str) -> SpeechPlayback:
        normalized = str(text or "").strip()
        if not normalized:
            raise ValueError("TTS text is required")
        reqid = f"tts-{uuid.uuid4().hex}"
        request_payload = {
            "app": {"appid": self.app_id, "token": self.access_token, "cluster": "volcano_tts"},
            "user": {"uid": "smart-agent-career"},
            "audio": {
                "voice_type": self.voice_type,
                "encoding": self.encoding,
                "speed_ratio": self.speed_ratio,
            },
            "request": {"reqid": reqid, "text": normalized, "operation": "query"},
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer; {self.access_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        payload = _post_json(
            request,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
        )
        code = int(payload.get("code", 0) or 0)
        if code != 3000:
            raise ValueError(str(payload.get("message") or f"Volcengine TTS failed with code {code}"))
        audio_base64 = str(payload.get("data") or "").strip()
        if not audio_base64:
            raise ValueError("Volcengine TTS returned no audio data")
        media_type = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "pcm": "audio/L16",
            "ogg_opus": "audio/ogg",
        }.get(self.encoding, "audio/mpeg")
        playback = SpeechPlayback(
            playback_id=f"playback-{uuid.uuid4().hex[:16]}",
            text=normalized,
            audio_base64=audio_base64,
            media_type=media_type,
        )
        self._active[playback.playback_id] = playback
        return playback

    def cancel(self, playback_id: str) -> None:
        current = self._active.get(playback_id)
        if current is not None:
            self._active[playback_id] = replace_playback(current, SpeechPlaybackState.CANCELLED)


class GPTSoVITSTextToSpeechAdapter:
    """GPT-SoVITS v2 ``/tts`` adapter returning provider audio bytes."""

    name = "gpt-sovits"

    def __init__(
        self,
        *,
        base_url: str,
        ref_audio_path: str,
        prompt_text: str = "",
        text_lang: str = "zh",
        prompt_lang: str = "zh",
        speed_factor: float = 1.0,
        timeout_seconds: float = 90.0,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.2,
    ) -> None:
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.ref_audio_path = str(ref_audio_path or "").strip()
        self.prompt_text = str(prompt_text or "").strip()
        self.text_lang = str(text_lang or "zh").strip()
        self.prompt_lang = str(prompt_lang or "zh").strip()
        self.speed_factor = float(speed_factor)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self._active: dict[str, SpeechPlayback] = {}
        if not self.base_url or not self.ref_audio_path:
            raise ValueError("GPT-SoVITS base_url and ref_audio_path are required")
        if self.speed_factor <= 0:
            raise ValueError("GPT-SoVITS speed_factor must be positive")

    def synthesize(self, text: str) -> SpeechPlayback:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise ValueError("TTS text is required")
        query = urllib.parse.urlencode(
            {
                "text": normalized_text,
                "text_lang": self.text_lang,
                "ref_audio_path": self.ref_audio_path,
                "prompt_lang": self.prompt_lang,
                "prompt_text": self.prompt_text,
                "text_split_method": "cut5",
                "batch_size": 1,
                "media_type": "wav",
                "streaming_mode": "false",
                "speed_factor": self.speed_factor,
            }
        )
        request = urllib.request.Request(f"{self.base_url}/tts?{query}", method="GET")
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                    audio = response.read()
                    media_type = str(response.headers.get("Content-Type") or "audio/wav").split(";", 1)[0]
                if not audio:
                    raise ValueError("GPT-SoVITS returned empty audio")
                playback = SpeechPlayback(
                    playback_id=f"playback-{uuid.uuid4().hex[:16]}",
                    text=normalized_text,
                    audio_base64=base64.b64encode(audio).decode("ascii"),
                    media_type=media_type,
                )
                self._active[playback.playback_id] = playback
                return playback
            except (TimeoutError, OSError, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                if self.retry_backoff_seconds > 0:
                    time.sleep(self.retry_backoff_seconds * (attempt + 1))
        raise RuntimeError(f"GPT-SoVITS request failed: {last_error}")

    def cancel(self, playback_id: str) -> None:
        current = self._active.get(playback_id)
        if current is not None:
            self._active[playback_id] = replace_playback(current, SpeechPlaybackState.CANCELLED)


def replace_playback(playback: SpeechPlayback, state: SpeechPlaybackState) -> SpeechPlayback:
    return SpeechPlayback(
        playback_id=playback.playback_id,
        text=playback.text,
        state=state,
        started_at=playback.started_at,
        finished_at=time.time() if state in {SpeechPlaybackState.CANCELLED, SpeechPlaybackState.COMPLETED} else None,
        audio_base64=playback.audio_base64,
        media_type=playback.media_type,
    )
