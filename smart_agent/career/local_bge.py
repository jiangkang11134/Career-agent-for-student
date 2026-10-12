"""Persistent local BGE-M3 embedding and reranker adapters."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable


class _RetryableBGEWorkerError(RuntimeError):
    """A worker transport/protocol error that is safe to retry once."""


def _required_path(value: str | Path, name: str, *, file: bool = False) -> Path:
    path = Path(value).expanduser().resolve()
    valid = path.is_file() if file else path.is_dir()
    if not valid:
        kind = "file" if file else "directory"
        raise ValueError(f"{name} must be an existing {kind}: {path}")
    return path


def _runtime_value(runtime: dict[str, Any], key: str, env_name: str) -> str:
    return str(runtime.get(key) or os.environ.get(env_name) or "").strip()


class _PersistentBGEWorker:
    def __init__(
        self,
        *,
        python_path: str | Path,
        model_path: str | Path,
        mode: str,
        device: str,
        process_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        if mode not in {"embedding", "reranker"}:
            raise ValueError("Local BGE worker mode is invalid")
        self.python_path = _required_path(python_path, "Local BGE Python", file=True)
        self.model_path = _required_path(model_path, "Local BGE model")
        self.mode = mode
        self.device = str(device or "cpu").strip().lower()
        if not (self.device == "cpu" or self.device.startswith("cuda")):
            raise ValueError("Local BGE device must be cpu or cuda[:index]")
        self._process_factory = process_factory
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._request_id = 0
        self._stderr: deque[str] = deque(maxlen=20)
        atexit.register(self.close)

    @property
    def ready(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _start(self) -> subprocess.Popen[str]:
        if self.ready:
            assert self._process is not None
            return self._process
        worker_script = Path(__file__).with_name("_local_bge_worker.py")
        command = [
            str(self.python_path),
            "-u",
            str(worker_script),
            "--mode",
            self.mode,
            "--model-path",
            str(self.model_path),
            "--device",
            self.device,
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = self._process_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        self._process = process
        if process.stderr is not None:
            threading.Thread(target=self._drain_stderr, args=(process,), daemon=True).start()
        ready_line = process.stdout.readline() if process.stdout is not None else ""
        if not ready_line:
            detail = " | ".join(self._stderr) or f"exit={process.poll()}"
            self.close()
            raise _RetryableBGEWorkerError(f"Local BGE worker failed to start: {detail}")
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            self.close()
            raise _RetryableBGEWorkerError("Local BGE worker emitted an invalid startup response") from exc
        if ready.get("status") != "ready" or ready.get("mode") != self.mode:
            self.close()
            raise _RetryableBGEWorkerError(f"Local BGE worker startup failed: {ready.get('error') or ready}")
        return process

    def _drain_stderr(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            value = line.strip()
            if value:
                self._stderr.append(value)

    def _request_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        process = self._start()
        if process.stdin is None or process.stdout is None:
            raise _RetryableBGEWorkerError("Local BGE worker pipes are unavailable")
        self._request_id += 1
        request_id = self._request_id
        request_payload = dict(payload)
        request_payload["id"] = request_id
        # The configured D:/RAG interpreter may use GBK for ``sys.stdin`` on
        # Windows. ASCII-escape payload text so Unicode project content cannot
        # corrupt the JSONL protocol before the worker parses it.
        process.stdin.write(json.dumps(request_payload, ensure_ascii=True) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            detail = " | ".join(self._stderr) or f"exit={process.poll()}"
            self.close()
            raise _RetryableBGEWorkerError(f"Local BGE worker stopped unexpectedly: {detail}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _RetryableBGEWorkerError("Local BGE worker emitted an invalid response") from exc
        if not isinstance(response, dict):
            raise _RetryableBGEWorkerError("Local BGE worker response must be a JSON object")
        response_id = response.get("id")
        if str(response_id) != str(request_id):
            raise _RetryableBGEWorkerError(
                "Local BGE worker response ID mismatch "
                f"(expected {request_id!r}, received {response_id!r}, "
                f"response_keys={sorted(response)}, worker_error={response.get('error')!r})"
            )
        if not response.get("ok"):
            raise _RetryableBGEWorkerError(
                f"Local BGE request failed: {response.get('error') or 'unknown error'}"
            )
        return response

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            for attempt in range(2):
                try:
                    return self._request_once(payload)
                except _RetryableBGEWorkerError as exc:
                    self.close()
                    if attempt:
                        raise RuntimeError(f"{exc}; worker restart retry exhausted") from exc
            raise AssertionError("unreachable")

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                    process.stdin.flush()
                process.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()


class LocalBGEInterviewEmbedding:
    def __init__(
        self,
        *,
        python_path: str | Path,
        model_path: str | Path,
        device: str = "cpu",
        process_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        self.model_path = _required_path(model_path, "Local BGE-M3 model")
        self.device = str(device or "cpu").strip().lower()
        self.name = f"local-bge-m3:{self.model_path.name}:{self.device}"
        self._worker = _PersistentBGEWorker(
            python_path=python_path,
            model_path=self.model_path,
            mode="embedding",
            device=self.device,
            process_factory=process_factory,
        )

    @classmethod
    def from_runtime(cls, runtime: dict[str, Any] | None) -> "LocalBGEInterviewEmbedding | None":
        runtime = runtime or {}
        python_path = _runtime_value(runtime, "careerLocalBGEPython", "CAREER_LOCAL_BGE_PYTHON")
        model_path = _runtime_value(runtime, "careerLocalBGEModel", "CAREER_LOCAL_BGE_MODEL")
        if not python_path and not model_path:
            return None
        if not python_path or not model_path:
            raise ValueError("Local BGE embedding requires both Python and model paths")
        device = _runtime_value(runtime, "careerLocalBGEDevice", "CAREER_LOCAL_BGE_DEVICE") or "cpu"
        return cls(python_path=python_path, model_path=model_path, device=device)

    @property
    def ready(self) -> bool:
        return self._worker.ready

    def embed(self, texts: list[str]) -> list[list[float]]:
        dense, _sparse = self.embed_with_sparse(texts)
        return dense

    def embed_with_sparse(self, texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
        if not texts:
            return [], []
        if any(not str(text).strip() for text in texts):
            raise ValueError("Local BGE embedding input cannot be empty")
        response = self._worker.request({"action": "embed", "texts": texts})
        vectors = response.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError("Local BGE returned an unexpected vector count")
        normalized = [[float(value) for value in vector] for vector in vectors if isinstance(vector, list)]
        if len(normalized) != len(texts) or any(not vector for vector in normalized):
            raise ValueError("Local BGE returned an empty vector")
        dimensions = {len(vector) for vector in normalized}
        if len(dimensions) != 1:
            raise ValueError("Local BGE returned inconsistent vector dimensions")
        raw_sparse = response.get("sparse_vectors")
        sparse: list[dict[str, float]] = []
        if isinstance(raw_sparse, list) and len(raw_sparse) == len(texts):
            for item in raw_sparse:
                if isinstance(item, dict):
                    sparse.append({str(key): float(value) for key, value in item.items() if float(value) != 0.0})
                else:
                    sparse.append({})
        else:
            sparse = [{} for _ in texts]
        return normalized, sparse

    def close(self) -> None:
        self._worker.close()


class LocalBGEInterviewReranker:
    def __init__(
        self,
        *,
        python_path: str | Path,
        model_path: str | Path,
        device: str = "cpu",
        process_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        self.model_path = _required_path(model_path, "Local BGE reranker model")
        self.device = str(device or "cpu").strip().lower()
        self.name = f"local-bge-reranker:{self.model_path.name}:{self.device}"
        self._worker = _PersistentBGEWorker(
            python_path=python_path,
            model_path=self.model_path,
            mode="reranker",
            device=self.device,
            process_factory=process_factory,
        )

    @classmethod
    def from_runtime(cls, runtime: dict[str, Any] | None) -> "LocalBGEInterviewReranker | None":
        runtime = runtime or {}
        python_path = _runtime_value(runtime, "careerLocalBGEPython", "CAREER_LOCAL_BGE_PYTHON")
        model_path = _runtime_value(runtime, "careerLocalBGERerankerModel", "CAREER_LOCAL_BGE_RERANKER_MODEL")
        if not model_path:
            return None
        if not python_path:
            raise ValueError("Local BGE reranker requires CAREER_LOCAL_BGE_PYTHON")
        device = _runtime_value(runtime, "careerLocalBGEDevice", "CAREER_LOCAL_BGE_DEVICE") or "cpu"
        return cls(python_path=python_path, model_path=model_path, device=device)

    @property
    def ready(self) -> bool:
        return self._worker.ready

    def rerank(self, query: str, rows: list[tuple[Any, Any, float]]) -> list[tuple[Any, Any, float]]:
        if not rows:
            return []
        response = self._worker.request(
            {"action": "rerank", "query": str(query), "documents": [row[0].text for row in rows]}
        )
        scores = response.get("scores")
        if not isinstance(scores, list) or len(scores) != len(rows):
            raise ValueError("Local BGE reranker returned an unexpected score count")
        ranked = [(row[0], row[1], float(scores[index])) for index, row in enumerate(rows)]
        if any(not 0 <= score <= 1 for _chunk, _source, score in ranked):
            raise ValueError("Local BGE reranker score must be between 0 and 1")
        return sorted(ranked, key=lambda row: (-row[2], row[0].chunk_id))

    def close(self) -> None:
        self._worker.close()
