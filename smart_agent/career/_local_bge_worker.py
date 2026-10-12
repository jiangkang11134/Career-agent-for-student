"""JSONL worker launched by local_bge.py using the model environment."""

from __future__ import annotations

import argparse
import json
import os
import sys


class _CompatEmbedding:
    """Direct CLS encoder for environments without FlagEmbedding."""

    def __init__(self, model_path: str, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_kwargs = {"torch_dtype": torch.float16} if self.device.type == "cuda" else {}
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def encode_with_sparse(self, texts: list[str]) -> tuple[list[list[float]], list[dict[str, float]]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            vectors = self.model(**inputs).last_hidden_state[:, 0].float()
            vectors = self._torch.nn.functional.normalize(vectors, p=2, dim=1)
        dense = vectors.cpu().tolist()
        sparse: list[dict[str, float]] = []
        for row, mask in zip(inputs["input_ids"].detach().cpu().tolist(), inputs["attention_mask"].detach().cpu().tolist(), strict=True):
            weights: dict[str, float] = {}
            for token_id, active in zip(row, mask, strict=True):
                if not active or int(token_id) in {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}:
                    continue
                key = str(int(token_id))
                weights[key] = weights.get(key, 0.0) + 1.0
            norm = sum(value * value for value in weights.values()) ** 0.5 or 1.0
            sparse.append({key: value / norm for key, value in weights.items()})
        return dense, sparse

    def encode(self, texts: list[str]) -> list[list[float]]:
        dense, _sparse = self.encode_with_sparse(texts)
        return dense


class _CompatReranker:
    """Direct pair scorer for transformers versions without prepare_for_model."""

    def __init__(self, model_path: str, device: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_kwargs = {"torch_dtype": torch.float16} if self.device.type == "cuda" else {}
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def compute_score(self, pairs: list[list[str]], **_kwargs: object) -> list[float]:
        queries = [pair[0] for pair in pairs]
        documents = [pair[1] for pair in pairs]
        inputs = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
            scores = self._torch.sigmoid(logits)
        return scores.cpu().tolist()


def _write(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _load(mode: str, model_path: str, device: str):
    use_fp16 = device.startswith("cuda")
    if mode == "embedding":
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            return _CompatEmbedding(model_path, device)
        return BGEM3FlagModel(
            model_path,
            devices=device,
            use_fp16=use_fp16,
            normalize_embeddings=True,
            return_dense=True,
            return_sparse=True,
        )
    return _CompatReranker(model_path, device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("embedding", "reranker"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        model = _load(args.mode, args.model_path, args.device)
    except Exception as exc:
        _write({"status": "error", "mode": args.mode, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    _write({"status": "ready", "mode": args.mode, "device": args.device})
    for line in sys.stdin:
        request: object = {}
        try:
            request = json.loads(line)
            if request.get("action") == "shutdown":
                return 0
            request_id = request.get("id")
            if args.mode == "embedding" and request.get("action") == "embed":
                texts = request.get("texts")
                if not isinstance(texts, list) or not texts:
                    raise ValueError("texts must be a non-empty list")
                if isinstance(model, _CompatEmbedding):
                    vectors, sparse = model.encode_with_sparse(texts)
                else:
                    result = model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
                    dense = result.get("dense_vecs")
                    vectors = dense.tolist() if hasattr(dense, "tolist") else dense
                    sparse = result.get("lexical_weights") or result.get("sparse_vecs") or []
                    sparse = [item.tolist() if hasattr(item, "tolist") else item for item in sparse]
                _write({"id": request_id, "ok": True, "vectors": vectors, "sparse_vectors": sparse})
            elif args.mode == "reranker" and request.get("action") == "rerank":
                query = str(request.get("query") or "").strip()
                documents = request.get("documents")
                if not query or not isinstance(documents, list) or not documents:
                    raise ValueError("query and documents are required")
                scores = model.compute_score([[query, str(document)] for document in documents], normalize=True)
                if not isinstance(scores, list):
                    scores = [scores]
                _write({"id": request_id, "ok": True, "scores": [float(score) for score in scores]})
            else:
                raise ValueError("worker action does not match its mode")
        except Exception as exc:
            _write(
                {
                    "id": request.get("id") if isinstance(request, dict) else None,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
