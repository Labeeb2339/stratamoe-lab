#!/usr/bin/env python3
"""Capture a canonical RouterTrace v2 from pinned Switch-Base-8 routing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_ID = "google/switch-base-8"
MODEL_REVISION = "92fe2d22b024d9937146fe097ba3d3a7ba146e1b"
CAPTURE_SEED = 2339
EXPECTED_SPARSE_ENCODER_BLOCKS = [1, 3, 5, 7, 9, 11]


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)


def load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Prompt manifest schema_version must be 1.")

    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompt manifest must contain a non-empty prompts array.")

    seen: set[str] = set()
    for index, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            raise TypeError(f"Prompt {index} must be an object.")
        prompt_id = prompt.get("id")
        prompt_text = prompt.get("text")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ValueError(f"Prompt {index} has an invalid id.")
        if prompt_id in seen:
            raise ValueError(f"Duplicate prompt id: {prompt_id}")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            raise ValueError(f"Prompt {prompt_id} has empty text.")
        seen.add(prompt_id)

    return manifest, sha256_bytes(raw)


def capture_once(
    model: Any,
    encoded: dict[str, torch.Tensor],
    prompt_ids: list[str],
) -> tuple[list[list[list[int]]], list[int], list[dict[str, Any]]]:
    batch_size = encoded["input_ids"].shape[0]
    decoder_start = model.config.decoder_start_token_id
    if decoder_start is None:
        raise ValueError("Model config has no decoder_start_token_id.")

    decoder_input_ids = torch.full(
        (batch_size, 1),
        int(decoder_start),
        dtype=torch.long,
        device="cpu",
    )

    with torch.inference_mode():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            output_router_logits=True,
            return_dict=True,
            use_cache=False,
        )

    router_outputs = outputs.encoder_router_logits
    if not isinstance(router_outputs, tuple):
        raise TypeError("Expected encoder_router_logits to be a tuple.")

    sparse_blocks: list[int] = []
    layer_top1: list[torch.Tensor] = []
    for block_index, router_output in enumerate(router_outputs):
        if not isinstance(router_output, tuple) or len(router_output) < 2:
            continue
        router_logits = router_output[0]
        if not isinstance(router_logits, torch.Tensor) or router_logits.ndim != 3:
            raise TypeError(f"Unexpected router logits at encoder block {block_index}.")
        sparse_blocks.append(block_index)
        layer_top1.append(torch.argmax(router_logits, dim=-1).to(torch.int64).cpu())

    if sparse_blocks != EXPECTED_SPARSE_ENCODER_BLOCKS:
        raise RuntimeError(
            f"Sparse encoder blocks changed: {sparse_blocks}; "
            f"expected {EXPECTED_SPARSE_ENCODER_BLOCKS}."
        )

    selections: list[list[list[int]]] = []
    prompt_spans: list[dict[str, Any]] = []
    token_cursor = 0
    for batch_index, prompt_id in enumerate(prompt_ids):
        valid_positions = torch.nonzero(
            encoded["attention_mask"][batch_index].to(torch.bool),
            as_tuple=False,
        ).flatten()
        token_ids = encoded["input_ids"][batch_index, valid_positions].to(torch.int64).cpu()
        start = token_cursor

        for token_position in valid_positions.tolist():
            selections.append(
                [
                    [int(layer_values[batch_index, token_position].item())]
                    for layer_values in layer_top1
                ]
            )
            token_cursor += 1

        prompt_spans.append(
            {
                "id": prompt_id,
                "token_start": start,
                "token_end_exclusive": token_cursor,
                "non_padding_tokens": int(valid_positions.numel()),
                "input_ids_sha256": sha256_bytes(token_ids.numpy().tobytes()),
            }
        )

    return selections, sparse_blocks, prompt_spans


def validate_selections(
    selections: list[list[list[int]]],
    layers: int,
    experts_per_layer: int,
) -> None:
    if not selections:
        raise ValueError("Trace contains no token selections.")

    for token_index, token_layers in enumerate(selections):
        if len(token_layers) != layers:
            raise ValueError(
                f"Token {token_index} has {len(token_layers)} layers; expected {layers}."
            )
        for layer_index, expert_ids in enumerate(token_layers):
            if len(expert_ids) != 1:
                raise ValueError(
                    f"Token {token_index}, layer {layer_index} is not top-1."
                )
            expert_id = expert_ids[0]
            if not isinstance(expert_id, int) or not 0 <= expert_id < experts_per_layer:
                raise ValueError(
                    f"Token {token_index}, layer {layer_index} has invalid expert {expert_id}."
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evidence/switch-base-8/prompt-manifest.json"),
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        default=Path("evidence/switch-base-8/router-trace.v2.json"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("evidence/switch-base-8/capture-metadata.json"),
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    set_determinism(CAPTURE_SEED)
    manifest, manifest_sha256 = load_manifest(args.manifest)
    prompt_ids = [prompt["id"] for prompt in manifest["prompts"]]
    prompt_texts = [prompt["text"] for prompt in manifest["prompts"]]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
        use_fast=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
        dtype=torch.float32,
    ).to("cpu")
    model.eval()

    resolved_model_revision = getattr(model.config, "_commit_hash", None)
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {})
    resolved_tokenizer_revision = (
        tokenizer_kwargs.get("_commit_hash") if isinstance(tokenizer_kwargs, dict) else None
    )
    if resolved_model_revision != MODEL_REVISION:
        raise RuntimeError(
            f"Resolved model revision {resolved_model_revision!r} does not match the pin."
        )
    if resolved_tokenizer_revision not in (None, MODEL_REVISION):
        raise RuntimeError(
            f"Resolved tokenizer revision {resolved_tokenizer_revision!r} does not match the pin."
        )

    encoded = tokenizer(
        prompt_texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=96,
        return_tensors="pt",
    )
    encoded = {key: value.to("cpu") for key, value in encoded.items()}

    first, sparse_blocks, prompt_spans = capture_once(model, encoded, prompt_ids)
    second, sparse_blocks_repeat, prompt_spans_repeat = capture_once(
        model, encoded, prompt_ids
    )
    if first != second or sparse_blocks != sparse_blocks_repeat or prompt_spans != prompt_spans_repeat:
        raise RuntimeError("Repeated capture produced different routing evidence.")

    experts_per_layer = int(model.config.num_experts)
    validate_selections(first, len(sparse_blocks), experts_per_layer)
    if sum(span["non_padding_tokens"] for span in prompt_spans) != len(first):
        raise RuntimeError("Prompt spans do not sum to the captured token count.")

    trace = {
        "version": 2,
        "source": {
            "kind": "captured",
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "tokenizer": {"revision": MODEL_REVISION},
            "software": {
                "transformersVersion": transformers.__version__,
                "pytorchVersion": torch.__version__,
            },
            "workload": {
                "kind": "prompt-manifest",
                "sha256": manifest_sha256,
            },
            "capture": {
                "seed": CAPTURE_SEED,
                "device": "cpu",
                "dtype": str(next(model.parameters()).dtype),
            },
        },
        "scenario": "domain-shift",
        "seed": CAPTURE_SEED,
        "tokens": len(first),
        "layers": len(sparse_blocks),
        "expertsPerLayer": experts_per_layer,
        "topK": 1,
        "selections": first,
    }
    trace_bytes = canonical_json_bytes(trace)
    selections_sha256 = sha256_bytes(canonical_json_bytes(first))
    experts_observed = [
        sorted({token[layer][0] for token in first}) for layer in range(len(sparse_blocks))
    ]
    metadata = {
        "schema_version": 1,
        "trace_file": args.trace_output.name,
        "trace_sha256": sha256_bytes(trace_bytes),
        "selections_sha256": selections_sha256,
        "selection_basis": "argmax(raw_encoder_router_logits)",
        "scope": "encoder sparse feed-forward layers",
        "padding_tokens_excluded": True,
        "special_tokens_included": True,
        "prompt_manifest_sha256": manifest_sha256,
        "prompt_spans": prompt_spans,
        "sparse_encoder_block_indices": sparse_blocks,
        "experts_observed_per_layer": experts_observed,
        "determinism": {"in_process_repeats": 2, "matched": True},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "sentencepiece": importlib.metadata.version("sentencepiece"),
            "device": "cpu",
            "model_dtype": str(next(model.parameters()).dtype),
            "router_logits_dtype": "torch.float32",
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
    }

    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.write_bytes(trace_bytes)
    args.metadata_output.write_bytes(canonical_json_bytes(metadata))

    print(
        json.dumps(
            {
                "trace": str(args.trace_output),
                "trace_sha256": metadata["trace_sha256"],
                "selections_sha256": selections_sha256,
                "tokens": len(first),
                "layers": len(sparse_blocks),
                "experts_per_layer": experts_per_layer,
                "top_k": 1,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
