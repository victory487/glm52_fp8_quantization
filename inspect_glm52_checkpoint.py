#!/usr/bin/env python3
"""
Fast metadata-only inspection of a sharded GLM-5.2 safetensors checkpoint.
No full tensor is loaded into RAM or VRAM.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import Counter
from pathlib import Path


PRESERVE_PATTERNS = {
    "core": [
        re.compile(r"^model\.embed_tokens(?:\.|$)"),
        re.compile(r"^lm_head(?:\.|$)"),
        re.compile(r".*\.mlp\.gate(?:\.|$)"),
    ],
    "official_fp8": [
        re.compile(r".*\.self_attn\.indexers_proj(?:\.|$)"),
        re.compile(r".*\.self_attn\.indexer\.weights_proj(?:\.|$)"),
        re.compile(r"^model\.layers\.78\.eh_proj(?:\.|$)"),
    ],
    "conservative_extra": [
        re.compile(r"^model\.layers\.[0-2](?:\.|$)"),
        re.compile(r"^model\.layers\.78(?:\.|$)"),
    ],
}


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as file:
        raw = file.read(8)
        if len(raw) != 8:
            raise ValueError(f"Invalid safetensors file: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        header = json.loads(file.read(header_len))
    header.pop("__metadata__", None)
    return header


def tensor_size_bytes(meta: dict) -> int:
    offsets = meta.get("data_offsets")
    if offsets and len(offsets) == 2:
        return int(offsets[1]) - int(offsets[0])

    elements = 1
    for dim in meta["shape"]:
        elements *= int(dim)
    return elements * DTYPE_BYTES[meta["dtype"]]


def matches(name: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.match(name) for pattern in patterns)


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    return f"{number:.2f} TiB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--profile",
        choices=["aggressive", "balanced", "conservative"],
        default="balanced",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    config = load_json(model_dir / "config.json")
    index = load_json(model_dir / "model.safetensors.index.json")
    weight_map: dict[str, str] = index["weight_map"]

    shard_headers: dict[str, dict] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_headers[shard_name] = read_safetensors_header(model_dir / shard_name)

    dtype_bytes: Counter[str] = Counter()
    dtype_tensors: Counter[str] = Counter()
    missing: list[str] = []
    lora_names: list[str] = []

    patterns = list(PRESERVE_PATTERNS["core"])
    if args.profile in {"balanced", "conservative"}:
        patterns.extend(PRESERVE_PATTERNS["official_fp8"])
    if args.profile == "conservative":
        patterns.extend(PRESERVE_PATTERNS["conservative_extra"])

    preserved_bytes = 0
    preserved_names: list[str] = []

    for name, shard_name in weight_map.items():
        meta = shard_headers[shard_name].get(name)
        if meta is None:
            missing.append(name)
            continue
        size = tensor_size_bytes(meta)
        dtype = meta["dtype"]
        dtype_bytes[dtype] += size
        dtype_tensors[dtype] += 1

        if "lora_" in name.lower() or ".adapter" in name.lower():
            lora_names.append(name)

        if matches(name, patterns):
            preserved_bytes += size
            preserved_names.append(name)

    print("=== Checkpoint metadata ===")
    print(f"path                 : {model_dir}")
    print(f"architecture         : {config.get('architectures')}")
    print(f"declared dtype       : {config.get('dtype', config.get('torch_dtype'))}")
    print(f"quantization_config  : {config.get('quantization_config') is not None}")
    print(f"moe_router_dtype     : {config.get('moe_router_dtype')!r}")
    print(f"num_hidden_layers    : {config.get('num_hidden_layers')}")
    print(f"num_nextn_layers     : {config.get('num_nextn_predict_layers')}")
    print(f"layer 78 present     : {any(n.startswith('model.layers.78.') for n in weight_map)}")
    print(f"safetensors shards   : {len(shard_headers)}")
    print(f"indexed tensors      : {len(weight_map)}")
    print(f"missing header items : {len(missing)}")
    print(f"LoRA/adapter tensors : {len(lora_names)}")

    print("\nTensor storage by dtype:")
    for dtype, size in dtype_bytes.most_common():
        print(
            f"  {dtype:10s} {dtype_tensors[dtype]:8d} tensors "
            f"{human_bytes(size):>12s}"
        )

    print(f"\nSelected profile     : {args.profile}")
    print(f"preserved tensors    : {len(preserved_names)}")
    print(f"preserved source size: {human_bytes(preserved_bytes)}")
    print(
        "approx. BF16-vs-FP8 overhead for preserved weights: "
        f"{human_bytes(preserved_bytes // 2)}"
    )

    print("\nPreserved-name examples:")
    for name in preserved_names[:30]:
        print(f"  {name}")
    if len(preserved_names) > 30:
        print(f"  ... and {len(preserved_names) - 30} more")

    errors: list[str] = []
    if config.get("quantization_config") is not None:
        errors.append("input config contains quantization_config")
    if lora_names:
        errors.append("LoRA/adapter tensors are still present")
    if missing:
        errors.append("index/header mismatch detected")
    if config.get("architectures") != ["GlmMoeDsaForCausalLM"]:
        errors.append("unexpected architecture declaration")
    if config.get("moe_router_dtype") not in {None, "float32"}:
        errors.append("unexpected moe_router_dtype; expected float32")

    if errors:
        print("\nFAILED preflight checks:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(2)

    print("\nPreflight checks passed.")


if __name__ == "__main__":
    main()
