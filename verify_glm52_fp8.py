#!/usr/bin/env python3
"""
Verify FP8_BLOCK output metadata and confirm BF16-retained tensors stay
unquantized. Optionally hash small preserved tensors to prove bit identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path


CORE_PATTERNS = [
    re.compile(r"^model\.embed_tokens(?:\.|$)"),
    re.compile(r"^lm_head(?:\.|$)"),
    re.compile(r".*\.mlp\.gate(?:\.|$)"),
]

OFFICIAL_FP8_PATTERNS = [
    re.compile(r".*\.self_attn\.indexers_proj(?:\.|$)"),
    re.compile(r".*\.self_attn\.indexer\.weights_proj(?:\.|$)"),
    re.compile(r"^model\.layers\.78\.eh_proj(?:\.|$)"),
]

PROFILE_PATTERNS = {
    "aggressive": [*CORE_PATTERNS],
    "balanced": [*CORE_PATTERNS, *OFFICIAL_FP8_PATTERNS],
    "conservative": [
        *CORE_PATTERNS,
        *OFFICIAL_FP8_PATTERNS,
        re.compile(r"^model\.layers\.[0-2](?:\.|$)"),
        re.compile(r"^model\.layers\.78(?:\.|$)"),
    ],
}


FP8_DTYPES = {"F8_E4M3", "F8_E5M2"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as file:
        raw = file.read(8)
        if len(raw) != 8:
            raise ValueError(f"Invalid safetensors file: {path}")
        header_len = struct.unpack("<Q", raw)[0]
        header = json.loads(file.read(header_len))
    header.pop("__metadata__", None)
    return header_len, header


def collect_metadata(model_dir: Path) -> tuple[dict, dict, dict]:
    config = load_json(model_dir / "config.json")
    index = load_json(model_dir / "model.safetensors.index.json")
    weight_map: dict[str, str] = index["weight_map"]

    headers: dict[str, tuple[int, dict]] = {}
    for shard in sorted(set(weight_map.values())):
        headers[shard] = read_header(model_dir / shard)

    tensor_meta: dict[str, dict] = {}
    for name, shard in weight_map.items():
        header_len, header = headers[shard]
        if name not in header:
            raise ValueError(f"{name} missing from header of {shard}")
        tensor_meta[name] = {
            **header[name],
            "shard": shard,
            "header_len": header_len,
        }
    return config, weight_map, tensor_meta


def matches(name: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.match(name) for pattern in patterns)


def tensor_bytes(meta: dict) -> int:
    start, end = meta["data_offsets"]
    return int(end) - int(start)


def hash_tensor(model_dir: Path, meta: dict, chunk_size: int = 16 << 20) -> str:
    path = model_dir / meta["shard"]
    start, end = (int(value) for value in meta["data_offsets"])
    absolute = 8 + int(meta["header_len"]) + start
    remaining = end - start

    digest = hashlib.sha256()
    with path.open("rb") as file:
        file.seek(absolute)
        while remaining:
            block = file.read(min(chunk_size, remaining))
            if not block:
                raise IOError(f"Unexpected EOF while hashing {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--quantized", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_PATTERNS),
        default="balanced",
    )
    parser.add_argument(
        "--hash-max-mb",
        type=float,
        default=0,
        help=(
            "Hash preserved tensors no larger than this size and compare source "
            "with output. Zero disables hashing."
        ),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    quantized = args.quantized.resolve()
    src_config, src_map, src_meta = collect_metadata(source)
    out_config, out_map, out_meta = collect_metadata(quantized)

    dtype_count = Counter(meta["dtype"] for meta in out_meta.values())
    fp8_names = [name for name, meta in out_meta.items() if meta["dtype"] in FP8_DTYPES]
    patterns = PROFILE_PATTERNS[args.profile]

    errors: list[str] = []
    preserved_checked = 0
    hashed = 0

    for name, src in src_meta.items():
        if not matches(name, patterns):
            continue
        preserved_checked += 1
        out = out_meta.get(name)
        if out is None:
            errors.append(f"preserved tensor missing in output: {name}")
            continue
        if src["shape"] != out["shape"]:
            errors.append(f"shape changed for preserved tensor: {name}")
        if out["dtype"] in FP8_DTYPES:
            errors.append(f"preserved tensor became FP8: {name}")
        if src["dtype"] != out["dtype"]:
            errors.append(
                f"dtype changed for preserved tensor {name}: "
                f"{src['dtype']} -> {out['dtype']}"
            )

        max_bytes = int(args.hash_max_mb * 1024 * 1024)
        if max_bytes > 0 and tensor_bytes(src) <= max_bytes:
            if hash_tensor(source, src) != hash_tensor(quantized, out):
                errors.append(f"preserved tensor data changed: {name}")
            hashed += 1

    qconfig = out_config.get("quantization_config")
    if qconfig is None:
        errors.append("output config.json has no quantization_config")
    if out_config.get("moe_router_dtype") != "float32":
        errors.append(
            "output config must set moe_router_dtype=float32 for GLM-5.2"
        )
    if not fp8_names:
        errors.append("no FP8 tensors were found in the output checkpoint")

    print("=== FP8 checkpoint verification ===")
    print(f"source             : {source}")
    print(f"quantized          : {quantized}")
    print(f"profile            : {args.profile}")
    print(f"output qconfig     : {json.dumps(qconfig, ensure_ascii=False)[:800]}")
    print(f"router compute     : {out_config.get('moe_router_dtype')!r}")
    print(f"output tensors     : {len(out_meta)}")
    print(f"FP8 tensors        : {len(fp8_names)}")
    print(f"preserved checked  : {preserved_checked}")
    print(f"preserved hashed   : {hashed}")
    print("output dtype counts:")
    for dtype, count in dtype_count.most_common():
        print(f"  {dtype:10s} {count:8d}")

    if errors:
        print("\nVerification FAILED:")
        for error in errors[:100]:
            print(f"  - {error}")
        if len(errors) > 100:
            print(f"  ... and {len(errors) - 100} more")
        raise SystemExit(2)

    print("\nVerification passed.")


if __name__ == "__main__":
    main()
