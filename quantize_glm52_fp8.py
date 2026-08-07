#!/usr/bin/env python3
"""
Quantize a LoRA-merged GLM-5.2 BF16 checkpoint to FP8_BLOCK while preserving
sensitive modules in BF16.

Requires:
    llmcompressor >= 0.12.0
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import torch
from llmcompressor import model_free_ptq


CORE_IGNORE = [
    # Large vocabulary matrices: little compute benefit and potentially
    # sensitive to quantization.
    "model.embed_tokens",
    "lm_head",

    # IMPORTANT: this preserves the router tensors in their source storage
    # dtype (normally BF16). GLM-5.2 performs router/gate computation in FP32
    # through config.moe_router_dtype="float32"; it does not require storing
    # every router weight tensor physically as FP32.
    r"re:.*\.mlp\.gate$",
]

OFFICIAL_FP8_IGNORE = [
    # Official GLM-5.2-FP8 excludes the DSA / IndexShare projection from FP8.
    # Both names are kept for compatibility with different exporter versions.
    r"re:.*\.self_attn\.indexers_proj$",
    r"re:.*\.self_attn\.indexer\.weights_proj$",

    # Official GLM-5.2-FP8 also leaves this MTP fusion projection unconverted.
    r"re:^model\.layers\.78\.eh_proj$",
]

PROFILE_IGNORE = {
    # Minimum high-precision set. This can quantize IndexShare projections and
    # MTP eh_proj, so use it only after the balanced profile passes evaluation.
    "aggressive": [],

    # Recommended default: semantically aligned with zai-org/GLM-5.2-FP8.
    # Router tensors remain unquantized, router computation is forced to FP32,
    # IndexShare projections stay BF16, and ordinary compatible Linear/MoE
    # matrices—including the first three dense layers—are FP8_BLOCK.
    "balanced": [
        *OFFICIAL_FP8_IGNORE,
    ],

    # Accuracy-first extension of the official-aligned profile. In addition,
    # retain the first three dense decoder layers and the complete MTP layer.
    "conservative": [
        *OFFICIAL_FP8_IGNORE,
        r"re:^model\.layers\.[0-2](?:\.|$)",
        r"re:^model\.layers\.78(?:\.|$)",
    ],
}



def parse_version_tuple(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        return (0, 0, 0)
    return tuple(int(x) for x in match.groups())


def check_llmcompressor_version() -> str:
    try:
        installed = version("llmcompressor")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "llmcompressor is not installed. Install llmcompressor>=0.12.0."
        ) from exc

    if parse_version_tuple(installed) < (0, 12, 0):
        raise RuntimeError(
            f"llmcompressor {installed} is too old. "
            "Use >=0.12.0 for multi-GPU model_free_ptq support."
        )
    return installed


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_input_model(model_dir: Path) -> dict:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Input model directory does not exist: {model_dir}")

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing model.safetensors.index.json: {index_path}\n"
            "This script expects a sharded safetensors Hugging Face checkpoint."
        )

    config = load_json(config_path)
    architectures = config.get("architectures") or []
    if "GlmMoeDsaForCausalLM" not in architectures:
        raise ValueError(
            "The checkpoint does not declare GlmMoeDsaForCausalLM. "
            f"architectures={architectures}"
        )

    declared_dtype = config.get("dtype", config.get("torch_dtype"))
    if declared_dtype not in {"bfloat16", "bf16", None}:
        raise ValueError(
            "Input checkpoint is expected to be BF16 after LoRA merge, but "
            f"config declares dtype={declared_dtype!r}."
        )

    if config.get("quantization_config") is not None:
        raise ValueError(
            "Input config.json still contains quantization_config. "
            "For a genuinely merged BF16 checkpoint, remove the stale field "
            "from a copy of config.json before running this script."
        )

    index = load_json(index_path)
    weight_names = list((index.get("weight_map") or {}).keys())
    if not weight_names:
        raise ValueError(f"No weights found in {index_path}")

    lora_names = [
        name for name in weight_names
        if "lora_" in name.lower() or ".adapter" in name.lower()
    ]
    if lora_names:
        preview = "\n  ".join(lora_names[:10])
        raise ValueError(
            "LoRA/adapter tensors are still present; the merge appears incomplete:\n"
            f"  {preview}"
        )

    if not any(name.startswith("model.layers.78.") for name in weight_names):
        print(
            "WARNING: model.layers.78 was not found. The checkpoint may omit the "
            "MTP layer; the conservative profile will still work."
        )

    router_dtype = config.get("moe_router_dtype")
    if router_dtype != "float32":
        print(
            "WARNING: input config does not declare "
            f"moe_router_dtype=float32 (found {router_dtype!r}). "
            "The exported config will be corrected."
        )

    return config


def enforce_router_compute_dtype(output_dir: Path) -> None:
    """Keep official GLM-5.2 router semantics in the exported config.

    The gate/router tensors remain in their original storage dtype (normally
    BF16), while inference frameworks use FP32 for router logits/computation.
    """
    config_path = output_dir / "config.json"
    config = load_json(config_path)
    previous = config.get("moe_router_dtype")
    config["moe_router_dtype"] = "float32"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "router config : moe_router_dtype=float32 "
        f"(previous={previous!r}; router tensors keep source storage dtype)"
    )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        has_files = any(output_dir.iterdir()) if output_dir.is_dir() else True
        if has_files and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}\n"
                "Use --overwrite-output only when you are certain it is safe."
            )
        if has_files and overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_devices(value: str | None) -> list[str] | None:
    if value is None or value.strip().lower() == "auto":
        return None

    devices: list[str] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        devices.append(raw if ":" in raw else f"cuda:{raw}")

    if not devices:
        raise ValueError("--devices did not contain any valid device IDs")
    return devices


def effective_device_count(devices: list[str] | None) -> int:
    if devices is not None:
        return len(devices)
    count = torch.cuda.device_count()
    return max(count, 1)


def build_ignore(profile: str, extra_ignore: Iterable[str]) -> list[str]:
    values = [*CORE_IGNORE, *PROFILE_IGNORE[profile], *extra_ignore]
    # Preserve ordering while removing duplicates.
    return list(dict.fromkeys(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize merged GLM-5.2 BF16 weights to FP8_BLOCK."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_IGNORE),
        default="balanced",
        help="BF16 retention profile; balanced is recommended.",
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help=(
            "Comma-separated visible device IDs, e.g. 0,1,2,3, or 'auto'. "
            "With llmcompressor>=0.12.0, auto uses all visible GPUs."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Concurrent shard workers. Default: one worker per selected GPU. "
            "Lower this when shared-storage I/O is the bottleneck."
        ),
    )
    parser.add_argument(
        "--extra-ignore",
        action="append",
        default=[],
        help="Additional exact module name or re: regex; may be repeated.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete a non-empty output directory before quantization.",
    )
    args = parser.parse_args()

    installed = check_llmcompressor_version()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different")

    config = validate_input_model(input_dir)
    prepare_output_dir(output_dir, args.overwrite_output)

    devices = parse_devices(args.devices)
    num_devices = effective_device_count(devices)
    max_workers = args.max_workers or num_devices
    if max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if max_workers > num_devices:
        print(
            f"WARNING: max_workers={max_workers} exceeds selected devices="
            f"{num_devices}. This can increase peak GPU memory."
        )

    ignore = build_ignore(args.profile, args.extra_ignore)

    print("=== GLM-5.2 FP8_BLOCK quantization ===")
    print(f"llmcompressor : {installed}")
    print(f"input         : {input_dir}")
    print(f"output        : {output_dir}")
    print(f"profile       : {args.profile}")
    print(f"devices       : {devices if devices is not None else 'auto'}")
    print(f"max_workers   : {max_workers}")
    print(f"model dtype   : {config.get('dtype', config.get('torch_dtype'))}")
    print("router storage: source dtype (normally BF16)")
    print("router compute: FP32 via moe_router_dtype=float32")
    print("BF16 ignore rules:")
    for item in ignore:
        print(f"  - {item}")

    model_free_ptq(
        model_stub=str(input_dir),
        save_directory=str(output_dir),
        scheme="FP8_BLOCK",
        ignore=ignore,
        max_workers=max_workers,
        device=devices,
    )

    enforce_router_compute_dtype(output_dir)

    print("\nQuantization completed.")
    print(f"Output checkpoint: {output_dir}")


if __name__ == "__main__":
    main()
