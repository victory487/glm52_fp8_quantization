# GLM-5.2 LoRA-Merged BF16 → FP8_BLOCK

## Recommended policy

Default profile: `balanced`.

`balanced` is now aligned with the official `zai-org/GLM-5.2-FP8` mixed-precision
policy at the module-category level:

- FP8 E4M3 weights with `128 × 128` block scales
- dynamic FP8 activations at inference
- router/gate tensors are **not quantized** and retain their source storage dtype
  (normally BF16)
- `config.json` is forced to contain `"moe_router_dtype": "float32"`, so router
  logits and routing computation use FP32
- embeddings and LM head stay BF16
- DSA / IndexShare `indexers_proj` stays BF16
- MTP `model.layers.78.eh_proj` stays BF16
- norm modules stay BF16 because `model_free_ptq` ignores non-quantizable norms
- compatible attention, dense MLP, shared-expert and routed-expert matrices are
  quantized to FP8_BLOCK, including the first three dense decoder layers

### Router FP32 does not mean FP32 checkpoint weights

The official GLM-5.2 BF16 and FP8 configs both declare:

```json
"dtype": "bfloat16",
"moe_router_dtype": "float32"
```

The router weight tensors normally remain BF16 on disk. The runtime casts or
computes router logits in FP32 according to `moe_router_dtype`. Storing every
router tensor as FP32 would increase memory without matching the official
checkpoint layout.

## Profiles

### balanced — recommended

Official-aligned high-precision exclusions:

- `model.embed_tokens`
- `lm_head`
- every `mlp.gate` router and its correction bias
- every `self_attn.indexers_proj` (or compatible `indexer.weights_proj` name)
- `model.layers.78.eh_proj`
- all norm tensors

The first three dense layers are otherwise quantized, matching the official FP8
checkpoint rather than the previous extra-conservative script version.

### conservative

Everything in `balanced`, plus:

- complete decoder layers 0–2 in BF16
- complete MTP layer 78 in BF16

Use this only when the official-aligned profile shows a measurable regression or
when MTP speculative-decoding acceptance rate is important.

### aggressive

Keeps only embedding, LM head and router/gate modules high precision. It may
quantize IndexShare and MTP-sensitive projections, so it is not the default.

## Environment

```bash
uv venv .venv-fp8 --python 3.12
source .venv-fp8/bin/activate
uv pip install "llmcompressor>=0.12.0,<0.13" safetensors
```

## Run

```bash
export INPUT_DIR=/cpfs01/models/GLM-5.2-LoRA-Merged-BF16
export OUTPUT_DIR=/cpfs01/models/GLM-5.2-LoRA-Merged-FP8-BLOCK
export PROFILE=balanced
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash run_quantization.sh
```

For slower shared storage:

```bash
python quantize_glm52_fp8.py \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --profile balanced \
  --devices auto \
  --max-workers 4
```

The exporter always writes `moe_router_dtype=float32` into the output config.
It does not rewrite router tensor bytes to FP32.

## Verify

```bash
python verify_glm52_fp8.py \
  --source "$INPUT_DIR" \
  --quantized "$OUTPUT_DIR" \
  --profile balanced \
  --hash-max-mb 64
```

The verifier checks that:

- output contains an FP8 quantization config
- some tensors are actually FP8
- router, embedding, LM head, IndexShare projection and MTP `eh_proj` remain
  unquantized under `balanced`
- output config contains `moe_router_dtype=float32`
- retained tensor shapes/dtypes match the BF16 source

## Evaluation order

1. Compare BF16 and FP8 with fixed deterministic prompts.
2. Run coding/tool-call regression and JSON validity checks.
3. Run SciCode, SWE-Bench and Terminal-Bench subsets.
4. Check 32K, 128K and production context lengths.
5. Evaluate FP8 KV cache separately; do not enable it in the first weight-quality
   comparison.
