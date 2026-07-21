## PGD Experiments

This directory contains the public entry point for the **PGD** attack — a
port of HarmBench's white-box image-perturbation baselines (`PGD`,
`PGDBlankImage`, `PGDPatch`). The attack perturbs an image so a
white-box vision-language model (VLM) begins its reply to a benign carrier behavior
with a target affirmative string (e.g. `"Sure, here is …"`). The output is a JSONL
manifest plus cached PNGs, consumed downstream by a `VisualPromptInjection` scenario.

> **Experimental.** APIs may change in any release without a deprecation cycle.

### Public API

The primary entry point is `PGD.execute_async` (`PGD` is an alias
for `PGDGenerator`):

```python
import asyncio

from pyrit.executor.promptgen.pgd import PGD, PGDModelConfig

generator = PGD(
    model=PGDModelConfig(vlm_id="llava-hf/llava-1.5-7b-hf"),
)
result = asyncio.run(
    generator.execute_async(
        seed_image_path="cat.png",
        behavior="Describe how to ...",
        target_text="Sure, here is how to ...",
        behavior_id="advbench_001",
    )
)
print(result.image_path, result.final_loss, result.succeeded)
```

`PGDConfig` is composed of nested sub-configs (`PGDModelConfig`,
`PGDAlgorithmConfig`, `PGDVariantConfig`, `PGDOutputConfig`,
`PGDDataConfig`); the cheap ones are re-exported from
`pyrit.executor.promptgen.pgd`. See
`pyrit/executor/promptgen/pgd/config.py` for the full surface and defaults.

You can also pass a pre-built white-box target instead of a `model` config:

```python
from pyrit.prompt_target import HuggingFaceVisionTarget

target = HuggingFaceVisionTarget(model_id="llava-hf/llava-1.5-7b-hf")
generator = PGD(target=target)  # caller owns target lifecycle
```

### Variants

Selected via `PGDVariantConfig(kind=...)`:

| `kind` value  | Behavior                                                              | Seed image |
| ------------- | -------------------------------------------------------------------- | ---------- |
| `eps_bounded` | Perturb a real seed photo within an L-infinity ε-ball around it.     | required   |
| `blank_image` | Start from random noise, optimize with no ε bound.                   | optional   |
| `patch`       | Perturb only a random square patch (`patch_fraction`) of the image.  | required   |

ε and `step_size` are expressed in the processor's **normalized** `pixel_values`
space (the differentiable model input) — a deliberate, model-agnostic deviation from
HarmBench's `[0, 1]` pixel space. The `patch` variant requires a fixed-resolution
`[N, C, H, W]` layout (e.g. LLaVA-1.5); dynamic-tiling models (e.g. Qwen2-VL) raise.

### Running on Azure ML

`run.py` is a thin CLI wrapper around `PGD.execute_async`. It takes a
`--config` JSON (a serialized `PGDConfig`) and a `--data` JSON (a serialized
`PGDDataConfig` pointing at a behaviors CSV):

```python
config.to_json_file("inputs/config.json")
data.to_json_file("inputs/data.json")
```

```
python -m pyrit.executor.promptgen.pgd.experiments.run \
    --config inputs/config.json \
    --data inputs/data.json \
    --output-dir results/
```

Example configs are in `configs/individual_llava.json` and
`configs/individual_qwen_vl.json`. The behaviors CSV needs a `behavior` column
(aliases: `behavior_text`, `goal`); optional `target` / `target_text`,
`seed_image_path` / `image_path`, and `behavior_id` / `id` columns are used when
present.

### VRAM & runtime guidance

Rough per-behavior guidance for a 7B VLM in half precision (`num_steps=500`). Actual
numbers depend on image resolution, sequence length, and the model:

| Hardware                | Model load | Per-behavior (500 steps) | Notes                                  |
| ----------------------- | ---------- | ------------------------ | -------------------------------------- |
| A100 / H100 (80 GB)     | ~14–16 GB  | ~2–5 min                 | Comfortable headroom; batch behaviors. |
| 24 GB consumer (4090)   | ~14–16 GB  | ~5–12 min                | Fits fp16 7B; keep resolution modest.  |
| < 16 GB                 | tight      | —                        | Use `bfloat16`/`float16`; expect OOM risk on Qwen2-VL. |

Tips: lower `num_steps` for quick iteration; set `stop_loss` to enable early exit;
`blank_image` tends to converge faster than `eps_bounded`. Use the `configs/*.json`
files as starting points and override `algorithm` fields as needed.

### Reference

Adapted from HarmBench's `baselines/multimodalpgd` (MIT). "[HarmBench: A Standardized
Evaluation Framework for Automated Red Teaming and Robust
Refusal](https://arxiv.org/abs/2402.04249)" — Mantas Mazeika et al. Official repo:
https://github.com/centerforaisafety/HarmBench.
