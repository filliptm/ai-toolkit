# Supported Models

This document is aligned with the current model registry in `extensions_built_in/*/AI_TOOLKIT_MODELS` and the UI model selector in `ui/src/app/jobs/new/options.ts`.

## Architecture Identifiers

Use these values in `config.process[].model.arch`. The UI may append a suffix such as `:2511` or `:14b-outpaint` for presets; `ModelConfig` strips the suffix before loading the backend model.

| Arch | UI presets / labels | Backend |
|------|---------------------|---------|
| `flux` | FLUX.1, Flex.1 | Built-in `StableDiffusion` FLUX path |
| `flex2` | Flex.2 | `extensions_built_in/flex2/Flex2` |
| `flux_kontext` | FLUX.1-Kontext-dev | `FluxKontextModel` |
| `flux2` | FLUX.2 | `Flux2Model` |
| `flux2_klein_4b` | FLUX.2-klein-base-4B | `Flux2Klein4BModel` |
| `flux2_klein_9b` | FLUX.2-klein-base-9B | `Flux2Klein9BModel` |
| `sd15` / `sd1` | SD 1.5 | Built-in SD 1.x path |
| `sdxl` | SDXL | Built-in SDXL path |
| `lumina2` | Lumina2 | Built-in Lumina2 path |
| `chroma` | Chroma | `ChromaModel` |
| `chroma_radiance` | Chroma Radiance | `ChromaRadianceModel` |
| `zeta_chroma` | Zeta Chroma | `ZetaChromaModel` |
| `qwen_image` | Qwen-Image, Qwen-Image-2512 | `QwenImageModel` |
| `qwen_image_edit` | Qwen-Image-Edit | `QwenImageEditModel` |
| `qwen_image_edit_plus` | Qwen-Image-Edit-2509/2511 | `QwenImageEditPlusModel` |
| `hidream` | HiDream | `HidreamModel` |
| `hidream_e1` | HiDream E1 | `HidreamE1Model` |
| `hidream_o1` | HiDream-O1 | `HidreamO1Model` |
| `omnigen2` | OmniGen2 | `OmniGen2Model` |
| `zimage` | Z-Image, Z-Image Turbo, Z-Image De-Turbo, Z-Image Turbo Outpaint ControlNet | `ZImageModel` |
| `nucleus_image` | Nucleus-Image | `NucleusImageModel` |
| `ernie_image` | ERNIE-Image | `ErnieImageModel` |
| `f-lite` | F-Lite | `FLiteModel` |
| `cogview4` | CogView4 | `toolkit.models.cogview4.CogView4` |
| `wan21` | Wan 2.1 T2V 1.3B/14B | `Wan21` |
| `wan21_i2v` | Wan 2.1 I2V 14B 480P/720P | `Wan21I2V` |
| `wan21_vace` | Wan 2.1 VACE 1.3B/14B, VACE Outpaint | `WanVACEModel` |
| `wan22_14b` | Wan 2.2 T2V 14B | `Wan2214bModel` |
| `wan22_14b_i2v` | Wan 2.2 I2V 14B | `Wan2214bI2VModel` |
| `wan22_5b` | Wan 2.2 TI2V 5B | `Wan225bModel` |
| `ltx2` | LTX-2 | `LTX2Model` |
| `ltx2.3` | LTX-2.3, LTX-2.3 IC / V2V, LTX-2.3 IC / Image Edit | `LTX23Model` |
| `ace_step_15` | ACE-Step 1.5 | `AceStep15Model` |
| `ace_step_15_xl` | ACE-Step 1.5 XL | `AceStep15XLModel` |

Older built-in Stable Diffusion paths also still support legacy arches such as `sd2`, `sd3`, `ssd`, `vega`, `pixart`, `pixart_sigma`, and `auraflow` when corresponding model configs are supplied.

## UI Preset Aliases

The UI uses these suffixed names for presets that share a backend arch:

| UI preset name | Backend arch after `ModelConfig` normalization |
|----------------|-----------------------------------------------|
| `qwen_image:2512` | `qwen_image` |
| `qwen_image_edit_plus:2511` | `qwen_image_edit_plus` |
| `wan21:1b` | `wan21` |
| `wan21:14b` | `wan21` |
| `wan21_i2v:14b480p` | `wan21_i2v` |
| `wan21_i2v:14b` | `wan21_i2v` |
| `wan21_vace:1.3b` | `wan21_vace` |
| `wan21_vace:14b` | `wan21_vace` |
| `wan21_vace:14b-outpaint` | `wan21_vace` |
| `wan22_14b:t2v` | `wan22_14b` |
| `zimage:turbo` | `zimage` |
| `zimage:turbo-outpaint-controlnet` | `zimage` |
| `zimage:deturbo` | `zimage` |
| `ltx2.3:ic_v2v` | `ltx2.3` |
| `ltx2.3:ic_image_edit` | `ltx2.3` |

## Image Models

| Model | Default path in UI/config examples | Notes |
|-------|------------------------------------|-------|
| FLUX.1 | `black-forest-labs/FLUX.1-dev` | Flow matching, transformer-only LoRA by default |
| FLUX.1-schnell | `black-forest-labs/FLUX.1-schnell` | Distilled; examples use an assistant/training adapter |
| FLUX.1-Kontext-dev | `black-forest-labs/FLUX.1-Kontext-dev` | Instruction/editing preset with control image support |
| FLUX.2 | `black-forest-labs/FLUX.2-dev` | Uses `match_target_res` model kwargs in UI presets |
| FLUX.2-klein-base-4B | `black-forest-labs/FLUX.2-klein-base-4B` | Smaller FLUX.2 preset |
| FLUX.2-klein-base-9B | `black-forest-labs/FLUX.2-klein-base-9B` | Larger klein preset |
| Flex.1 | `ostris/Flex.1-alpha` | Uses `arch: flex1`, migrated internally to `flux` |
| Flex.2 | `ostris/Flex.2-preview` | Depth, line, pose, and inpaint controls |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | DDPM-style scheduler presets |
| SD 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | 512px baseline |
| Lumina2 | `Alpha-VLLM/Lumina-Image-2.0` | Flow matching |
| Chroma | `lodestones/Chroma1-Base` | Flow matching |
| Zeta Chroma | `lodestones/Zeta-Chroma/zeta-chroma-base-x0-pixel-dino-distance.safetensors` | Uses `extras_name_or_path: Tongyi-MAI/Z-Image-Turbo` |
| OmniGen2 | `OmniGen2/OmniGen2` | Multimodal image model |
| Nucleus-Image | `NucleusAI/Nucleus-Image` | MoE-like layer exclusions in UI defaults |
| ERNIE-Image | `baidu/ERNIE-Image` | Registered custom model |
| F-Lite | model path supplied by config | Registered custom model |
| CogView4 | model path supplied by config | Core model class exists outside the UI preset list |

## Instruction And Edit Models

| Model | Default path | Notes |
|-------|--------------|-------|
| Qwen-Image | `Qwen/Qwen-Image` | T2I preset |
| Qwen-Image-2512 | `Qwen/Qwen-Image-2512` | Same backend arch with UI suffix |
| Qwen-Image-Edit | `Qwen/Qwen-Image-Edit` | Single control image edit preset |
| Qwen-Image-Edit-2509 | `Qwen/Qwen-Image-Edit-2509` | Multi-control edit preset |
| Qwen-Image-Edit-2511 | `Qwen/Qwen-Image-Edit-2511` | Multi-control edit preset |
| HiDream | `HiDream-ai/HiDream-I1-Full` | UI excludes expert/gate layers by default |
| HiDream E1 | `HiDream-ai/HiDream-E1-1` | Edit/instruction preset |
| HiDream-O1 | `HiDream-ai/HiDream-O1-Image` | Uses Qwen3-VL components and large sample defaults |
| Wan 2.1 VACE | `Wan-AI/Wan2.1-VACE-1.3B-diffusers` or `Wan-AI/Wan2.1-VACE-14B-diffusers` | Edit, reference, video, and outpaint conditioning |

## Video Models

| Model | Default path | Notes |
|-------|--------------|-------|
| Wan 2.1 T2V 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | T2V preset |
| Wan 2.1 T2V 14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | T2V preset |
| Wan 2.1 I2V 14B 480P | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | I2V preset |
| Wan 2.1 I2V 14B 720P | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | I2V preset |
| Wan 2.2 T2V 14B | `ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16` | Multistage high/low-noise training flags |
| Wan 2.2 I2V 14B | `ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16` | I2V multistage preset |
| Wan 2.2 TI2V 5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | TI2V, 121-frame UI default |
| LTX-2 | `Lightricks/LTX-2` | Video and audio-aware dataset options |
| LTX-2.3 | `Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors` | Video/audio preset |
| LTX-2.3 IC / V2V | `Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors` | Reference-frame IC/V2V LoRA preset |
| LTX-2.3 IC / Image Edit | `Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors` | Image edit preset using the LTX-2.3 backend |

## Audio Models

| Model | Default path | Notes |
|-------|--------------|-------|
| ACE-Step 1.5 | `ostris/ace_step_1.5_ComfyUI_files/ace_step_1.5_base_aio.safetensors` | Audio sample fields include caption, lyrics, BPM, key, time signature, duration, and language |
| ACE-Step 1.5 XL | `ostris/ace_step_1.5_ComfyUI_files/ace_step_1.5_xl_base_aio.safetensors` | Larger ACE-Step preset |

## Common Model Settings

Most current image/video/audio UI presets use:

```yaml
model:
  quantize: true
  quantize_te: true
  qtype: qfloat8
  qtype_te: qfloat8
  low_vram: false
train:
  noise_scheduler: flowmatch
  dtype: bf16
sample:
  sampler: flowmatch
```

Quantization options exposed by the UI are `qfloat8`, `uint7`, `uint6`, `uint5`, `uint4`, `uint3`, and `uint2`. An Accuracy Recovery Adapter can be embedded in `qtype` as `uint4|org/repo/path.safetensors`; `ModelConfig` splits that into `qtype` and `accuracy_recovery_adapter`.

Memory reduction options are `quantize`, `quantize_te`, `low_vram`, `layer_offloading`, `layer_offloading_transformer_percent`, `layer_offloading_text_encoder_percent`, `cache_latents_to_disk`, and `unload_text_encoder`.
