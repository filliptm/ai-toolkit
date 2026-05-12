# Supported Models

This document lists all model architectures supported by AI-Toolkit, along with their specific requirements and recommended settings.

## Image Models

### FLUX.1 (by Black Forest Labs)

| Model | HuggingFace ID | VRAM | Notes |
|-------|----------------|------|-------|
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | ~24GB (quantized) | Best quality |
| FLUX.1-schnell | `black-forest-labs/FLUX.1-schnell` | ~24GB (quantized) | Faster, requires training adapter |
| FLUX.2-dev | `black-forest-labs/FLUX.2-dev` | ~24GB | Newer version |
| FLUX.2-klein-4B | `black-forest-labs/FLUX.2-klein-base-4B` | ~16GB | Smaller variant |
| FLUX.2-klein-9B | `black-forest-labs/FLUX.2-klein-base-9B` | ~20GB | Medium variant |

**Configuration:**
```yaml
model:
  name_or_path: "black-forest-labs/FLUX.1-dev"
  arch: "flux"
  quantize: true
  quantize_te: true
train:
  noise_scheduler: "flowmatch"
  dtype: bf16
  timestep_type: "sigmoid"
sample:
  sampler: "flowmatch"
  guidance_scale: 4
```

**Special Notes:**
- FLUX.1-schnell requires `assistant_lora_path` for training
- Does not support conv layers in LoRA
- Guidance scale typically 3.5-4.5

---

### Flex.1 / Flex.2 (by Ostris)

| Model | HuggingFace ID | VRAM | Notes |
|-------|----------------|------|-------|
| Flex.1-alpha | `ostris/Flex.1-alpha` | ~24GB (quantized) | Guidance-free variant |
| Flex.2-preview | `ostris/Flex.2-preview` | ~24GB (quantized) | With controls |

**Configuration:**
```yaml
model:
  name_or_path: "ostris/Flex.1-alpha"
  arch: "flex1"
  quantize: true
train:
  bypass_guidance_embedding: true
  noise_scheduler: "flowmatch"
```

**Flex.2 Controls:**
Supports depth, line, pose, and inpaint:
```yaml
model:
  name_or_path: "ostris/Flex.2-preview"
  arch: "flex2"
  model_kwargs:
    invert_inpaint_mask_chance: 0.2
    inpaint_dropout: 0.5
    control_dropout: 0.5
```

---

### Stable Diffusion XL

| Model | HuggingFace ID | VRAM | Notes |
|-------|----------------|------|-------|
| SDXL Base | `stabilityai/stable-diffusion-xl-base-1.0` | ~16GB | Standard SDXL |
| SDXL Turbo | `stabilityai/sdxl-turbo` | ~16GB | Distilled |

**Configuration:**
```yaml
model:
  name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
  arch: "sdxl"
train:
  noise_scheduler: "ddpm"
  dtype: fp16
sample:
  sampler: "ddpm"
  guidance_scale: 7
```

**Special Notes:**
- Supports conv layers in LoRA
- Use DDPM noise scheduler (not flowmatch)

---

### Stable Diffusion 1.5

| Model | HuggingFace ID | VRAM | Notes |
|-------|----------------|------|-------|
| SD 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | ~8GB | Classic SD |

**Configuration:**
```yaml
model:
  name_or_path: "stable-diffusion-v1-5/stable-diffusion-v1-5"
  arch: "sd1"
train:
  noise_scheduler: "ddpm"
sample:
  width: 512
  height: 512
  guidance_scale: 7
```

---

### Lumina2

| Model | HuggingFace ID | VRAM | Notes |
|-------|----------------|------|-------|
| Lumina Image 2.0 | `Alpha-VLLM/Lumina-Image-2.0` | ~24GB | |

**Configuration:**
```yaml
model:
  name_or_path: "Alpha-VLLM/Lumina-Image-2.0"
  arch: "lumina2"
  quantize_te: true
train:
  noise_scheduler: "flowmatch"
```

---

### HiDream

| Model | HuggingFace ID | VRAM | Notes |
|-------|----------------|------|-------|
| HiDream I1 Full | `HiDream-ai/HiDream-I1-Full` | ~24GB | MoE architecture |
| HiDream E1 | `HiDream-ai/HiDream-E1-1` | ~24GB | Edit model |

**Configuration:**
```yaml
model:
  name_or_path: "HiDream-ai/HiDream-I1-Full"
  arch: "hidream"
  quantize: true
train:
  lr: 0.0002
  timestep_type: "shift"
network:
  network_kwargs:
    ignore_if_contains:
      - "ff_i.experts"
      - "ff_i.gate"
```

**Special Notes:**
- Has MoE (Mixture of Experts) layers
- Ignore expert layers for faster training
- Higher learning rate recommended

---

### Qwen-Image / Qwen-Image-Edit

| Model | HuggingFace ID | VRAM | Notes |
|-------|----------------|------|-------|
| Qwen-Image | `Qwen/Qwen-Image` | ~24GB | Text-to-image |
| Qwen-Image-2512 | `Qwen/Qwen-Image-2512` | ~24GB | Higher res |
| Qwen-Image-Edit | `Qwen/Qwen-Image-Edit` | ~24GB | Image editing |
| Qwen-Image-Edit-2509 | `Qwen/Qwen-Image-Edit-2509` | ~24GB | Updated edit |
| Qwen-Image-Edit-2511 | `Qwen/Qwen-Image-Edit-2511` | ~24GB | Latest edit |

**Configuration:**
```yaml
model:
  name_or_path: "Qwen/Qwen-Image"
  arch: "qwen_image"
  quantize: true
  quantize_te: true
  low_vram: true
train:
  timestep_type: "weighted"
```

**Available ARAs (Accuracy Recovery Adapters):**
- 3-bit with ARA for most variants

---

### Z-Image (by Tongyi-MAI)

| Model | HuggingFace ID | VRAM | Notes |
|-------|----------------|------|-------|
| Z-Image | `Tongyi-MAI/Z-Image` | ~24GB | Full model |
| Z-Image Turbo | `Tongyi-MAI/Z-Image-Turbo` | ~24GB | Distilled |
| Z-Image De-Turbo | `ostris/Z-Image-De-Turbo` | ~24GB | De-distilled |

**Z-Image Turbo (with training adapter):**
```yaml
model:
  name_or_path: "Tongyi-MAI/Z-Image-Turbo"
  assistant_lora_path: "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors"
sample:
  guidance_scale: 1
  sample_steps: 8
```

---

### Chroma

| Model | HuggingFace ID | VRAM | Notes |
|-------|----------------|------|-------|
| Chroma1-Base | `lodestones/Chroma1-Base` | ~24GB | |

**Configuration:**
```yaml
model:
  name_or_path: "lodestones/Chroma1-Base"
  arch: "chroma"
  quantize: true
```

---

### OmniGen2

| Model | HuggingFace ID | VRAM | Notes |
|-------|----------------|------|-------|
| OmniGen2 | `OmniGen2/OmniGen2` | ~24GB | Multi-modal |

**Configuration:**
```yaml
model:
  name_or_path: "OmniGen2/OmniGen2"
  arch: "omnigen2"
  quantize_te: true
```

---

## Video Models

### Wan 2.1 / Wan 2.2

| Model | HuggingFace ID | VRAM | Notes |
|-------|----------------|------|-------|
| Wan 2.1 T2V 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | ~16GB | Smaller |
| Wan 2.1 T2V 14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | ~24GB | Full T2V |
| Wan 2.1 I2V 14B 480P | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | ~24GB | I2V 480p |
| Wan 2.1 I2V 14B 720P | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | ~24GB | I2V 720p |
| Wan 2.2 T2V 14B | `ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16` | ~24GB | Updated |
| Wan 2.2 I2V 14B | `ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16` | ~24GB | Updated I2V |
| Wan 2.2 TI2V 5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | ~20GB | Smaller TI2V |

**Configuration (T2V):**
```yaml
model:
  name_or_path: "Wan-AI/Wan2.1-T2V-14B-Diffusers"
  arch: "wan21"
  quantize: true
  quantize_te: true
  low_vram: true
train:
  unload_text_encoder: true
datasets:
  - folder_path: "/path/to/videos"
    num_frames: 41
sample:
  num_frames: 40
  fps: 15
```

**Configuration (I2V):**
```yaml
model:
  name_or_path: "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
  arch: "wan21_i2v"
train:
  timestep_type: "weighted"
datasets:
  - folder_path: "/path/to/videos"
    do_i2v: true
```

**Wan 2.2 Multistage:**
```yaml
model:
  model_kwargs:
    train_high_noise: true
    train_low_noise: true
train:
  timestep_type: "linear"
```

---

### LTX-2

| Model | HuggingFace ID | VRAM | Notes |
|-------|----------------|------|-------|
| LTX-2 | `Lightricks/LTX-2` | ~24GB | Fast video |

**Configuration:**
```yaml
model:
  name_or_path: "Lightricks/LTX-2"
  arch: "ltx2"
  quantize: true
  low_vram: true
datasets:
  - folder_path: "/path/to/videos"
    do_audio: true      # Supports audio
    do_i2v: false
    fps: 24
sample:
  num_frames: 121
  fps: 24
  width: 768
  height: 768
```

**Special Notes:**
- Supports audio conditioning
- Fast inference

---

## Instruction-Following Models

### FLUX.1-Kontext

| Model | HuggingFace ID | VRAM | Notes |
|-------|----------------|------|-------|
| FLUX.1-Kontext-dev | `black-forest-labs/FLUX.1-Kontext-dev` | ~24GB | Image editing |

**Configuration:**
```yaml
model:
  name_or_path: "black-forest-labs/FLUX.1-Kontext-dev"
  arch: "flux_kontext"
  quantize: true
train:
  timestep_type: "weighted"
datasets:
  - folder_path: "/path/to/pairs"
    control_path: "/path/to/source_images"
```

---

## Quantization Options

All models support various quantization levels:

| Option | Size Reduction | Quality Impact |
|--------|---------------|----------------|
| `qfloat8` | ~50% | Minimal |
| `uint7` | ~56% | Low |
| `uint6` | ~63% | Low-Medium |
| `uint5` | ~69% | Medium |
| `uint4` | ~75% | Medium-High |
| `uint3` | ~81% | High |

**With Accuracy Recovery Adapter (ARA):**
```yaml
model:
  qtype: "uint4|ostris/accuracy_recovery_adapters/model_ara.safetensors"
```

---

## Memory Requirements Summary

| Model Category | Min VRAM (Quantized) | Recommended |
|---------------|---------------------|-------------|
| SD 1.5 | 6GB | 8GB |
| SDXL | 12GB | 16GB |
| FLUX.1 | 18GB | 24GB |
| Wan 14B | 20GB | 24GB+ |
| Video models | 20GB | 24GB+ |

**Memory reduction strategies:**
- `quantize: true` - ~50% reduction
- `low_vram: true` - Peak VRAM reduction
- `layer_offloading: true` - Dynamic offloading
- `cache_latents_to_disk: true` - VAE not needed during training
- `unload_text_encoder: true` - For 24GB with large models
