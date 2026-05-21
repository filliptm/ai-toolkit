# Configuration Guide

This document provides a complete reference for all configuration options in AI-Toolkit.

## Configuration File Structure

Configuration files can be YAML or JSON. The basic structure is:

```yaml
job: extension           # Job type (usually 'extension' for training)
config:
  name: "my_lora_v1"     # Output name
  process:               # List of processes to run
    - type: 'diffusion_trainer' # Process type (extension uid)
      # ... process configuration
meta:                    # Optional metadata
  name: "[name]"         # [name] is replaced with config.name
  version: '1.0'
```

## Environment Variables

You can use environment variables in configs:

```yaml
model:
  name_or_path: "${HF_MODEL_PATH}"  # Replaced with env var value
```

## Complete Configuration Reference

### Top-Level Keys

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `job` | string | Yes | Job type: `extension`, `train`, `extract`, `mod`, `generate` |
| `config` | object | Yes | Main configuration object |
| `config.name` | string | Yes | Name for outputs and checkpoints |
| `config.process` | array | Yes | List of process configurations |
| `meta` | object | No | Additional metadata saved with model |

---

## Process Configuration

### General Process Settings

```yaml
process:
  - type: 'diffusion_trainer'       # Extension UID
    training_folder: "output"       # Root output folder
    device: "cuda"                  # Device; UI sets CUDA_VISIBLE_DEVICES for selected GPUs
    trigger_word: "sks"             # Trigger word for training
    performance_log_every: 1000     # Log performance stats every N steps
    sqlite_db_path: "./aitk_db.db"  # Used by diffusion_trainer/UI logging
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | Required | Extension UID (`diffusion_trainer`, `sd_trainer`, `concept_slider`, etc.) |
| `training_folder` | string | Required | Root folder for outputs |
| `device` | string | `"cuda"` | Device to use |
| `trigger_word` | string | None | Trigger word to inject into captions |
| `performance_log_every` | int | 0 | Log performance every N steps (0 = disabled) |
| `sqlite_db_path` | string | None | SQLite DB used by UI-aware processes |

---

## Model Configuration

```yaml
model:
  name_or_path: "black-forest-labs/FLUX.1-dev"
  arch: "flux"                      # Architecture identifier
  is_flux: true                     # Legacy architecture flag
  quantize: true                    # Enable 8-bit quantization
  quantize_te: true                 # Quantize text encoder
  qtype: "qfloat8"                  # Quantization type
  qtype_te: "qfloat8"               # Text encoder quantization type
  low_vram: false                   # Low VRAM mode
  layer_offloading: false           # Offload layers to CPU
  layer_offloading_transformer_percent: 1.0
  layer_offloading_text_encoder_percent: 1.0
  vae_path: null                    # Custom VAE path
  extras_name_or_path: null         # Shared extras source for model variants
  assistant_lora_path: null         # Training adapter for distilled models
  accuracy_recovery_adapter: null   # Optional ARA path; can also be encoded in qtype
  model_kwargs: {}                  # Model-specific options
```

### Model Configuration Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name_or_path` | string | Required | HuggingFace model ID or local path |
| `arch` | string | Auto | Architecture identifier; see [Supported Models](./models.md) |
| `is_flux` | bool | false | Legacy flag for FLUX models |
| `is_xl` | bool | false | Legacy flag for SDXL models |
| `is_v2` | bool | false | Legacy flag for SD 2.x models |
| `quantize` | bool | false | Quantize transformer/UNet to 8-bit |
| `quantize_te` | bool | false | Quantize text encoder to 8-bit |
| `qtype` | string | `"qfloat8"` | Quantization type: `qfloat8`, `uint2`-`uint7`; `int8`/`float8` are also used internally in some paths |
| `qtype_te` | string | `"qfloat8"` | Text encoder quantization type |
| `low_vram` | bool | false | Quantize on CPU (slower, less VRAM) |
| `layer_offloading` | bool | false | Offload layers to CPU during training |
| `layer_offloading_transformer_percent` | float | 1.0 | Fraction of transformer layers eligible for offload |
| `layer_offloading_text_encoder_percent` | float | 1.0 | Fraction of text encoder layers eligible for offload |
| `vae_path` | string | null | Custom VAE model path |
| `te_name_or_path` | string | null | Custom text encoder path |
| `extras_name_or_path` | string | `name_or_path` | Alternate source for shared model extras such as VAE/text encoder files |
| `assistant_lora_path` | string | null | Training adapter for distilled models |
| `accuracy_recovery_adapter` | string | null | Adapter for quantization accuracy recovery |
| `compile` | bool | false | Enable `torch.compile`; disabled automatically if quantization is enabled |
| `model_kwargs` | object | `{}` | Model-specific backend options |
| `model_paths` | object | `{}` | Model-specific local component paths |

### Architecture Identifiers

| Arch | Models |
|------|--------|
| `sd1` | Stable Diffusion 1.x |
| `sd2` | Stable Diffusion 2.x |
| `sdxl` | Stable Diffusion XL |
| `sd3` | Stable Diffusion 3 |
| `flux` | FLUX.1, Flex.1 |
| `flex2` | Flex.2 |
| `flux_kontext` | FLUX.1-Kontext |
| `flux2`, `flux2_klein_4b`, `flux2_klein_9b` | FLUX.2 family |
| `wan21`, `wan21_i2v`, `wan21_vace` | Wan 2.1 T2V/I2V/VACE |
| `wan22_14b`, `wan22_14b_i2v`, `wan22_5b` | Wan 2.2 |
| `ltx2`, `ltx2.3` | LTX video models |
| `lumina2` | Lumina Image 2.0 |
| `qwen_image`, `qwen_image_edit`, `qwen_image_edit_plus` | Qwen-Image family |
| `hidream`, `hidream_e1`, `hidream_o1` | HiDream family |
| `chroma`, `chroma_radiance`, `zeta_chroma` | Chroma family |
| `zimage` | Z-Image variants |
| `omnigen2` | OmniGen2 |
| `ernie_image` | ERNIE-Image |
| `nucleus_image` | Nucleus-Image |
| `ace_step_15`, `ace_step_15_xl` | ACE-Step audio |
| `sd1`, `sd2`, `sdxl`, `sd3`, `ssd`, `vega`, `pixart`, `pixart_sigma`, `auraflow` | Legacy/core Stable Diffusion-style architectures |

---

## Network Configuration (LoRA/LoKr)

```yaml
network:
  type: "lora"                      # Network type
  linear: 32                        # Rank for linear layers
  linear_alpha: 32                  # Alpha for linear layers
  conv: 16                          # Rank for conv layers (optional)
  conv_alpha: 16                    # Alpha for conv layers
  dropout: null                     # Dropout rate
  lokr_full_rank: false             # Full rank for LoKr
  lokr_factor: -1                   # LoKr factor (-1 = auto)
  network_kwargs:
    only_if_contains: []            # Only train layers containing these strings
    ignore_if_contains: []          # Skip layers containing these strings
```

### Network Types

| Type | Description |
|------|-------------|
| `lora` | Standard LoRA (Low-Rank Adaptation) |
| `lokr` | LoKr (Low-Rank Kronecker Product) |
| `locon` | LoCon (LoRA with Convolution) |
| `lorm` | LoRM (experimental) |

### Network Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"lora"` | Network type |
| `linear` | int | 4 | Rank for linear layers |
| `linear_alpha` | float | 1.0 | Alpha scaling for linear |
| `conv` | int | null | Rank for conv layers (null = no conv training) |
| `conv_alpha` | float | null | Alpha for conv layers |
| `dropout` | float | null | Dropout probability |
| `transformer_only` | bool | true | Only apply to transformer blocks |
| `lokr_full_rank` | bool | false | Use full rank for LoKr |
| `lokr_factor` | int | -1 | LoKr factor (-1 = auto find largest) |
| `old_lokr_format` | bool | false | Save/use the older LoKr format |
| `split_multistage_loras` | bool | true | Split LoRAs for multistage models when supported |
| `layer_offloading` | bool | false | Experimental network layer offloading |
| `pretrained_lora_path` | string | null | Initialize from existing LoRA |

### Layer Targeting

Use `network_kwargs` to target specific layers:

```yaml
network:
  type: "lora"
  linear: 128
  network_kwargs:
    # Only train these specific layers
    only_if_contains:
      - "transformer.single_transformer_blocks.7.proj_out"
      - "transformer.single_transformer_blocks.20.proj_out"

    # OR: Train all except these
    ignore_if_contains:
      - "transformer.single_transformer_blocks."
```

---

## Training Configuration

```yaml
train:
  batch_size: 1                     # Images per step
  steps: 3000                       # Total training steps
  lr: 1e-4                          # Learning rate
  optimizer: "adamw8bit"            # Optimizer type
  gradient_checkpointing: true      # Enable gradient checkpointing
  gradient_accumulation: 1          # Accumulate gradients over N steps
  train_unet: true                  # Train UNet/Transformer
  train_text_encoder: false         # Train text encoder
  noise_scheduler: "flowmatch"      # Noise scheduler type
  dtype: "bf16"                     # Training precision
  timestep_type: "sigmoid"          # Timestep sampling method
  max_grad_norm: 1.0                # Gradient clipping
```

### Training Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `batch_size` | int | 1 | Batch size per step |
| `steps` | int | 1000 | Total training steps |
| `lr` | float | 1e-6 | Base learning rate |
| `unet_lr` | float | lr | UNet-specific learning rate |
| `text_encoder_lr` | float | lr | Text encoder learning rate |
| `optimizer` | string | `"adamw"` | Optimizer type (see below) |
| `optimizer_params` | object | {} | Extra optimizer parameters |
| `lr_scheduler` | string | `"constant"` | Learning rate scheduler |
| `lr_scheduler_params` | object | {} | Scheduler parameters |
| `gradient_checkpointing` | bool | true | Trade compute for memory |
| `gradient_accumulation` | int | 1 | Accumulate over N batches |
| `train_unet` | bool | true | Train UNet/Transformer |
| `train_text_encoder` | bool | false | Train text encoders |
| `noise_scheduler` | string | `"ddpm"` | `ddpm`, `flowmatch` |
| `dtype` | string | `"fp32"` | `fp16`, `bf16`, `fp32` |
| `max_grad_norm` | float | 1.0 | Gradient clipping norm |

### Optimizers

| Optimizer | Description |
|-----------|-------------|
| `adamw` | Standard AdamW |
| `adamw8bit` | 8-bit AdamW (recommended) |
| `adafactor` | Memory-efficient Adafactor |
| `prodigy` | Prodigy (adaptive learning rate) |
| `prodigy8bit` | 8-bit Prodigy |

### Learning Rate Schedulers

| Scheduler | Description |
|-----------|-------------|
| `constant` | Constant learning rate |
| `cosine` | Cosine annealing |
| `cosine_with_restarts` | Cosine with warm restarts |
| `linear` | Linear decay |
| `polynomial` | Polynomial decay |

### Timestep Sampling Types

| Type | Description | Best For |
|------|-------------|----------|
| `sigmoid` | Sigmoid distribution (default) | General training |
| `linear` | Linear distribution | Even coverage |
| `weighted` | Weighted toward extremes | Video models, instruction models |
| `shift` | Shifted distribution | HiDream |
| `lognorm_blend` | Log-normal blend | Advanced experimentation |

### Advanced Training Options

```yaml
train:
  # SNR (Signal-to-Noise Ratio) weighting
  snr_gamma: 5.0                    # Min-SNR gamma (null = disabled)
  min_snr_gamma: null               # Alternative SNR weighting

  # Noise settings
  noise_offset: 0.0                 # Noise offset for better dark/light images
  noise_multiplier: 1.0             # Scale applied noise

  # Loss configuration
  loss_type: "mse"                  # mse, mae, wavelet
  loss_target: "noise"              # noise, source, v_prediction

  # EMA (Exponential Moving Average)
  ema_config:
    use_ema: true
    ema_decay: 0.99

  # Dropout settings
  prompt_dropout_prob: 0.0          # Caption dropout probability

  # Memory optimization
  unload_text_encoder: false        # Unload TE after encoding trigger
  cache_text_embeddings: false      # Cache all text embeddings

  # Output preservation (prevents forgetting)
  diff_output_preservation: false   # DOP training
  diff_output_preservation_multiplier: 1.0
  diff_output_preservation_class: "person"  # Class to replace trigger

  # Skip/force sampling
  skip_first_sample: false          # Skip pre-training sample
  force_first_sample: false         # Force first sample
  disable_sampling: false           # Disable all sampling
```

---

## Dataset Configuration

```yaml
datasets:
  - folder_path: "/path/to/images"  # Image folder
    caption_ext: "txt"              # Caption file extension
    caption_dropout_rate: 0.05      # Caption dropout probability
    shuffle_tokens: false           # Shuffle comma-separated tokens
    cache_latents_to_disk: true     # Cache encoded latents
    resolution: [512, 768, 1024]    # Training resolution(s)
    buckets: true                   # Enable aspect ratio bucketing
    flip_x: false                   # Random horizontal flip
    flip_y: false                   # Random vertical flip
    is_reg: false                   # Regularization dataset
    network_weight: 1.0             # Loss weight for this dataset
```

### Dataset Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `folder_path` | string | Required | Path to image folder |
| `dataset_path` | string | null | Alternative: JSON dataset file |
| `caption_ext` | string | `".txt"` | Caption file extension |
| `default_caption` | string | null | Caption if no file exists |
| `trigger_word` | string | null | Dataset-specific trigger word |
| `resolution` | int/array | 512 | Training resolution(s) |
| `buckets` | bool | true | Enable aspect ratio bucketing |
| `bucket_tolerance` | int | 64 | Bucket size tolerance |
| `scale` | float | 1.0 | Image scale factor |
| `random_scale` | bool | false | Random scaling |
| `random_crop` | bool | false | Random cropping |
| `flip_x` | bool | false | Horizontal flip augmentation |
| `flip_y` | bool | false | Vertical flip augmentation |
| `cache_latents_to_disk` | bool | false | Cache latents to disk |
| `cache_text_embeddings` | bool | false | Cache text embeddings |
| `caption_dropout_rate` | float | 0.0 | Probability to drop caption |
| `token_dropout_rate` | float | 0.0 | Probability to drop tokens |
| `shuffle_tokens` | bool | false | Shuffle tokens in caption |
| `keep_tokens` | int | 0 | Tokens to keep when shuffling |
| `is_reg` | bool | false | Regularization dataset flag |
| `network_weight` | float | 1.0 | Loss weight multiplier |

### Control Images (for ControlNet, etc.)

```yaml
datasets:
  - folder_path: "/path/to/images"
    control_path: "/path/to/control_images"  # Single control type
    # OR multiple control types:
    control_path_1: "/path/to/depth"
    control_path_2: "/path/to/edges"
    control_path_3: "/path/to/pose"
```

### Video Dataset Options

```yaml
datasets:
  - folder_path: "/path/to/videos"
    num_frames: 41                  # Frames to sample
    shrink_video_to_frames: true    # Shrink to num_frames if longer
    do_i2v: false                   # Image-to-video mode
    do_audio: false                 # Include audio embeddings
    fps: 16                         # Target framerate
```

---

## Save Configuration

```yaml
save:
  dtype: "float16"                  # Save precision
  save_every: 250                   # Save checkpoint every N steps
  max_step_saves_to_keep: 4         # Keep N recent checkpoints
  save_format: "safetensors"        # safetensors or diffusers
  push_to_hub: false                # Push to HuggingFace Hub
  hf_repo_id: "username/model"      # Hub repository ID
  hf_private: true                  # Private repository
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `dtype` | string | `"float16"` | Save precision |
| `save_every` | int | 1000 | Steps between saves |
| `max_step_saves_to_keep` | int | 5 | Maximum checkpoints to keep |
| `save_format` | string | `"safetensors"` | `safetensors` or `diffusers` |
| `push_to_hub` | bool | false | Push to HuggingFace Hub |
| `hf_repo_id` | string | null | Hub repository ID |
| `hf_private` | bool | false | Make Hub repo private |

---

## Sample Configuration

```yaml
sample:
  sampler: "flowmatch"              # Sampler type
  sample_every: 250                 # Sample every N steps
  width: 1024                       # Sample width
  height: 1024                      # Sample height
  guidance_scale: 4.0               # CFG scale
  sample_steps: 20                  # Inference steps
  seed: 42                          # Random seed
  walk_seed: true                   # Increment seed per prompt
  neg: ""                           # Negative prompt
  samples:                          # Sample prompts
    - prompt: "a photo of sks person"
      width: 1024                   # Optional per-sample override
      height: 1024
    - prompt: "another prompt"
      seed: 123                     # Optional per-sample seed
```

### Sample Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sampler` | string | `"ddpm"` | Sampler type |
| `sample_every` | int | 100 | Steps between samples |
| `width` | int | 512 | Default sample width |
| `height` | int | 512 | Default sample height |
| `guidance_scale` | float | 7.0 | CFG guidance scale |
| `sample_steps` | int | 20 | Inference steps |
| `seed` | int | 0 | Random seed |
| `walk_seed` | bool | false | Increment seed per prompt |
| `neg` | string | `""` | Negative prompt |
| `samples` | array | [] | List of sample configurations |
| `network_multiplier` | float | 1.0 | LoRA strength for sampling |
| `num_frames` | int | 1 | Video frames (for video models) |
| `fps` | int | 16 | Video framerate |

---

## Logging Configuration

```yaml
logging:
  log_every: 100                    # Log loss every N steps
  verbose: false                    # Verbose logging
  use_wandb: false                  # Use Weights & Biases
  use_ui_logger: true               # Log to UI database
  project_name: "ai-toolkit"        # W&B project name
```

---

## Adapter Configuration

For training T2I adapters, ControlNets, IP-Adapters:

```yaml
adapter:
  type: "control_net"               # t2i, ip, ip+, clip, control_net
  train: true                       # Train the adapter
  name_or_path: null                # Pretrained adapter path
  image_encoder_path: null          # CLIP image encoder path
  train_image_encoder: false        # Train image encoder
```

---

## Embedding Configuration

For textual inversion / embedding training:

```yaml
embedding:
  trigger: "custom_token"           # Embedding trigger word
  tokens: 4                         # Number of embedding tokens
  init_words: "*"                   # Initialization words
  save_format: "safetensors"        # Save format
```

---

## Example Complete Configuration

```yaml
job: extension
config:
  name: "my_flux_lora_v1"
  process:
    - type: 'diffusion_trainer'
      training_folder: "output"
      device: cuda
      trigger_word: "ohwx"

      network:
        type: "lora"
        linear: 16
        linear_alpha: 16

      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4

      datasets:
        - folder_path: "/data/my_images"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          cache_latents_to_disk: true
          resolution: [512, 768, 1024]

      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16

      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: true

      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 1024
        height: 1024
        samples:
          - prompt: "ohwx person in a coffee shop"
          - prompt: "ohwx person on a beach at sunset"
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 4
        sample_steps: 20

meta:
  name: "[name]"
  version: '1.0'
```
