# Advanced Training Techniques

This guide covers advanced training configurations and experimental techniques for power users.

## Differential Output Preservation (DOP)

DOP helps prevent catastrophic forgetting by maintaining model behavior on related concepts.

### How It Works

1. During training, the model sees your trigger (e.g., "sks person")
2. DOP also shows the model the base concept (e.g., "person" without trigger)
3. The network is trained to only activate for the trigger word

### Configuration

```yaml
train:
  diff_output_preservation: true
  diff_output_preservation_multiplier: 1.0
  diff_output_preservation_class: "person"  # Class word to preserve
```

**Requirements:**
- Must have a `trigger_word` set
- Must be using a network (LoRA/LoKr)
- Cannot train text encoder with DOP

**Recommended Settings:**
- `multiplier: 0.5-2.0` - Balance between preservation and learning
- `class: "person"` - The base concept your trigger replaces

---

## Blank Prompt Preservation

Similar to DOP but preserves behavior on empty prompts.

```yaml
train:
  blank_prompt_preservation: true
```

Useful when you want the model to maintain its unconditional generation capabilities.

---

## EMA (Exponential Moving Average)

EMA maintains a smoothed version of weights during training.

```yaml
train:
  ema_config:
    use_ema: true
    ema_decay: 0.99        # Higher = smoother, slower learning
    use_feedback: false    # Feed EMA difference back to params
    param_multiplier: 1.0  # Weight scaling (for bias-free networks)
```

**Decay Guidelines:**
- `0.99` - Standard, good for most cases
- `0.999` - Very smooth, for longer training
- `0.9` - More responsive, shorter training

---

## Timestep Sampling Strategies

Control which noise levels the model trains on.

### Sigmoid (Default)
```yaml
train:
  timestep_type: "sigmoid"
```
Standard distribution, good general purpose.

### Linear
```yaml
train:
  timestep_type: "linear"
```
Uniform distribution across all timesteps.

### Weighted
```yaml
train:
  timestep_type: "weighted"
```
Emphasizes extreme timesteps (very noisy and very clean). Good for:
- Video models
- Instruction-following models
- Image editing models

### Shift
```yaml
train:
  timestep_type: "shift"
```
Shifted distribution for specific architectures (HiDream).

### Lognorm Blend
```yaml
train:
  timestep_type: "lognorm_blend"
```
Experimental blend distribution.

---

## SNR (Signal-to-Noise Ratio) Weighting

Weight loss by signal-to-noise ratio to balance training across timesteps.

```yaml
train:
  snr_gamma: 5.0  # Min-SNR gamma weighting
```

**Values:**
- `5.0` - Standard recommendation
- `1.0-3.0` - More emphasis on high noise
- `10.0+` - More emphasis on low noise

---

## Noise Offset

Add offset to noise for better dark/bright image reproduction.

```yaml
train:
  noise_offset: 0.05  # 0.0-0.1 typical range
```

**Effects:**
- Higher values → Better extremes (very dark/bright)
- Too high → Color shifts and artifacts

---

## Multi-Dataset Training

Train on multiple datasets with different weights.

```yaml
datasets:
  # Primary dataset
  - folder_path: "/path/to/training"
    trigger_word: "sks"
    network_weight: 1.0

  # Style dataset (lower weight)
  - folder_path: "/path/to/style_images"
    default_caption: "artistic style"
    network_weight: 0.3

  # Regularization (prevents overfitting)
  - folder_path: "/path/to/generic_people"
    is_reg: true
    network_weight: 0.5
```

**Tips:**
- Use `network_weight` to balance datasets
- Regularization datasets should have lower weights
- Different triggers for different concepts

---

## Layer-Specific Training

Train only specific transformer blocks for focused learning.

### Target Specific Layers

```yaml
network:
  type: "lora"
  linear: 128
  network_kwargs:
    only_if_contains:
      - "transformer.single_transformer_blocks.7.proj_out"
      - "transformer.single_transformer_blocks.20.proj_out"
      - "transformer.single_transformer_blocks.25.proj_out"
```

### Exclude Layers

```yaml
network:
  network_kwargs:
    ignore_if_contains:
      - "transformer.single_transformer_blocks."
      - "ff_i.experts"  # Exclude MoE experts
```

### Common Layer Targets

For FLUX:
- `transformer.single_transformer_blocks` - Single-stream blocks
- `transformer.transformer_blocks` - Joint attention blocks

For HiDream:
- Exclude `ff_i.experts` and `ff_i.gate` for faster training

---

## Guidance Loss Training

Train the model to respond to CFG guidance in specific ways.

```yaml
train:
  do_guidance_loss: true
  guidance_loss_target: 3.0  # Target guidance scale
  do_guidance_loss_cfg_zero: false
```

**Use cases:**
- Training guidance-free models
- Fine-tuning CFG response

---

## Contrastive/Differential Guidance

```yaml
train:
  do_differential_guidance: true
  differential_guidance_scale: 3.0
```

Trains the model to differentiate between guided and unguided predictions.

---

## Optimal Noise Pairing

Sample multiple noise patterns and use the best match.

```yaml
train:
  optimal_noise_pairing_samples: 4  # Number of noise samples to try
```

**Effects:**
- Can improve training stability
- Increases computation per step
- `1` = disabled (default)

---

## Force Consistent Noise

Use same noise for same image across training.

```yaml
train:
  force_consistent_noise: true
```

Creates deterministic noise based on image content. May help with:
- Reducing noise in training signal
- More consistent optimization

---

## Blended Blur Noise

Blend in blurred noise for smoother training.

```yaml
train:
  blended_blur_noise: true
```

Experimental technique for potentially smoother results.

---

## Prior Divergence Loss

Add loss term to keep network outputs close to original model.

```yaml
train:
  do_prior_divergence: true
```

Helps prevent drift from base model behavior.

---

## Wavelet Loss

Use wavelet decomposition for frequency-aware loss.

```yaml
train:
  loss_type: "wavelet"
```

**Benefits:**
- Better high-frequency detail preservation
- Can reduce artifacts

---

## Multi-Stage Training (Video Models)

For video models with stage boundaries (Wan 2.2):

```yaml
train:
  switch_boundary_every: 1  # Steps between boundary switches

model:
  model_kwargs:
    train_high_noise: true   # Train high-noise stage
    train_low_noise: true    # Train low-noise stage
```

---

## Parameter Swapping

Train different subsets of parameters at different times.

```yaml
train:
  do_paramiter_swapping: true
  paramiter_swapping_factor: 0.1  # 10% of params active at once
```

**Effects:**
- Reduces VRAM usage
- Slower convergence
- Can improve generalization

---

## Quantization with Accuracy Recovery

Use low-bit quantization with accuracy recovery adapters (ARA).

```yaml
model:
  quantize: true
  qtype: "uint4|ostris/accuracy_recovery_adapters/model_ara.safetensors"
```

**Available Qtypes:**
- `qfloat8` - 8-bit float (default, best quality)
- `uint7` through `uint2` - Integer quantization

**ARA Benefits:**
- Lower memory with maintained quality
- Pre-trained to recover quantization loss

---

## Layer Offloading

Dynamically offload transformer layers to CPU.

```yaml
model:
  layer_offloading: true
  layer_offloading_transformer_percent: 0.8  # 80% of layers offloadable
  layer_offloading_text_encoder_percent: 1.0  # 100% of TE offloadable
```

**Trade-offs:**
- Significantly reduces VRAM
- Increases training time (CPU ↔ GPU transfers)
- Enable for training large models on smaller GPUs

---

## Text Embedding Caching

Cache text encoder outputs for faster training.

```yaml
train:
  cache_text_embeddings: true
  unload_text_encoder: true  # Unload TE after caching
```

**Best for:**
- Single-trigger training
- Memory-constrained setups
- Long training runs

**Note:** Requires pre-encoding all captions before training.

---

## Diffusion Feature Extraction

Use pre-trained feature extractors for additional loss.

```yaml
train:
  diffusion_feature_extractor_path: "/path/to/dfe_model"
  diffusion_feature_extractor_weight: 1.0
```

Adds perceptual-style loss using diffusion features.

---

## Adapter Training

Train T2I adapters, ControlNets, or IP-Adapters.

### T2I Adapter

```yaml
adapter:
  type: "t2i"
  train: true
  name_or_path: null  # Train from scratch
```

### ControlNet

```yaml
adapter:
  type: "control_net"
  train: true
  name_or_path: "path/to/pretrained"  # Or null for scratch
```

### IP-Adapter

```yaml
adapter:
  type: "ip"
  train: true
  image_encoder_path: "openai/clip-vit-large-patch14"
  train_image_encoder: false
```

---

## Concept Sliders

Train sliders that control specific attributes.

```yaml
process:
  - type: 'concept_slider'  # Note: different process type
    slider:
      guidance_strength: 3.0
      anchor_strength: 1.0
      positive_prompt: "person who is happy"
      negative_prompt: "person who is sad"
      target_class: "person"
      anchor_class: ""

    # Multiple targets
    targets:
      - positive: "young person"
        negative: "old person"
        weight: 1.0
      - positive: "smiling"
        negative: "frowning"
        weight: 0.5
```

---

## Experimental: Loss Targets

```yaml
train:
  loss_target: "noise"  # Default
  # Or:
  loss_target: "source"      # Predict clean image
  loss_target: "v_prediction"  # V-prediction parameterization
```

---

## Combining Techniques

Example advanced configuration combining multiple techniques:

```yaml
process:
  - type: 'diffusion_trainer'
    trigger_word: "sks"

    network:
      type: "lokr"
      lokr_full_rank: true
      lokr_factor: 8
      network_kwargs:
        only_if_contains:
          - "transformer_blocks"

    train:
      steps: 3000
      lr: 5e-5
      optimizer: "prodigy"
      optimizer_params:
        weight_decay: 0.01
        decouple: true

      ema_config:
        use_ema: true
        ema_decay: 0.995

      diff_output_preservation: true
      diff_output_preservation_multiplier: 1.0
      diff_output_preservation_class: "person"

      timestep_type: "weighted"
      snr_gamma: 5.0
      noise_offset: 0.03

      cache_text_embeddings: true
      gradient_checkpointing: true

    datasets:
      - folder_path: "/data/training"
        caption_dropout_rate: 0.1
        shuffle_tokens: true
        keep_tokens: 1
        resolution: [768, 1024]
        cache_latents_to_disk: true

      - folder_path: "/data/regularization"
        is_reg: true
        network_weight: 0.3

    model:
      name_or_path: "black-forest-labs/FLUX.1-dev"
      arch: "flux"
      quantize: true
      layer_offloading: true
```
