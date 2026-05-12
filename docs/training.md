# Training Guide

This guide covers different training approaches and best practices for using AI-Toolkit.

## Training Types

### 1. LoRA Training (Most Common)

LoRA (Low-Rank Adaptation) trains a small adapter that modifies model behavior:

```yaml
network:
  type: "lora"
  linear: 16          # Rank - higher = more capacity, more VRAM
  linear_alpha: 16    # Alpha - typically equals rank
```

**Use cases:**
- Character/person training
- Style adaptation
- Concept learning
- Object training

**Recommended settings:**
- Rank 8-32 for characters/concepts
- Rank 64-128 for complex styles
- 500-4000 steps depending on dataset size

### 2. LoKr Training

LoKr uses Kronecker product decomposition for efficient training:

```yaml
network:
  type: "lokr"
  lokr_full_rank: true
  lokr_factor: 8      # Decomposition factor
```

**Benefits:**
- Often smaller file sizes
- Can capture more complex patterns

### 3. Dreambooth / Full Fine-Tuning

Train the full model (no LoRA network):

```yaml
# No network section
train:
  train_unet: true
  train_text_encoder: true  # Optional
```

**Note:** This modifies the base model directly and requires more VRAM.

### 4. Textual Inversion

Train custom embeddings without modifying the model:

```yaml
embedding:
  trigger: "my_concept"
  tokens: 4
  init_words: "person"    # Or "*" for random init
```

### 5. Concept Sliders

Train sliders that control specific attributes:

```yaml
process:
  - type: 'concept_slider'
    slider:
      guidance_strength: 3.0
      positive_prompt: "person who is happy"
      negative_prompt: "person who is sad"
      target_class: "person"
```

---

## Training Workflow

### Step 1: Prepare Dataset

```
my_dataset/
├── image001.jpg
├── image001.txt      # Caption: "a photo of sks person"
├── image002.png
├── image002.txt      # Caption: "sks person smiling"
└── ...
```

**Caption Best Practices:**
- Use a unique trigger word (e.g., "sks", "ohwx")
- Describe the subject AND the scene
- Be consistent with trigger word placement
- Include relevant details (clothing, setting, etc.)

### Step 2: Create Configuration

Copy an example config and modify:

```bash
cp config/examples/train_lora_flux_24gb.yaml config/my_training.yaml
```

Key settings to customize:
- `name` - Unique training run name
- `folder_path` - Path to your dataset
- `trigger_word` - Your unique trigger
- `steps` - Training duration
- `lr` - Learning rate

### Step 3: Run Training

```bash
python run.py config/my_training.yaml
```

### Step 4: Monitor Progress

Check output folder:
- `samples/` - Generated samples
- `*.safetensors` - Model checkpoints
- `config.yaml` - Saved configuration

---

## Model-Specific Guides

### FLUX.1 Training

```yaml
model:
  name_or_path: "black-forest-labs/FLUX.1-dev"
  is_flux: true
  quantize: true        # Required for 24GB GPUs

train:
  noise_scheduler: "flowmatch"
  dtype: bf16           # BF16 required for FLUX

sample:
  sampler: "flowmatch"
  guidance_scale: 4     # Lower than SDXL
```

**FLUX Tips:**
- Requires ~24GB VRAM with quantization
- Use `low_vram: true` if GPU has monitors attached
- Training works well with 500-2000 steps
- Lower learning rates (1e-4 to 5e-5) work well

### FLUX.1-schnell Training

For the distilled schnell model, use a training adapter:

```yaml
model:
  name_or_path: "black-forest-labs/FLUX.1-schnell"
  assistant_lora_path: "ostris/FLUX.1-schnell-training-adapter"
  is_flux: true
  quantize: true

sample:
  guidance_scale: 1     # Schnell doesn't use guidance
  sample_steps: 4       # Fewer steps needed
```

### SDXL Training

```yaml
model:
  name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
  is_xl: true

train:
  noise_scheduler: "ddpm"    # DDPM for SD models
  dtype: fp16

sample:
  sampler: "ddpm"
  guidance_scale: 7
```

### Wan 2.1/2.2 Video Training

```yaml
model:
  name_or_path: "Wan-AI/Wan2.1-T2V-14B-Diffusers"
  arch: "wan21"
  quantize: true
  quantize_te: true
  low_vram: true

datasets:
  - folder_path: "/path/to/videos"
    num_frames: 41        # Frames per video
    shrink_video_to_frames: true

train:
  unload_text_encoder: true   # Required for 24GB

sample:
  num_frames: 40
  fps: 15
```

---

## Training Tips & Optimization

### Learning Rate Guidelines

| Model | Recommended LR |
|-------|---------------|
| FLUX.1 | 1e-4 to 5e-5 |
| SDXL | 1e-4 to 5e-5 |
| SD 1.5 | 1e-4 to 1e-5 |
| Video models | 5e-5 to 1e-5 |

### Dataset Size Guidelines

| Dataset Size | Recommended Steps |
|--------------|-------------------|
| 5-15 images | 500-1000 steps |
| 15-50 images | 1000-2000 steps |
| 50-200 images | 2000-4000 steps |
| 200+ images | 3000-6000 steps |

### Memory Optimization

**For 24GB GPUs:**

```yaml
model:
  quantize: true
  quantize_te: true
  low_vram: true        # If using GPU for display

train:
  gradient_checkpointing: true
  batch_size: 1

datasets:
  - cache_latents_to_disk: true
```

**For 16GB GPUs:**

```yaml
model:
  quantize: true
  quantize_te: true
  low_vram: true
  layer_offloading: true

train:
  gradient_checkpointing: true
  batch_size: 1
  unload_text_encoder: true

datasets:
  - cache_latents_to_disk: true
    resolution: [512]     # Lower resolution
```

### Preventing Overfitting

1. **Caption Dropout:**
```yaml
datasets:
  - caption_dropout_rate: 0.05  # 5% dropout
```

2. **Token Shuffling:**
```yaml
datasets:
  - shuffle_tokens: true
    keep_tokens: 1      # Keep first token (trigger)
```

3. **EMA (Exponential Moving Average):**
```yaml
train:
  ema_config:
    use_ema: true
    ema_decay: 0.99
```

4. **Differential Output Preservation (DOP):**
```yaml
train:
  diff_output_preservation: true
  diff_output_preservation_multiplier: 1.0
  diff_output_preservation_class: "person"
```

### Multi-Resolution Training

Train on multiple resolutions for better generalization:

```yaml
datasets:
  - resolution: [512, 768, 1024]
    buckets: true
```

### Regularization Images

Add regularization to prevent overfitting:

```yaml
datasets:
  - folder_path: "/path/to/training_images"
    trigger_word: "sks"
  - folder_path: "/path/to/regularization_images"
    is_reg: true
    network_weight: 0.5   # Lower weight for regularization
```

---

## Training Specific Layers

### Train Only Specific Blocks

```yaml
network:
  type: "lora"
  linear: 128
  network_kwargs:
    only_if_contains:
      - "transformer.single_transformer_blocks.7.proj_out"
      - "transformer.single_transformer_blocks.20.proj_out"
```

### Exclude Certain Layers

```yaml
network:
  network_kwargs:
    ignore_if_contains:
      - "transformer.single_transformer_blocks."
```

---

## Resuming Training

Training automatically resumes from the last checkpoint:

```bash
python run.py config/my_training.yaml
```

The toolkit will find the latest checkpoint in the output folder and continue.

To start fresh, either:
1. Change the `name` in config
2. Delete the output folder
3. Use `train.start_step: 0`

---

## Multiple Configurations

Run multiple training runs sequentially:

```bash
python run.py config/run1.yaml config/run2.yaml config/run3.yaml
```

Use `-r` to continue if one fails:

```bash
python run.py -r config/run1.yaml config/run2.yaml
```

---

## Training Quality Indicators

### Good Signs
- Loss decreasing over time
- Samples improving progressively
- Trigger word producing expected results

### Warning Signs
- Loss not decreasing (LR too low or dataset issues)
- Loss going to zero (overfitting)
- Samples degrading (LR too high or too many steps)
- Color/contrast shifts (noise offset issues)

### Typical Loss Values

| Model | Expected Loss Range |
|-------|---------------------|
| FLUX.1 | 0.02 - 0.10 |
| SDXL | 0.02 - 0.08 |
| SD 1.5 | 0.03 - 0.10 |

---

## Command Line Options

```bash
# Basic usage
python run.py config/my_config.yaml

# With name override
python run.py config/my_config.yaml -n "experiment_001"

# Recover from failures
python run.py -r config/config1.yaml config/config2.yaml

# With logging
python run.py config/my_config.yaml -l training.log
```
