# Architecture Overview

This document describes the internal architecture of AI-Toolkit, explaining how components interact to enable flexible diffusion model training.

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        run.py (Entry Point)                       │
│  - Parses command line arguments                                  │
│  - Loads config files                                             │
│  - Dispatches to appropriate Job type                             │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                      toolkit/job.py                               │
│  get_job() - Factory function that creates job instances          │
│  - extract → ExtractJob                                           │
│  - train → TrainJob                                               │
│  - mod → ModJob                                                   │
│  - generate → GenerateJob                                         │
│  - extension → ExtensionJob (most common)                         │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                      jobs/BaseJob.py                              │
│  Base class for all jobs:                                         │
│  - Loads configuration                                            │
│  - Manages process list                                           │
│  - Executes run() and cleanup()                                   │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                   jobs/ExtensionJob.py                            │
│  Loads processes from extensions:                                 │
│  - Scans extensions/ and extensions_built_in/                     │
│  - Maps process types to Extension classes                        │
│  - Instantiates appropriate Process class                         │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│              Process Classes (Training Logic)                     │
│                                                                   │
│  BaseProcess → BaseTrainProcess → BaseSDTrainProcess → SDTrainer  │
│                                                                   │
│  Each layer adds functionality:                                   │
│  - BaseProcess: Config access, timing                             │
│  - BaseTrainProcess: Tensorboard, checkpointing                   │
│  - BaseSDTrainProcess: SD model loading, LoRA, sampling           │
│  - SDTrainer: Actual training loop implementation                 │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│              StableDiffusion (Model Wrapper)                      │
│  toolkit/stable_diffusion_model.py                                │
│                                                                   │
│  Unified interface for all model architectures:                   │
│  - SD 1.5, SD 2.x, SDXL, SD3                                     │
│  - FLUX.1, Flex.1/2, Chroma                                      │
│  - PixArt, AuraFlow, Lumina2                                     │
│  - Wan 2.1/2.2, LTX-2                                            │
│  - HiDream, OmniGen2, Qwen-Image                                 │
│                                                                   │
│  Handles: model loading, encoding, noise scheduling, generation   │
└──────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Configuration System (`toolkit/config.py`, `toolkit/config_modules.py`)

The configuration system is the backbone of AI-Toolkit:

```python
# toolkit/config.py
def get_config(config_file_path_or_dict, name=None):
    """
    Loads config from YAML/JSON file or dict.
    Supports:
    - Environment variable substitution: ${VAR_NAME}
    - Name tag replacement: [name]
    - Auto-extension detection (.yaml, .yml, .json)
    """
```

Configuration is parsed into typed dataclasses in `config_modules.py`:

| Config Class | Purpose |
|-------------|---------|
| `ModelConfig` | Model path, architecture, quantization settings |
| `TrainConfig` | Learning rate, steps, optimizer, scheduler |
| `NetworkConfig` | LoRA/LoKr rank, alpha, type |
| `DatasetConfig` | Dataset paths, resolution, augmentations |
| `SaveConfig` | Checkpoint frequency, format, dtype |
| `SampleConfig` | Sampling prompts, dimensions, steps |
| `AdapterConfig` | IP-Adapter, ControlNet, T2I adapter settings |
| `EmbeddingConfig` | Textual inversion trigger and tokens |

### 2. Extension System (`toolkit/extension.py`)

Extensions provide pluggable training processes:

```python
# Extension base class
class Extension:
    name: str = None    # Display name
    uid: str = None     # Unique identifier (used in config)

    @classmethod
    def get_process(cls):
        """Returns the Process class to use"""
        pass
```

Extensions are discovered from two directories:
- `extensions/` - User custom extensions
- `extensions_built_in/` - Shipped extensions

Each extension module must export `AI_TOOLKIT_EXTENSIONS` list:

```python
# extensions_built_in/sd_trainer/__init__.py
AI_TOOLKIT_EXTENSIONS = [
    SDTrainerExtension,      # uid: "sd_trainer"
    UITrainerExtension,      # uid: "ui_trainer"
    DiffusionTrainerExtension,  # uid: "diffusion_trainer"
]
```

### 3. Process Hierarchy

```
BaseProcess
    │
    ├── get_conf() - Access nested config values
    ├── run() - Abstract execution method
    └── timer - Performance tracking

BaseTrainProcess (extends BaseProcess)
    │
    ├── TensorBoard integration
    ├── save_training_config()
    └── training_folder management

BaseSDTrainProcess (extends BaseTrainProcess)
    │
    ├── Model loading and management
    ├── Network (LoRA/LoKr) setup
    ├── Optimizer and scheduler creation
    ├── Dataset loading
    ├── Checkpoint save/load
    ├── Sample generation
    └── EMA (Exponential Moving Average)

SDTrainer (extends BaseSDTrainProcess)
    │
    └── hook_train_loop() - The actual training step
        ├── Get batch from dataloader
        ├── Encode images to latents
        ├── Add noise at timestep
        ├── Get text embeddings
        ├── Forward pass through UNet/Transformer
        ├── Compute loss
        ├── Backward pass
        └── Optimizer step
```

### 4. StableDiffusion Model Wrapper

`toolkit/stable_diffusion_model.py` provides a unified interface across architectures:

```python
class StableDiffusion:
    def __init__(self, device, model_config, dtype, ...):
        self.arch: ModelArch  # 'sd1', 'sdxl', 'flux', etc.
        self.vae: AutoencoderKL
        self.unet: UNet2DConditionModel  # or Transformer
        self.text_encoder: List[CLIPTextModel]
        self.tokenizer: List[CLIPTokenizer]
        self.noise_scheduler: DDPMScheduler

    def load_model(self):
        """Load all model components based on arch"""

    def encode_prompt(self, prompt, ...):
        """Encode text to embeddings"""

    def encode_images(self, images):
        """Encode images to latent space via VAE"""

    def decode_latents(self, latents):
        """Decode latents to images via VAE"""

    def generate_images(self, config_list):
        """Full inference pipeline for sampling"""
```

### 5. Data Loading System (`toolkit/data_loader.py`)

The data loading system uses mixins for modularity:

```python
class AiToolkitDataset(
    Dataset,
    CaptionMixin,           # Caption loading and processing
    BucketsMixin,           # Aspect ratio bucketing
    LatentCachingMixin,     # Cache encoded latents to disk
    CLIPCachingMixin,       # Cache CLIP embeddings
    ControlCachingMixin,    # Cache control images
    TextEmbeddingCachingMixin  # Cache text encoder outputs
):
    """Main dataset class combining all capabilities"""
```

Key features:
- **Bucketing**: Groups images by aspect ratio to minimize padding
- **Latent Caching**: Pre-encodes images to latent space for faster training
- **Caption Processing**: Token shuffling, dropout, trigger word injection
- **Multi-resolution**: Train on multiple resolutions simultaneously

### 6. Network Types (LoRA, LoKr, etc.)

Located in `toolkit/lora_special.py` and `toolkit/lycoris_special.py`:

```python
class LoRASpecialNetwork:
    """LoRA network implementation"""

    def __init__(self, ...):
        self.lora_dim = linear          # Rank
        self.alpha = linear_alpha       # Alpha for scaling
        self.dropout = dropout

    def apply_to(self, model, ...):
        """Inject LoRA layers into model"""

    def save_weights(self, path, ...):
        """Save LoRA weights to safetensors"""
```

Supported network types:
- **LoRA** - Low-Rank Adaptation
- **LoKr** - Low-Rank Kronecker Product
- **LoCon** - LoRA with Convolution layers
- **LoRM** - Low-Rank Matrix (experimental)

### 7. Accelerator Integration

Uses HuggingFace Accelerate for multi-GPU and mixed precision:

```python
# toolkit/accelerator.py
def get_accelerator():
    """Returns singleton Accelerator instance"""

# Used throughout for:
# - Automatic device placement
# - Mixed precision training (fp16, bf16)
# - Multi-GPU distribution
# - Gradient accumulation
```

## Data Flow During Training

```
1. Config Loading
   config.yaml → get_config() → Config dataclasses

2. Model Initialization
   ModelConfig → StableDiffusion.load_model() →
   VAE, UNet, TextEncoders on device

3. Dataset Preparation
   DatasetConfig → AiToolkitDataset →
   Cache latents (optional) → DataLoader

4. Training Loop (per step)

   batch = next(dataloader)
   ├── images: [B, C, H, W] or cached latents: [B, C, H/8, W/8]
   ├── captions: List[str]
   └── control_images: Optional[Tensor]

   latents = vae.encode(images) if not cached

   noise = torch.randn_like(latents)
   timesteps = sample_timesteps(batch_size)
   noisy_latents = scheduler.add_noise(latents, noise, timesteps)

   embeddings = text_encoder(captions)

   with network.active():  # Enable LoRA
       prediction = unet(noisy_latents, timesteps, embeddings)

   loss = mse_loss(prediction, noise)  # or v-prediction target

   loss.backward()
   optimizer.step()
   scheduler.step()

5. Checkpointing
   Every N steps → save_weights() → .safetensors

6. Sampling
   Every N steps → generate_images() → sample images to disk
```

## Memory Optimization Strategies

AI-Toolkit employs several strategies for memory efficiency:

1. **8-bit Quantization** (`model.quantize: true`)
   - Uses optimum-quanto for 8-bit inference
   - Reduces model memory by ~50%

2. **Gradient Checkpointing** (`train.gradient_checkpointing: true`)
   - Trades compute for memory
   - Recomputes activations during backward pass

3. **Latent Caching** (`datasets[].cache_latents_to_disk: true`)
   - Pre-encodes images to latents
   - Avoids loading VAE during training

4. **Text Encoder Unloading** (`train.unload_text_encoder: true`)
   - Caches trigger word embeddings
   - Moves text encoder to CPU after encoding

5. **Low VRAM Mode** (`model.low_vram: true`)
   - Quantizes on CPU before moving to GPU
   - Slower but uses less peak memory

6. **Layer Offloading** (`model.layer_offloading: true`)
   - Offloads transformer layers to CPU when not in use
   - Enables training larger models on smaller GPUs

## File Outputs

Training produces the following outputs in `output/{name}/`:

```
output/my_lora_v1/
├── config.yaml              # Copy of training config
├── my_lora_v1_000000500.safetensors  # Checkpoint at step 500
├── my_lora_v1_000001000.safetensors  # Checkpoint at step 1000
├── optimizer.pt             # Optimizer state (for resuming)
└── samples/
    ├── 20240101-120000_000000500_0.jpg
    ├── 20240101-120000_000000500_1.jpg
    └── ...
```
