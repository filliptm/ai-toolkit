# Extensions System

AI-Toolkit uses an extension system for modularity. This document explains how to use existing extensions and create new ones.

## Built-in Extensions

### sd_trainer

The main training extension for LoRA, Dreambooth, and fine-tuning.

```yaml
process:
  - type: 'sd_trainer'
```

**Features:**
- LoRA/LoKr/LoCon training
- Full fine-tuning
- Textual inversion
- Multi-dataset support
- Sample generation

### diffusion_trainer

Universal trainer supporting both UI and API:

```yaml
process:
  - type: 'diffusion_trainer'
```

**Same capabilities as sd_trainer, plus:**
- SQLite logging for UI
- Real-time progress updates

### ui_trainer

Legacy trainer for UI (deprecated, use `diffusion_trainer`):

```yaml
process:
  - type: 'ui_trainer'
```

---

## Extension Architecture

### File Structure

```
extensions_built_in/
└── sd_trainer/
    ├── __init__.py       # Extension registration
    ├── SDTrainer.py      # Training process class
    ├── UITrainer.py      # UI-specific trainer
    └── DiffusionTrainer.py
```

### Extension Registration

Every extension module must export `AI_TOOLKIT_EXTENSIONS`:

```python
# extensions_built_in/sd_trainer/__init__.py

from toolkit.extension import Extension

class SDTrainerExtension(Extension):
    uid = "sd_trainer"        # Unique ID (used in config type)
    name = "SD Trainer"       # Display name

    @classmethod
    def get_process(cls):
        from .SDTrainer import SDTrainer
        return SDTrainer

# Export list of extensions
AI_TOOLKIT_EXTENSIONS = [
    SDTrainerExtension,
]
```

---

## Creating Custom Extensions

### Step 1: Create Extension Directory

```bash
mkdir -p extensions/my_extension
touch extensions/my_extension/__init__.py
touch extensions/my_extension/MyProcess.py
```

### Step 2: Define Process Class

```python
# extensions/my_extension/MyProcess.py

from collections import OrderedDict
from jobs.process import BaseSDTrainProcess

class MyProcess(BaseSDTrainProcess):
    """Custom training process"""

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        # Custom initialization
        self.my_custom_setting = self.get_conf('my_setting', default='value')

    def before_model_load(self):
        """Called before loading the model"""
        super().before_model_load()
        print("Custom pre-load logic")

    def hook_before_train_loop(self):
        """Called before training starts"""
        super().hook_before_train_loop()
        print("Training is about to start")

    def hook_train_loop(self, batch):
        """
        Main training step - called every iteration.
        Override this for custom training logic.
        """
        # Get batch data
        imgs = batch['images']
        captions = batch['captions']

        # Your training logic here
        # ...

        # Return loss dict
        return {'loss': loss.item()}

    def sample(self, step):
        """Generate sample images"""
        super().sample(step)
        # Custom sampling logic

    def cleanup(self):
        """Called when training ends"""
        super().cleanup()
        print("Cleaning up")
```

### Step 3: Register Extension

```python
# extensions/my_extension/__init__.py

from toolkit.extension import Extension

class MyExtension(Extension):
    uid = "my_trainer"
    name = "My Custom Trainer"

    @classmethod
    def get_process(cls):
        from .MyProcess import MyProcess
        return MyProcess

AI_TOOLKIT_EXTENSIONS = [
    MyExtension,
]
```

### Step 4: Use in Config

```yaml
job: extension
config:
  name: "my_training"
  process:
    - type: 'my_trainer'      # Your extension uid
      my_setting: "custom"    # Your custom setting
      # ... rest of config
```

---

## Process Class Hierarchy

Understanding the base classes helps when extending:

```
BaseProcess
    ├── get_conf(key, default, required)  # Access config values
    ├── run()                              # Abstract method
    └── print()                            # Logging

BaseTrainProcess (extends BaseProcess)
    ├── tensorboard integration
    ├── training_folder management
    └── save_training_config()

BaseSDTrainProcess (extends BaseTrainProcess)
    ├── sd (StableDiffusion instance)
    ├── network (LoRA network)
    ├── optimizer, scheduler
    ├── data_loader
    │
    ├── before_model_load()        # Hook: before loading
    ├── hook_before_train_loop()   # Hook: before training
    ├── hook_train_loop(batch)     # Hook: each training step
    ├── hook_after_train_loop()    # Hook: after training
    ├── sample(step)               # Generate samples
    └── save_checkpoint(step)      # Save weights

SDTrainer (extends BaseSDTrainProcess)
    └── Full implementation of training loop
```

---

## Key Hook Points

### before_model_load()

Called before the model is loaded. Use for:
- Modifying model config
- Setting up custom components
- Logging

```python
def before_model_load(self):
    super().before_model_load()
    self.custom_component = load_my_component()
```

### hook_before_train_loop()

Called after model load, before training. Use for:
- Preparing training state
- Caching embeddings
- Final setup

```python
def hook_before_train_loop(self):
    super().hook_before_train_loop()
    self.cached_embeds = self.sd.encode_prompt("trigger")
```

### hook_train_loop(batch)

The main training step. Override for custom training:

```python
def hook_train_loop(self, batch):
    # Don't call super() - replace the training logic

    with torch.no_grad():
        latents = self.sd.encode_images(batch['images'])

    noise = torch.randn_like(latents)
    timesteps = self.sample_timesteps(latents.shape[0])
    noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

    embeds = self.sd.encode_prompt(batch['captions'])

    with self.network:
        pred = self.sd.unet(noisy_latents, timesteps, embeds)

    loss = F.mse_loss(pred, noise)
    loss.backward()

    return {'loss': loss.item()}
```

### sample(step)

Generate sample images:

```python
def sample(self, step):
    if self.sample_config is None:
        return

    # Custom sampling logic
    for prompt in self.sample_config.prompts:
        image = self.sd.generate_image(prompt)
        image.save(f"samples/{step}_{prompt[:20]}.png")
```

---

## Accessing Configuration

Use `self.get_conf()` to access nested config values:

```python
# Config:
# process:
#   - type: 'my_trainer'
#     custom:
#       nested:
#         value: 42

# Access:
value = self.get_conf('custom.nested.value', default=0)

# Required values:
required = self.get_conf('important_setting', required=True)
```

---

## Pre-built Config Classes

Use existing config classes for type safety:

```python
from toolkit.config_modules import (
    ModelConfig,
    TrainConfig,
    NetworkConfig,
    DatasetConfig,
    SaveConfig,
    SampleConfig,
)

class MyProcess(BaseSDTrainProcess):
    def __init__(self, ...):
        super().__init__(...)

        # Access typed configs
        self.model_config: ModelConfig = self.model_config
        self.train_config: TrainConfig = self.train_config
        self.network_config: NetworkConfig = self.network_config
```

---

## Adding Custom Config Options

For new config options, modify or extend config classes:

```python
# In your process
def __init__(self, ...):
    super().__init__(...)

    # Simple custom options
    self.my_option = self.get_conf('my_option', default=False)

    # Nested custom options
    custom_config = self.get_conf('custom', default={})
    self.custom_lr = custom_config.get('lr', 1e-4)
    self.custom_steps = custom_config.get('steps', 100)
```

Config usage:

```yaml
process:
  - type: 'my_trainer'
    my_option: true
    custom:
      lr: 0.0001
      steps: 500
```

---

## Example: Custom Loss Extension

Here's a complete example adding a custom loss function:

```python
# extensions/custom_loss/__init__.py
from toolkit.extension import Extension

class CustomLossExtension(Extension):
    uid = "custom_loss_trainer"
    name = "Custom Loss Trainer"

    @classmethod
    def get_process(cls):
        from .CustomLossTrainer import CustomLossTrainer
        return CustomLossTrainer

AI_TOOLKIT_EXTENSIONS = [CustomLossExtension]
```

```python
# extensions/custom_loss/CustomLossTrainer.py
import torch
import torch.nn.functional as F
from collections import OrderedDict
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer

class CustomLossTrainer(SDTrainer):
    """Trainer with custom loss function"""

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)

        # Custom loss settings
        self.loss_type = self.get_conf('train.loss_type', default='custom')
        self.perceptual_weight = self.get_conf('train.perceptual_weight', default=0.1)

    def compute_loss(self, pred, target, timesteps):
        """Custom loss computation"""
        # Base MSE loss
        mse_loss = F.mse_loss(pred, target)

        # Add custom perceptual component
        if self.perceptual_weight > 0:
            perceptual = self.compute_perceptual_loss(pred, target)
            total_loss = mse_loss + self.perceptual_weight * perceptual
        else:
            total_loss = mse_loss

        return total_loss

    def compute_perceptual_loss(self, pred, target):
        """Your custom perceptual loss"""
        # Implementation here
        return torch.tensor(0.0)
```

Config:

```yaml
job: extension
config:
  name: "custom_loss_training"
  process:
    - type: 'custom_loss_trainer'
      train:
        loss_type: 'custom'
        perceptual_weight: 0.1
        # ... rest of train config
```

---

## Tips for Extension Development

1. **Start from existing code** - Copy SDTrainer as a base
2. **Use hooks** - Override hooks rather than rewriting everything
3. **Test incrementally** - Add features one at a time
4. **Check types** - Use config classes for type safety
5. **Handle errors** - Add proper error handling
6. **Document** - Add docstrings and comments
7. **Keep it modular** - Separate concerns into methods
