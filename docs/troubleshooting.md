# Troubleshooting Guide

This guide addresses common issues and their solutions when using AI-Toolkit.

## Installation Issues

### CUDA/PyTorch Issues

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
```yaml
# Reduce memory usage:
model:
  quantize: true
  quantize_te: true
  low_vram: true

train:
  batch_size: 1
  gradient_checkpointing: true

datasets:
  - cache_latents_to_disk: true
```

**Problem:** `CUDA not available`

**Check:**
```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch sees CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Fix:**
```bash
# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

---

### Dependency Issues

**Problem:** `ModuleNotFoundError: No module named 'xxx'`

**Fix:**
```bash
pip install -r requirements.txt
```

**Problem:** Version conflicts

**Fix:**
```bash
# Create fresh environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

---

## Training Issues

### Loss Not Decreasing

**Causes:**
1. Learning rate too low
2. Dataset issues
3. Captions don't match images

**Solutions:**

1. **Increase learning rate:**
```yaml
train:
  lr: 1e-4  # Try higher if stuck
```

2. **Check dataset:**
```bash
# Verify images load correctly
python -c "from PIL import Image; Image.open('path/to/image.jpg').show()"
```

3. **Verify captions:**
- Ensure `.txt` files exist for each image
- Check trigger word is present
- Verify captions describe images accurately

---

### Loss Goes to Zero

**Cause:** Overfitting

**Solutions:**
```yaml
train:
  # Reduce steps
  steps: 1000  # Instead of 3000

  # Add caption dropout
datasets:
  - caption_dropout_rate: 0.1

  # Use EMA
  ema_config:
    use_ema: true
    ema_decay: 0.99
```

---

### NaN Loss

**Causes:**
1. Learning rate too high
2. Numerical instability
3. Bad data

**Solutions:**

1. **Lower learning rate:**
```yaml
train:
  lr: 1e-5  # Much lower
```

2. **Use gradient clipping:**
```yaml
train:
  max_grad_norm: 1.0
```

3. **Check for corrupt images:**
```python
# Script to find corrupt images
from PIL import Image
import os

for f in os.listdir("dataset"):
    if f.endswith(('.jpg', '.png')):
        try:
            Image.open(f"dataset/{f}").verify()
        except:
            print(f"Corrupt: {f}")
```

---

### Out of Memory During Training

**Progressive solutions:**

1. **Enable quantization:**
```yaml
model:
  quantize: true
  quantize_te: true
```

2. **Enable gradient checkpointing:**
```yaml
train:
  gradient_checkpointing: true
```

3. **Cache latents:**
```yaml
datasets:
  - cache_latents_to_disk: true
```

4. **Reduce batch size:**
```yaml
train:
  batch_size: 1
  gradient_accumulation: 4  # Effective batch of 4
```

5. **Lower resolution:**
```yaml
datasets:
  - resolution: 512  # Instead of 1024
```

6. **Enable layer offloading:**
```yaml
model:
  layer_offloading: true
```

7. **Unload text encoder:**
```yaml
train:
  unload_text_encoder: true
```

---

### Samples Look Bad

**Problem:** Generated samples during training look wrong

**Causes & Solutions:**

1. **Guidance scale wrong:**
```yaml
sample:
  guidance_scale: 4  # FLUX
  # or
  guidance_scale: 7  # SDXL
```

2. **Wrong sampler:**
```yaml
sample:
  sampler: "flowmatch"  # Must match train.noise_scheduler
```

3. **Trigger word not in prompt:**
```yaml
sample:
  samples:
    - prompt: "sks person at the beach"  # Include trigger
```

4. **Not enough steps:**
```yaml
sample:
  sample_steps: 20  # Increase if needed
```

---

### Training Too Slow

**Solutions:**

1. **Cache latents:**
```yaml
datasets:
  - cache_latents_to_disk: true
```

2. **Use 8-bit optimizer:**
```yaml
train:
  optimizer: "adamw8bit"
```

3. **Reduce sample frequency:**
```yaml
sample:
  sample_every: 500  # Instead of 100
```

4. **Use bf16 (if supported):**
```yaml
train:
  dtype: bf16  # Faster than fp16 on new GPUs
```

---

## Model Loading Issues

### Model Not Found

**Problem:** `Could not find model at xxx`

**Solutions:**

1. Check path is correct:
```yaml
model:
  name_or_path: "black-forest-labs/FLUX.1-dev"  # HuggingFace ID
  # or
  name_or_path: "/path/to/local/model"  # Local path
```

2. Login to HuggingFace:
```bash
huggingface-cli login
```

3. Accept model license on HuggingFace website

---

### Quantization Errors

**Problem:** Errors during quantization

**Solutions:**

1. **Use low_vram mode:**
```yaml
model:
  quantize: true
  low_vram: true  # Quantize on CPU
```

2. **Try different qtype:**
```yaml
model:
  qtype: "float8"  # Instead of qfloat8
```

---

## Dataset Issues

### Images Not Loading

**Problem:** `Error loading image`

**Check:**
- File extensions: `.jpg`, `.jpeg`, `.png` (not `.webp`)
- File permissions
- File not corrupted

**Fix corrupted files:**
```python
from PIL import Image
import os

for f in os.listdir("dataset"):
    if f.endswith(('.jpg', '.png')):
        try:
            img = Image.open(f"dataset/{f}")
            img = img.convert('RGB')
            img.save(f"dataset/{f}")  # Resave
        except Exception as e:
            print(f"Error with {f}: {e}")
```

---

### Caption Files Not Found

**Problem:** Training uses default caption or blank

**Check:**
- Caption file has same name as image
- Extension matches config (`caption_ext: "txt"`)
- File is readable

**Example structure:**
```
dataset/
├── image001.jpg
├── image001.txt    # Must match!
├── image002.png
└── image002.txt    # Must match!
```

---

### Resolution Mismatch

**Problem:** Images too small

**Error:** `Image too small for resolution`

**Solution:**
```yaml
datasets:
  - resolution: 512  # Lower resolution
    # or
    scale: 0.5  # Scale images down
```

---

## Checkpoint Issues

### Can't Resume Training

**Problem:** Training starts from scratch

**Causes:**
1. Changed `name` in config
2. Deleted output folder
3. No checkpoint found

**Solution:**
Ensure config `name` matches existing training folder.

---

### Checkpoint Won't Load

**Problem:** `Error loading checkpoint`

**Solutions:**

1. **Check file integrity:**
```python
from safetensors.torch import load_file
state = load_file("checkpoint.safetensors")
print(f"Keys: {len(state)}")
```

2. **Remove corrupted checkpoint** and use earlier one

---

## UI Issues

### UI Won't Start

**Check:**
```bash
cd ui
node --version  # Need 18+
npm --version
```

**Fix:**
```bash
rm -rf node_modules .next
npm install
npm run dev
```

---

### Jobs Not Appearing

**Causes:**
1. Database path mismatch
2. Wrong training folder

**Check config:**
```yaml
process:
  - type: 'diffusion_trainer'
    training_folder: "output"  # Must match
    sqlite_db_path: "./aitk_db.db"  # Must match UI
```

---

### API Errors

**Problem:** `500 Internal Server Error`

**Debug:**
```bash
# Run in dev mode
npm run dev

# Check console output for errors
```

---

## Common Error Messages

### `RuntimeError: Expected all tensors on same device`

**Cause:** Model components on different devices

**Fix:**
```yaml
model:
  low_vram: true  # Ensures consistent device handling
```

---

### `ValueError: Attempting to unscale FP16 gradients`

**Cause:** Mixed precision issues

**Fix:**
```yaml
train:
  dtype: bf16  # Use bf16 instead of fp16
```

---

### `AssertionError: No images found`

**Cause:** Dataset folder empty or wrong path

**Check:**
- Path is correct (use absolute path on Windows)
- Folder contains supported images
- File extensions are lowercase

---

### `torch.cuda.OutOfMemoryError`

See "Out of Memory During Training" section above.

---

## Performance Tips

### Faster Training

1. **Cache everything:**
```yaml
datasets:
  - cache_latents_to_disk: true
train:
  cache_text_embeddings: true
```

2. **Use 8-bit optimizer:**
```yaml
train:
  optimizer: "adamw8bit"
```

3. **Compile model (experimental):**
```yaml
model:
  compile: true
```

### Better Quality

1. **Use EMA:**
```yaml
train:
  ema_config:
    use_ema: true
```

2. **More diverse data:**
- Add more images
- Vary poses, lighting, backgrounds

3. **Regularization:**
```yaml
datasets:
  - folder_path: "/training"
  - folder_path: "/regularization"
    is_reg: true
```

---

## Getting Help

If issues persist:

1. **Check logs:**
   - Console output
   - TensorBoard logs
   - UI logs

2. **Verify versions:**
```bash
python --version
pip show torch
pip show diffusers
```

3. **Report issues:**
   - GitHub: https://github.com/ostris/ai-toolkit/issues
   - Include: config, error message, system info
