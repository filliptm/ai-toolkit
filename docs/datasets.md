# Dataset Preparation Guide

This guide covers how to prepare datasets for training with AI-Toolkit.

## Dataset Structure

### Basic Image Dataset

```
my_dataset/
├── image001.jpg
├── image001.txt
├── image002.png
├── image002.txt
├── image003.webp
├── image003.txt
└── ...
```

**Requirements:**
- Images: `.jpg`, `.jpeg`, `.png` (`.webp` may have issues)
- Captions: `.txt` files with same name as image
- One caption per image

### Caption File Format

Plain text, one line:

```
a photo of sks person standing in a park, sunny day, casual clothing
```

Or with trigger placeholder:

```
a photo of [trigger] standing in a park, sunny day, casual clothing
```

The `[trigger]` placeholder is replaced with your `trigger_word` from config.

---

## Caption Best Practices

### For Character/Person Training

```
sks person sitting at a cafe, casual outfit, natural lighting
sks person portrait, professional headshot, studio lighting
sks person full body shot, walking in the city, urban background
```

**Tips:**
- Place trigger word at the start or consistently
- Describe pose, clothing, lighting, background
- Vary descriptions across images
- Include both close-ups and full body shots

### For Style Training

```
painting in the style of artstyle, landscape with mountains
artstyle artwork of a portrait, oil painting texture
digital art in artstyle style, abstract composition
```

**Tips:**
- Use consistent style trigger
- Describe subject matter and technique
- Include variety of subjects
- Note distinctive style elements

### For Object Training

```
product shot of sks object on white background
sks object held in hands, lifestyle photography
close-up of sks object showing details
```

---

## Advanced Dataset Formats

### JSON Dataset

For more control, use a JSON dataset file:

```json
{
  "images": [
    {
      "path": "/full/path/to/image1.jpg",
      "caption": "a photo of sks person",
      "mask_path": "/path/to/mask1.png"
    },
    {
      "path": "/full/path/to/image2.jpg",
      "caption": "another caption",
      "short_caption": "short version",
      "long_caption": "detailed long version"
    }
  ]
}
```

Config usage:

```yaml
datasets:
  - dataset_path: "/path/to/dataset.json"
```

### With Control Images

For ControlNet-style training:

```
dataset/
├── images/
│   ├── image001.jpg
│   └── image002.jpg
├── captions/
│   ├── image001.txt
│   └── image002.txt
└── control/
    ├── image001.jpg    # Depth/edge/pose map
    └── image002.jpg
```

Config:

```yaml
datasets:
  - folder_path: "dataset/images"
    caption_ext: "txt"
    control_path: "dataset/control"
```

### With Masks (Alpha Channel)

For masked training, use PNG/WebP with alpha channel:

```yaml
datasets:
  - folder_path: "/path/to/images"
    alpha_mask: true    # Use alpha channel as mask
```

Or separate mask folder:

```yaml
datasets:
  - folder_path: "/path/to/images"
    mask_path: "/path/to/masks"
    mask_min_value: 0.1
```

---

## Video Datasets

### For Video Models (Wan, LTX, etc.)

```
video_dataset/
├── video001.mp4
├── video001.txt
├── video002.mp4
├── video002.txt
└── ...
```

Config:

```yaml
datasets:
  - folder_path: "/path/to/videos"
    num_frames: 41              # Frames to sample
    shrink_video_to_frames: true
    fps: 16                     # Target FPS
```

**Supported formats:** `.mp4`, `.avi`, `.mov`, `.webm`, `.mkv`

### Image-to-Video Training

For I2V models, organize with first frames:

```yaml
datasets:
  - folder_path: "/path/to/videos"
    do_i2v: true    # Use first frame as conditioning
    num_frames: 41
```

---

## Dataset Configuration Reference

### Resolution Settings

```yaml
datasets:
  # Single resolution
  - resolution: 1024

  # Multiple resolutions (bucketing)
  - resolution: [512, 768, 1024]
    buckets: true
    bucket_tolerance: 64
```

### Caption Processing

```yaml
datasets:
  - folder_path: "/path/to/images"
    caption_ext: "txt"           # Caption extension
    default_caption: null        # Fallback if no caption file
    trigger_word: "sks"          # Auto-inject trigger

    # Regularization
    caption_dropout_rate: 0.05   # 5% chance to drop caption
    token_dropout_rate: 0.0      # Drop individual tokens
    shuffle_tokens: false        # Shuffle comma-separated tokens
    keep_tokens: 1               # Tokens to keep when shuffling
```

### Random Triggers

For variety, use random trigger words:

```yaml
datasets:
  - folder_path: "/path/to/images"
    random_triggers:
      - "trigger1"
      - "trigger2"
      - "trigger3"
    random_triggers_max: 1       # Max triggers to add
```

Or from a file:

```yaml
datasets:
  - random_triggers: "/path/to/triggers.txt"
```

### Augmentations

```yaml
datasets:
  - folder_path: "/path/to/images"
    flip_x: true                 # Random horizontal flip
    flip_y: false                # Random vertical flip
    random_crop: true            # Random cropping
    random_scale: true           # Random scaling
    scale: 1.0                   # Base scale factor
```

### Caching Options

```yaml
datasets:
  - folder_path: "/path/to/images"
    cache_latents_to_disk: true      # Cache VAE latents
    cache_clip_vision_to_disk: false # Cache CLIP embeddings
    cache_text_embeddings: true      # Cache text embeddings
```

**When to cache:**
- `cache_latents_to_disk`: Almost always (saves VRAM)
- `cache_text_embeddings`: When training many steps, saves TE memory

---

## Regularization Datasets

Prevent overfitting with regularization images:

```yaml
datasets:
  # Training images
  - folder_path: "/path/to/training"
    trigger_word: "sks"

  # Regularization images (no trigger)
  - folder_path: "/path/to/regularization"
    is_reg: true
    network_weight: 0.5   # Lower loss weight
```

**Regularization sources:**
- Images generated by the base model
- Generic images of the same class
- Stock photos or public domain images

---

## Dataset Quality Guidelines

### Image Quality

| Aspect | Recommendation |
|--------|----------------|
| Resolution | At least 512px on shortest side |
| Format | JPG, PNG (avoid WebP) |
| Quality | High quality, minimal compression |
| Consistency | Similar lighting/style if possible |

### Dataset Size

| Training Goal | Recommended Images |
|--------------|-------------------|
| Character/person | 10-50 images |
| Style | 50-200 images |
| Concept | 10-30 images |
| Complex subject | 50-100+ images |

### Caption Quality

| Do | Don't |
|----|-------|
| Describe what's visible | Describe emotions or abstract concepts |
| Use consistent trigger | Use different triggers |
| Include relevant details | Be overly verbose |
| Vary descriptions | Copy-paste same caption |

---

## Common Dataset Issues

### Problem: Loss not decreasing

**Causes:**
- Captions don't match images
- Images too low resolution
- Dataset too small

**Solutions:**
- Review and fix captions
- Ensure images meet resolution requirements
- Add more diverse images

### Problem: Model only works with trigger

**Causes:**
- Overfit to trigger word
- Trigger appears in every caption identically

**Solutions:**
- Use caption dropout
- Vary trigger word position
- Add regularization images

### Problem: Quality degradation

**Causes:**
- Overfitting
- Too many training steps
- Learning rate too high

**Solutions:**
- Use EMA
- Reduce steps
- Lower learning rate
- Add more diverse data

---

## Dataset Preprocessing Tips

### Batch Captioning

Use Florence-2 or similar for auto-captioning:

```bash
# Using the built-in captioning tool
python -m toolkit.caption_tool --folder /path/to/images
```

### Image Preparation

1. **Crop to subject** - Remove unnecessary background
2. **Normalize sizes** - Not required (auto-resized) but helps
3. **Check quality** - Remove blurry or corrupted images
4. **Deduplicate** - Remove very similar images

### Organization

```
project/
├── raw_images/          # Original images
├── processed/           # Cropped/cleaned images
│   ├── image001.jpg
│   ├── image001.txt
│   └── ...
├── regularization/      # Regularization images
└── config/
    └── training.yaml
```
