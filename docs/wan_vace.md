# Wan VACE Training

Wan VACE is the Wan edit/conditioning architecture. It is different from Wan
T2V and Wan I2V: the transformer receives the denoising latents and a separate
VACE conditioning stream. In AI Toolkit this is exposed as `arch: wan21_vace`.

## A/B Edit Dataset

Use the target/output image as the normal dataset image and the source/input
image as `control_path`.

```text
datasets/my_edit/target/
  0001.png
  0001.txt
  0002.png
  0002.txt

datasets/my_edit/source/
  0001.png
  0002.png
```

The target and source filenames must match by basename. Captions are read from
the target folder and should describe the edit instruction.

```yaml
datasets:
  - folder_path: "datasets/my_edit/target"
    control_path: "datasets/my_edit/source"
    caption_ext: "txt"
    num_frames: 1
    resolution: [512, 768]
```

## Masks

Masks are optional. White means editable/reactive, black means
preserved/inactive. If `mask_path` is omitted, `model.model_kwargs.default_mask:
full` uses a full white mask for full-frame A/B edits.

```yaml
datasets:
  - folder_path: "datasets/my_edit/target"
    control_path: "datasets/my_edit/source"
    mask_path: "datasets/my_edit/masks"
```

Optional reference images can be paired by basename with `reference_path`.

```yaml
datasets:
  - folder_path: "datasets/my_edit/target"
    control_path: "datasets/my_edit/source"
    reference_path: "datasets/my_edit/reference"
```

## Sampling

Wan VACE samples require `ctrl_img`. `mask_img` is optional.

```yaml
sample:
  samples:
    - prompt: "apply the edit"
      ctrl_img: "datasets/my_edit/source/0001.png"
      mask_img: "datasets/my_edit/masks/0001.png"
```

`reference_img` is also supported for sample configs when using VACE reference
conditioning.

## ComfyUI LoRA Format

Wan VACE LoRAs are saved with ComfyUI-compatible VACE projection names by
default. The VACE projection keys are written as `before_proj` and `after_proj`
rather than Diffusers' internal `proj_in` and `proj_out` names.

## Recommended Config

Start from:

```text
config/examples/train_lora_wan_vace_edit_1b.yaml
config/examples/train_lora_wan_vace_edit_14b.yaml
```

For video targets, start from:

```text
config/examples/train_lora_wan_vace_video_14b.yaml
```

## Notes

- Train LoRA first. Full fine-tuning VACE is possible architecturally but much
  heavier.
- `batch_size: 1` is the safest setting for 14B.
- `cache_text_embeddings: true` is recommended to keep VRAM lower.
- Source-video pairing uses matching basenames in `control_path`, the same as
  image A/B pairing.
