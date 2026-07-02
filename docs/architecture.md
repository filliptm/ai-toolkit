# Architecture Overview

AI-Toolkit has two runtime surfaces:

- A Python CLI (`run.py`) that loads YAML/JSON configs and dispatches jobs.
- A Next.js UI (`ui/`) that stores jobs/settings in SQLite and launches the same Python CLI through a queue worker.

## Runtime Flow

```text
Config file or UI job JSON
        |
        v
run.py
        |
        v
toolkit/job.py:get_job()
        |
        +-- job: extract   -> jobs/ExtractJob.py
        +-- job: train     -> jobs/TrainJob.py
        +-- job: mod       -> jobs/ModJob.py
        +-- job: generate  -> jobs/GenerateJob.py
        `-- job: extension -> jobs/ExtensionJob.py
                                |
                                v
                       toolkit/extension.py
                       scans extensions/ and extensions_built_in/
                                |
                                v
                       Process class selected by config.process[].type
```

`extension` is the normal path for current training configs. The UI default job uses `process[0].type: diffusion_trainer`.

## Core Components

| Area | Files | Responsibility |
|------|-------|----------------|
| CLI entry | `run.py` | Parses config paths, `--recover`, `--name`, and `--log`; runs jobs sequentially |
| Config loading | `toolkit/config.py`, `toolkit/config_modules.py` | Reads YAML/JSON, resolves env vars/name tags, constructs typed config objects |
| Job dispatch | `toolkit/job.py`, `jobs/*.py` | Selects job class from `job` and loads process classes |
| Processes | `jobs/process/*`, `extensions_built_in/*` | Training, generation, extraction, modification, captioning, dataset tools, and custom workflows |
| Model loading | `toolkit/stable_diffusion_model.py`, `toolkit/models/*`, `extensions_built_in/*/AI_TOOLKIT_MODELS` | Core SD-style loading plus custom model subclasses |
| Data loading | `toolkit/data_loader.py`, `toolkit/dataloader_mixins.py` | Image/video/audio items, captions, buckets, masks, controls, references, cache behavior |
| Optimization | `toolkit/optimizer.py`, `toolkit/optimizers/*` | Adam/AdamW, bitsandbytes 8-bit, Prodigy, Adafactor, Automagic, Automagic2 |
| Saving | `toolkit/saving.py`, trainer classes | Safetensors/diffusers save formats, step checkpoints, hub push metadata |
| UI | `ui/src`, `ui/cron`, `ui/prisma` | Job builder, queue, settings, monitoring, API routes |

## Job Types

`toolkit/job.py` currently supports:

| `job` | Job class | Process source |
|-------|-----------|----------------|
| `extension` | `ExtensionJob` | All registered extension UIDs |
| `train` | `TrainJob` | Legacy built-in train processes such as `vae`, `slider`, `rescale_sd`, `esrgan` |
| `extract` | `ExtractJob` | `lora`, `locon` extraction |
| `mod` | `ModJob` | `rescale_lora` |
| `generate` | `GenerateJob` | `to_folder` |

`MergeJob` exists in `jobs/MergeJob.py`, but `toolkit/job.py` does not currently dispatch a `merge` job type.

## Extension Discovery

`toolkit/extension.py` scans:

```text
extensions/
extensions_built_in/
```

Each package can export `AI_TOOLKIT_EXTENSIONS`. `ExtensionJob` builds a dictionary where each extension `uid` maps to the process class returned by `get_process()`.

Current major built-in process groups:

- `sd_trainer`: `sd_trainer`, `textual_inversion_trainer`, `ui_trainer`, `diffusion_trainer`
- `concept_slider`, `image_reference_slider_trainer`, `ultimate_slider_trainer`, `concept_replacer`
- `dataset_tools`, `sync_from_collection`, `super_tagger`
- `AceStepCaptioner`, `Qwen3VLCaptioner`
- `reference_generator`, `pure_lora_generator`, `batch_img2img`

## Process Hierarchy

```text
BaseProcess
|-- BaseExtractProcess
|-- BaseMergeProcess
|-- BaseExtensionProcess
`-- BaseTrainProcess
    `-- BaseSDTrainProcess
        `-- SDTrainer
            |-- UITrainer
            `-- DiffusionTrainer
```

Specialized extension trainers can inherit from these classes or directly from `BaseExtensionProcess`.

## Model Registry

AI-Toolkit has both legacy/core model loading and custom model packages.

Custom model discovery is handled by `toolkit/util/get_model.py`, which scans `extensions/` and `extensions_built_in/` for `AI_TOOLKIT_MODELS`. Current shipped model packages include:

- `extensions_built_in/diffusion_models`
- `extensions_built_in/audio_models`
- `extensions_built_in/flex2`

The current custom registry includes Chroma, Zeta Chroma, HiDream, HiDream E1/O1, F-Lite, OmniGen2, FLUX Kontext, Wan 2.2, Wan VACE, Qwen-Image variants, FLUX.2 variants, Z-Image, LTX-2/LTX-2.3, ERNIE-Image, Nucleus-Image, ACE-Step, and Flex.2. See [Supported Models](./models.md) for architecture IDs.

## Training Data Flow

```text
DatasetConfig
    |
    v
AiToolkitDataset / data loader mixins
    |
    +-- image/video/audio paths
    +-- captions and trigger handling
    +-- masks, control images, reference images
    +-- buckets, repeats, augmentations, latent/text caches
    |
    v
BaseSDTrainProcess / SDTrainer
    |
    +-- encode latents/text/audio/reference inputs as needed
    +-- sample timesteps/noise
    +-- apply LoRA/adapter/model-specific behavior
    +-- compute configured loss
    +-- optimizer step and scheduler update
    |
    v
checkpoint, samples, logs, UI progress
```

## UI Architecture

The UI stack is:

- Next.js 15 with App Router
- React 19
- TypeScript
- Tailwind CSS
- Prisma 6 with SQLite
- A Node queue worker in `ui/cron/worker.ts`

The worker launches `python run.py <job_config> --log <log_path>` with a detached process. It writes `.job_config.json`, rotates `log.txt`, stores the PID in SQLite and `pid.txt`, and sets UI-specific environment variables.

## Outputs

Training output is written under the process `training_folder` and job name, commonly `output/<name>/`. Typical files include:

```text
output/<name>/
|-- <name>.safetensors or diffusers checkpoint folders
|-- <name>_<step>.safetensors or step folders
|-- samples/
|-- log.txt
|-- logs/
|-- pid.txt
`-- .job_config.json
```

The UI reads this output tree for logs, samples, files, and progress while the database tracks queue state and job metadata.

## Memory Strategies

Current memory controls are spread across model, training, network, and dataset config:

- `model.quantize`, `model.quantize_te`, `model.qtype`, `model.qtype_te`
- `model.low_vram`
- `model.layer_offloading`
- `model.layer_offloading_transformer_percent`
- `model.layer_offloading_text_encoder_percent`
- `train.gradient_checkpointing`
- `train.unload_text_encoder`
- `train.cache_text_embeddings`
- `datasets[].cache_latents_to_disk`
- `network.layer_offloading`

Large video/audio/instruction models commonly combine quantization, low VRAM mode, text encoder unloading, and disk caches.
