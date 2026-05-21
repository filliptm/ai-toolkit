# AI-Toolkit Documentation

This folder documents the current AI-Toolkit codebase: a Python diffusion/audio training toolkit with a Next.js web UI, SQLite-backed job queue, built-in extension processes, and CLI entry points.

## Table of Contents

1. [Architecture Overview](./architecture.md) - Runtime layout, job dispatch, extension loading, model registry, and UI worker flow
2. [Configuration Guide](./configuration.md) - YAML/JSON configuration reference
3. [Training Guide](./training.md) - Common LoRA, fine-tune, textual inversion, and slider workflows
4. [Dataset Preparation](./datasets.md) - Image, video, control, mask, reference, and audio dataset options
5. [Supported Models](./models.md) - Model architecture identifiers exposed by the core registry and UI
6. [Extensions System](./extensions.md) - Built-in process UIDs and custom extension patterns
7. [UI Guide](./ui.md) - Next.js app, worker, queue, settings, and API routes
8. [Advanced Techniques](./advanced.md) - Advanced training options and experiments
9. [Troubleshooting](./troubleshooting.md) - Common installation, training, model, dataset, checkpoint, and UI issues
10. [Wan VACE Training](./wan_vace.md) - Wan VACE edit, outpaint, and video conditioning workflows

## Quick Start

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python -m venv venv
source venv/bin/activate  # Linux/macOS
# Windows: .\venv\Scripts\activate

# Install PyTorch first, then the toolkit requirements.
pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Run one or more config files from the CLI.
python run.py config/examples/train_lora_flux_24gb.yaml

# Start the UI and queue worker in development mode.
cd ui
npm install
npm run update_db
npm run dev
```

The root `start.sh` and `start.bat` scripts also start the UI dev server after clearing the configured UI port. `npm run start` serves the production UI on port `8675` after `npm run build`.

## Project Structure

```text
ai-toolkit/
|-- run.py                    # CLI entry point for config-driven jobs
|-- toolkit/                  # Core config, model, data loading, training, saving, and utility code
|-- jobs/                     # Built-in job dispatch and base process classes
|-- extensions/               # User extension packages discovered at runtime
|-- extensions_built_in/      # Shipped trainer, dataset, generator, captioner, and model packages
|-- config/examples/          # Current example configs for image, video, audio, edit, and utility jobs
|-- scripts/                  # Dataset/model conversion and validation utilities
|-- ui/                       # Next.js 15 / React 19 UI, Prisma schema, and queue worker
|-- datasets/                 # Default dataset root used by the UI
|-- output/                   # Default training output root
|-- data/                     # Default data root
`-- docs/                     # This documentation
```
