# AI-Toolkit Documentation

This documentation provides a comprehensive reference for understanding and using the AI-Toolkit training software by Ostris.

## Table of Contents

1. [Architecture Overview](./architecture.md) - System design and component relationships
2. [Configuration Guide](./configuration.md) - Complete YAML/JSON configuration reference
3. [Training Guide](./training.md) - How to train models with different approaches
4. [Dataset Preparation](./datasets.md) - Dataset formats, structures, and best practices
5. [Supported Models](./models.md) - List of supported model architectures
6. [Extensions System](./extensions.md) - How to extend the toolkit
7. [UI Guide](./ui.md) - Web interface usage and API endpoints
8. [Advanced Techniques](./advanced.md) - Advanced training configurations and experiments
9. [Troubleshooting](./troubleshooting.md) - Common issues and solutions
10. [Wan VACE Training](./wan_vace.md) - A/B edit training with Wan VACE

## Quick Start

```bash
# Clone and setup
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Run training
python run.py config/your_config.yaml
```

## Project Structure Overview

```
ai-toolkit/
├── run.py                 # Main entry point for CLI training
├── config/                # Configuration files
│   └── examples/          # Example training configs
├── toolkit/               # Core training library
│   ├── config.py          # Configuration parsing
│   ├── config_modules.py  # Config dataclasses
│   ├── data_loader.py     # Dataset loading
│   ├── stable_diffusion_model.py  # SD model wrapper
│   └── ...
├── jobs/                  # Job types and processes
│   ├── TrainJob.py        # Training job handler
│   └── process/           # Training process implementations
├── extensions/            # Custom user extensions
├── extensions_built_in/   # Built-in extensions
│   ├── sd_trainer/        # Main SD training extension
│   └── ...
├── ui/                    # Next.js web interface
└── output/                # Training outputs (models, samples)
```
