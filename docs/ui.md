# UI Guide

AI-Toolkit includes a web-based user interface built with Next.js. This guide covers using and understanding the UI.

## Starting the UI

### Quick Start

```bash
cd ui
npm install
npm run dev
```

Or use the provided batch file:

```bash
./run-ui.bat
```

The UI will be available at `http://localhost:3000`.

### Configuration

The UI can be configured via environment variables or a `.env.local` file:

```bash
# ui/.env.local
PORT=3000
TRAINING_FOLDER=./output
DATABASE_PATH=./aitk_db.db
```

---

## UI Features

### Dashboard

The main dashboard shows:
- Running jobs
- Recent training runs
- GPU utilization
- System status

### Jobs Page

Create and manage training jobs:

1. **New Job** - Create training configuration
2. **Job List** - View all jobs
3. **Job Details** - Monitor running jobs

### Datasets Page

Manage training datasets:

1. **Upload Images** - Drag and drop images
2. **Add Captions** - Auto or manual captioning
3. **Preview** - View dataset with captions

---

## Creating a Training Job

### Step 1: Select Model Architecture

Choose from supported models:
- FLUX.1, FLUX.2, Flex.1/2
- SDXL, SD 1.5
- Wan 2.1/2.2, LTX-2
- And more...

### Step 2: Configure Training

| Setting | Description |
|---------|-------------|
| Name | Unique training run name |
| Trigger Word | Word to invoke your training |
| Steps | Total training steps |
| Learning Rate | Training speed |
| Batch Size | Images per step |
| Network Rank | LoRA capacity |

### Step 3: Add Dataset

1. Click "Add Dataset"
2. Select folder path
3. Configure resolution
4. Set caption settings

### Step 4: Configure Sampling

Add prompts to test during training:

1. Click "Add Prompt"
2. Enter test prompt with `[trigger]`
3. Set sample frequency

### Step 5: Start Training

Click "Start Training" to begin. Monitor:
- Loss graph
- Sample images
- Training progress

---

## API Endpoints

The UI exposes a REST API for programmatic access.

### Jobs API

#### List Jobs
```bash
GET /api/jobs

Response:
{
  "jobs": [
    {
      "id": "my_lora_v1",
      "status": "running",
      "step": 500,
      "total_steps": 2000
    }
  ]
}
```

#### Get Job Details
```bash
GET /api/jobs/{jobID}

Response:
{
  "id": "my_lora_v1",
  "status": "running",
  "step": 500,
  "total_steps": 2000,
  "loss": 0.05,
  "config": { ... }
}
```

#### Start Job
```bash
POST /api/jobs/{jobID}/start
```

#### Stop Job
```bash
POST /api/jobs/{jobID}/stop
```

#### Delete Job
```bash
DELETE /api/jobs/{jobID}/delete
```

#### Get Job Logs
```bash
GET /api/jobs/{jobID}/log

Response:
{
  "logs": [
    {"step": 100, "loss": 0.08, "lr": 0.0001},
    {"step": 200, "loss": 0.06, "lr": 0.0001}
  ]
}
```

#### Get Job Samples
```bash
GET /api/jobs/{jobID}/samples

Response:
{
  "samples": [
    {"step": 250, "prompt": "test prompt", "path": "/samples/..."}
  ]
}
```

#### Get Loss History
```bash
GET /api/jobs/{jobID}/loss

Response:
{
  "loss": [
    {"step": 100, "loss": 0.08},
    {"step": 200, "loss": 0.06}
  ]
}
```

### Datasets API

#### List Datasets
```bash
GET /api/datasets/list

Response:
{
  "datasets": [
    {"name": "my_dataset", "path": "/path/to/dataset", "count": 50}
  ]
}
```

#### Create Dataset
```bash
POST /api/datasets/create
Content-Type: application/json

{
  "name": "my_new_dataset",
  "path": "/path/to/images"
}
```

#### Delete Dataset
```bash
DELETE /api/datasets/delete

{
  "name": "my_dataset"
}
```

#### List Dataset Images
```bash
GET /api/datasets/listImages?dataset=my_dataset

Response:
{
  "images": [
    {"name": "image1.jpg", "caption": "a photo of..."}
  ]
}
```

### Image API

#### Upload Image
```bash
POST /api/img/upload
Content-Type: multipart/form-data

file: <image file>
dataset: "my_dataset"
```

#### Get Caption
```bash
GET /api/caption/get?image=/path/to/image.jpg

Response:
{
  "caption": "a photo of..."
}
```

#### Save Caption
```bash
POST /api/img/caption
Content-Type: application/json

{
  "image": "/path/to/image.jpg",
  "caption": "new caption"
}
```

### System API

#### GPU Status
```bash
GET /api/gpu

Response:
{
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA RTX 4090",
      "memory_used": 12000,
      "memory_total": 24576,
      "utilization": 85
    }
  ]
}
```

#### CPU Status
```bash
GET /api/cpu

Response:
{
  "cpu_percent": 45,
  "memory_percent": 60
}
```

### Queue API

#### List Queue
```bash
GET /api/queue

Response:
{
  "queue": [
    {"id": "job1", "position": 1},
    {"id": "job2", "position": 2}
  ]
}
```

#### Start Queue
```bash
POST /api/queue/{queueID}/start
```

#### Stop Queue
```bash
POST /api/queue/{queueID}/stop
```

---

## UI Architecture

### Frontend Stack

- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **SWR** - Data fetching

### Directory Structure

```
ui/
├── src/
│   ├── app/               # Next.js app router pages
│   │   ├── api/           # API routes
│   │   ├── dashboard/     # Dashboard page
│   │   ├── datasets/      # Dataset pages
│   │   ├── jobs/          # Job pages
│   │   └── settings/      # Settings page
│   ├── components/        # React components
│   ├── hooks/             # Custom React hooks
│   ├── utils/             # Utility functions
│   └── types.ts           # TypeScript types
└── public/                # Static assets
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `JobOverview` | Job status and controls |
| `JobLossGraph` | Loss visualization |
| `SampleImages` | Sample image gallery |
| `GPUMonitor` | GPU usage display |
| `DatasetImageCard` | Dataset image preview |

### Data Fetching Hooks

| Hook | Purpose |
|------|---------|
| `useJob` | Fetch single job data |
| `useJobsList` | Fetch all jobs |
| `useJobLog` | Fetch job logs |
| `useJobLossLog` | Fetch loss history |
| `useSampleImages` | Fetch sample images |
| `useDatasetList` | Fetch datasets |
| `useGPUInfo` | Fetch GPU status |

---

## Job Configuration (UI)

The UI generates configurations compatible with `diffusion_trainer`:

```typescript
// ui/src/app/jobs/new/jobConfig.ts

export const defaultJobConfig: JobConfig = {
  job: 'extension',
  config: {
    name: 'my_first_lora_v1',
    process: [{
      type: 'diffusion_trainer',
      training_folder: 'output',
      sqlite_db_path: './aitk_db.db',
      device: 'cuda',
      trigger_word: null,
      network: {
        type: 'lora',
        linear: 32,
        linear_alpha: 32,
      },
      // ... rest of config
    }]
  }
};
```

### Model Architectures

Available models are defined in `options.ts`:

```typescript
export const modelArchs: ModelArch[] = [
  {
    name: 'flux',
    label: 'FLUX.1',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['black-forest-labs/FLUX.1-dev', ''],
      'config.process[0].model.quantize': [true, false],
      // ...
    },
    disableSections: ['network.conv'],
  },
  // ... more models
];
```

---

## Customizing the UI

### Adding New Model Support

1. Edit `ui/src/app/jobs/new/options.ts`:

```typescript
{
  name: 'my_model',
  label: 'My Custom Model',
  group: 'image',
  defaults: {
    'config.process[0].model.name_or_path': ['org/my-model', ''],
    'config.process[0].model.arch': ['my_arch', null],
  },
}
```

2. Add any model-specific UI sections as needed.

### Adding New Settings

1. Update types in `ui/src/types.ts`
2. Add form inputs in job creation page
3. Handle in API routes

---

## Troubleshooting

### UI Won't Start

```bash
# Check Node.js version
node --version  # Should be 18+

# Clear cache and reinstall
rm -rf node_modules .next
npm install
npm run dev
```

### Jobs Not Showing

- Ensure `training_folder` matches
- Check database path
- Verify file permissions

### API Errors

Check console for errors:
```bash
npm run dev
# Check terminal output
```

### GPU Not Detected

- Ensure CUDA is installed
- Check `nvidia-smi` works
- Verify PyTorch sees GPU:
```python
import torch
print(torch.cuda.is_available())
```
