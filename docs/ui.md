# UI Guide

The UI is a Next.js 15 / React 19 application in `ui/`. It manages jobs, datasets, settings, sample previews, system status, and a SQLite-backed queue. The queue worker runs beside the UI in development and production scripts.

## Starting The UI

```bash
cd ui
npm install
npm run update_db
npm run dev
```

`npm run dev` starts two processes with `concurrently`:

- `ts-node-dev --project tsconfig.worker.json --respawn --watch cron --transpile-only cron/worker.ts`
- `next dev`

The root `start.sh` and `start.bat` scripts clear `UI_PORT`/`3000`, enter `ui/`, and run `npm run dev`. Production startup is:

```bash
cd ui
npm install
npm run update_db
npm run build
npm run start
```

`npm run start` launches the compiled worker and `next start --port 8675`.

## Storage And Settings

The UI uses Prisma with SQLite at `aitk_db.db` through `ui/prisma/schema.prisma`.

| Table | Purpose |
|-------|---------|
| `Settings` | Persistent UI settings such as `TRAINING_FOLDER`, `DATASETS_FOLDER`, `DATA_ROOT`, and `HF_TOKEN` |
| `Queue` | Per-GPU queue state keyed by `gpu_ids` |
| `Job` | Job config JSON, status, PID, queue position, progress, and metadata |

Default paths come from `ui/src/paths.ts` and `ui/cron/paths.ts`:

| Setting | Default |
|---------|---------|
| `TRAINING_FOLDER` | `<repo>/output` |
| `DATASETS_FOLDER` | `<repo>/datasets` |
| `DATA_ROOT` | `<repo>/data` |

The worker injects `AITK_JOB_ID`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES`, `IS_AI_TOOLKIT_UI=1`, and `HF_TOKEN` when launching `python run.py`.

## Pages And Features

| Route | Purpose |
|-------|---------|
| `/` | Main app entry |
| `/dashboard` | System/job overview |
| `/jobs` | Job list |
| `/jobs/new` | Job builder |
| `/jobs/[jobID]` | Job detail, logs, loss graph, samples, and controls |
| `/datasets` | Dataset list and management |
| `/datasets/[datasetName]` | Dataset image/caption management |
| `/settings` | Settings persisted to SQLite |

The job builder writes `job: extension` configs with `process[0].type: diffusion_trainer` by default. The UI also exposes a Concept Slider job type using `process[0].type: concept_slider`.

## Model Presets

UI presets live in `ui/src/app/jobs/new/options.ts`. They define:

- `modelArchs`: model labels, architecture IDs, defaults, disabled fields, extra form sections, and optional Accuracy Recovery Adapters
- `jobTypeOptions`: currently `diffusion_trainer` and `concept_slider`
- `quantizationOptions`: `qfloat8`, `uint7`, `uint6`, `uint5`, `uint4`, `uint3`, `uint2`

The default job config lives in `ui/src/app/jobs/new/jobConfig.ts`. `migrateJobConfig()` upgrades old prompt arrays, changes `ui_trainer` to `diffusion_trainer`, and migrates `auto_memory` to `layer_offloading`.

## API Routes

Current route handlers are:

### Jobs

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/api/jobs` | List jobs; supports `id`, `job_ref`, and `job_type` query filters |
| `POST` | `/api/jobs` | Create/update a job from a JSON body |
| `GET` | `/api/jobs/[jobID]/start` | Mark a job running and launch it through the worker action |
| `GET` | `/api/jobs/[jobID]/stop` | Request stop |
| `GET` | `/api/jobs/[jobID]/mark_stopped` | Mark stopped |
| `GET` | `/api/jobs/[jobID]/delete` | Delete job |
| `GET` | `/api/jobs/[jobID]/log` | Read training log |
| `GET` | `/api/jobs/[jobID]/loss` | Read logged loss/progress values; supports `key`, `limit`, `since_step`, and `stride` |
| `GET` | `/api/jobs/[jobID]/samples` | List generated sample media |
| `GET` | `/api/jobs/[jobID]/files` | List job files |

### Queue

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/api/queue` | List queues/jobs |
| `GET` | `/api/queue/[queueID]/start` | Start queue processing |
| `GET` | `/api/queue/[queueID]/stop` | Stop queue processing |

### Datasets And Media

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/api/datasets/list` | List datasets under the configured dataset root |
| `POST` | `/api/datasets/create` | Create dataset |
| `POST` | `/api/datasets/delete` | Delete dataset |
| `POST` | `/api/datasets/listImages` | List images for a dataset |
| `POST` | `/api/datasets/upload` | Upload files to a dataset |
| `POST` | `/api/datasets/referencePairs` | Build/list reference pairs |
| `POST` | `/api/datasets/outpaint` | Build outpaint training assets |
| `GET` | `/api/img/[...imagePath]` | Serve images/media with range support |
| `POST` | `/api/img/upload` | Upload an image |
| `POST` | `/api/img/delete` | Delete an image |
| `POST` | `/api/img/caption` | Save an image caption |
| `POST` | `/api/caption/get` | Generate/read caption data |
| `GET` | `/api/audio/art/[...audioPath]` | Serve embedded/generated audio artwork |
| `GET` | `/api/files/[...filePath]` | Serve files with range support |
| `POST` | `/api/zip` | Create zip output from requested files |

### System And Utilities

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/api/gpu` | GPU status from `nvidia-smi`, plus macOS handling |
| `GET` | `/api/cpu` | CPU and memory status from `systeminformation` |
| `GET` | `/api/settings` | Read persisted settings |
| `POST` | `/api/settings` | Save settings and flush settings cache |
| `GET` | `/api/auth` | Auth/session check |
| `POST` | `/api/scripts` | Run allowed UI script actions |

## Queue Worker

`ui/cron/worker.ts` runs every second and calls `processQueue()`. The queue logic:

1. Reads all `Queue` rows ordered by `id`.
2. If a queue is stopped, running jobs on that queue's `gpu_ids` are marked to return to queue.
3. If a queue is running and no job is currently `running` or `stopping` for those GPUs, the next `queued` job is started.
4. If no queued job exists for that queue, the queue is stopped.

`startJob()` writes `.job_config.json`, rotates `log.txt` into `logs/`, records `pid.txt`, and launches `run.py` detached so training can continue if the UI process exits.

## Frontend Structure

```text
ui/
|-- cron/                 # Queue worker, Prisma client, and process launch actions
|-- prisma/schema.prisma  # SQLite schema
|-- src/app/              # Next.js app routes and API routes
|-- src/components/       # Shared UI components
|-- src/hooks/            # Data-fetching hooks
|-- src/helpers/          # Job/sample/caption helpers
|-- src/utils/            # API, queue, job, script utilities
|-- src/types.ts          # Shared TypeScript types
`-- public/               # Static assets
```

Important hooks include `useJobsList`, `useJob`, `useJobByRef`, `useQueueList`, `useJobLog`, `useJobLossLog`, `useSampleImages`, `useDatasetList`, `useFilesList`, `useGPUInfo`, and `useCPUInfo`.

## Troubleshooting

```bash
cd ui
npm run update_db
npm run dev
```

If the UI cannot see jobs, verify the `TRAINING_FOLDER` setting and that `aitk_db.db` is reachable from the repo root. If jobs launch but do not train, check the job folder's `log.txt` and `pid.txt` under the configured training folder.
