# Extensions System

AI-Toolkit discovers extension packages from both `extensions/` and `extensions_built_in/`. Each package can expose `AI_TOOLKIT_EXTENSIONS`, a list of classes derived from `toolkit.extension.Extension`. `ExtensionJob` builds a process map from each extension's `uid`, then instantiates the returned process class for matching `config.process[].type` values.

## Built-In Process UIDs

These UIDs are currently registered by shipped extension packages.

| UID | Package | Process | Purpose |
|-----|---------|---------|---------|
| `sd_trainer` | `extensions_built_in/sd_trainer` | `SDTrainer` | Generic LoRA, LoKr, LoCon, DreamBooth/full fine-tune, sampling |
| `textual_inversion_trainer` | `extensions_built_in/sd_trainer` | `SDTrainer` | Backward-compatible alias for textual inversion configs |
| `ui_trainer` | `extensions_built_in/sd_trainer` | `UITrainer` | Legacy UI trainer |
| `diffusion_trainer` | `extensions_built_in/sd_trainer` | `DiffusionTrainer` | Current UI/API trainer; extends SD training with UI/job logging behavior |
| `concept_slider` | `extensions_built_in/concept_slider` | `ConceptSliderTrainer` | Prompt-pair concept slider training |
| `image_reference_slider_trainer` | `extensions_built_in/image_reference_slider_trainer` | `ImageReferenceSliderTrainerProcess` | Image-reference slider training |
| `ultimate_slider_trainer` | `extensions_built_in/ultimate_slider_trainer` | `UltimateSliderTrainerProcess` | Extended slider workflow |
| `concept_replacer` | `extensions_built_in/concept_replacer` | `ConceptReplacer` | Concept replacement workflow |
| `dataset_tools` | `extensions_built_in/dataset_tools` | `DatasetTools` | Dataset utility process |
| `sync_from_collection` | `extensions_built_in/dataset_tools` | `SyncFromCollection` | Sync a dataset from a collection |
| `super_tagger` | `extensions_built_in/dataset_tools` | `SuperTagger` | Caption/tag dataset items |
| `AceStepCaptioner` | `extensions_built_in/captioner` | `AceStepCaptioner` | Audio captioning |
| `Qwen3VLCaptioner` | `extensions_built_in/captioner` | `Qwen3VLCaptioner` | Vision-language captioning |
| `reference_generator` | `extensions_built_in/advanced_generator` | `ReferenceGenerator` | Reference image generation |
| `pure_lora_generator` | `extensions_built_in/advanced_generator` | `PureLoraGenerator` | LoRA generation utility |
| `batch_img2img` | `extensions_built_in/advanced_generator` | `Img2ImgGenerator` | Batch image-to-image generation |

Most training configs should use:

```yaml
job: extension
config:
  name: my_lora
  process:
    - type: diffusion_trainer
      # ...
```

## Model Extension Packages

Model packages use `AI_TOOLKIT_MODELS`, not `AI_TOOLKIT_EXTENSIONS`. They are discovered by `toolkit/util/get_model.py` and allow `model.arch` to resolve to a custom `BaseModel` subclass. Current model packages include `extensions_built_in/diffusion_models`, `extensions_built_in/audio_models`, and `extensions_built_in/flex2`.

## Extension Architecture

```text
extensions_built_in/
|-- sd_trainer/
|   |-- __init__.py        # AI_TOOLKIT_EXTENSIONS registration
|   |-- SDTrainer.py
|   |-- UITrainer.py
|   `-- DiffusionTrainer.py
|-- dataset_tools/
|-- captioner/
|-- advanced_generator/
`-- ...
```

Every process extension module must export `AI_TOOLKIT_EXTENSIONS`:

```python
from toolkit.extension import Extension

class MyExtension(Extension):
    uid = "my_trainer"
    name = "My Trainer"

    @classmethod
    def get_process(cls):
        from .MyProcess import MyProcess
        return MyProcess

AI_TOOLKIT_EXTENSIONS = [MyExtension]
```

## Creating A Custom Extension

Create a package under `extensions/`:

```text
extensions/my_extension/
|-- __init__.py
`-- MyProcess.py
```

`MyProcess.py` can inherit from `BaseExtensionProcess`, `SDTrainer`, or another existing process depending on how much behavior you need:

```python
from jobs.process import BaseExtensionProcess

class MyProcess(BaseExtensionProcess):
    def __init__(self, process_id, job, config):
        super().__init__(process_id, job, config)
        self.custom_value = self.get_conf("custom.value", default=42)

    def run(self):
        super().run()
        print(f"custom_value={self.custom_value}")
```

Register it in `__init__.py`:

```python
from toolkit.extension import Extension

class MyExtension(Extension):
    uid = "my_trainer"
    name = "My Trainer"

    @classmethod
    def get_process(cls):
        from .MyProcess import MyProcess
        return MyProcess

AI_TOOLKIT_EXTENSIONS = [MyExtension]
```

Use it in a config:

```yaml
job: extension
config:
  name: my_extension_run
  process:
    - type: my_trainer
      custom:
        value: 123
```

## Process Class Hierarchy

```text
BaseProcess
`-- BaseExtensionProcess
    `-- BaseTrainProcess
        `-- BaseSDTrainProcess
            `-- SDTrainer
                |-- UITrainer
                `-- DiffusionTrainer
```

Specialized trainers such as `ConceptSliderTrainer` typically inherit from `DiffusionTrainer` or related SD trainer classes.

## Configuration Access

`BaseProcess.get_conf()` supports nested keys:

```yaml
process:
  - type: my_trainer
    custom:
      nested:
        value: 42
```

```python
value = self.get_conf("custom.nested.value", default=0)
required = self.get_conf("custom.required", required=True)
```

## Useful Hooks

The SD trainer stack provides hooks such as:

| Hook | Purpose |
|------|---------|
| `before_model_load()` | Prepare state before model loading |
| `hook_before_train_loop()` | Initialize custom loss state, datasets, or logging before training |
| `hook_train_loop(batch)` | Override or augment the training step |
| `sample(step)` | Customize periodic sampling |
| `on_error(error)` | Handle cleanup or logging when a job fails |

Use the narrowest hook that fits the change. For pure custom jobs, inherit from `BaseExtensionProcess`; for training behavior, inherit from `SDTrainer` or `DiffusionTrainer`.
