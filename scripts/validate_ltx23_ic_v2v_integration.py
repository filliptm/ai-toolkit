import os
import sys
import tempfile
from types import SimpleNamespace

import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from toolkit.config_modules import DatasetConfig, ModelConfig
from toolkit.data_loader import AiToolkitDataset
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.prompt_utils import PromptEmbeds
from extensions_built_in.diffusion_models.ltx2.ltx2 import LTX2Model


TEST_ROOT = r"C:\AI\Git Projects\LTX-IC-Trainer\projects\test\datasets\test"
TARGET_DIR = os.path.join(TEST_ROOT, "target")
REFERENCE_DIR = os.path.join(TEST_ROOT, "reference")
REFERENCE_CACHE_DIR = os.path.join(TEST_ROOT, "cache_ref")


class FakeSD:
    is_xl = False
    is_vega = False
    is_ssd = False
    is_v3 = False
    is_auraflow = False
    is_flux = False
    is_transformer = True
    is_audio_model = False
    encode_control_in_text_embeddings = False
    use_raw_control_images = False
    te_padding_side = "left"
    text_embedding_space_version = "ltx2_v2"
    latent_space_version = "ltx2_v2"
    sample_rate = 48000
    vae = SimpleNamespace(config=SimpleNamespace(scale_factor_temporal=8))
    unet = SimpleNamespace(config=SimpleNamespace(temporal_compression_ratio=8))
    model_config = SimpleNamespace(arch="ltx2.3", latent_space_version="ltx2_v2", is_pixart_sigma=False)
    device = "cpu"
    device_torch = torch.device("cpu")
    torch_dtype = torch.float32

    def get_bucket_divisibility(self):
        return 32

    def set_device_state_preset(self, _preset):
        pass

    def restore_device_state(self):
        pass

    def encode_images(self, imgs):
        if imgs.dim() == 5:
            batch, frames, _channels, height, width = imgs.shape
            return torch.ones((batch, 4, frames, max(1, height // 8), max(1, width // 8)), dtype=self.torch_dtype)
        batch, _channels, height, width = imgs.shape
        return torch.ones((batch, 4, 1, max(1, height // 8), max(1, width // 8)), dtype=self.torch_dtype)


class FakeFileItem:
    def __init__(self, idx: int):
        self.path = f"target_{idx}.mp4"
        self.reference_path = f"reference_{idx}.mp4"
        self.dataset_config = SimpleNamespace(load_image_when_caching_latents=False)
        self.is_latent_cached = True
        self._encoded_latent = torch.full((4, 2, 4, 4), idx, dtype=torch.float32)
        self._cached_first_frame_latent = None
        self._cached_audio_latent = None
        self._cached_reference_latent = torch.full((4, 1, 4, 4), idx + 10, dtype=torch.float32)
        self.tensor = None
        self.control_tensor = None
        self.control_tensor_list = None
        self.inpaint_tensor = None
        self.clip_image_tensor = None
        self.mask_tensor = None
        self.unaugmented_tensor = None
        self.unconditional_tensor = None
        self.unconditional_latent = None
        self.clip_image_embeds = None
        self.clip_image_embeds_unconditional = None
        self.prompt_embeds = None
        self.prompt_embeds_unconditional = None
        self.audio_data = None
        self.audio_tensor = None
        self.extra_values = []
        self.num_frames = 33
        self.loss_multiplier = 1.0
        self.network_weight = 1.0
        self.is_reg = False
        self.caption = ""
        self.caption_short = ""

    def get_latent(self):
        return self._encoded_latent

    def cleanup(self):
        pass


class FakeModule:
    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, _device):
        self.device = torch.device("cpu")
        return self


class FakeRope:
    def prepare_video_coords(self, batch_size, num_frames, height, width, device, fps=24):
        return torch.zeros((batch_size, 3, num_frames * height * width, 2), device=device)

    def prepare_audio_coords(self, batch_size, audio_num_frames, device):
        return torch.zeros((batch_size, 1, audio_num_frames, 2), device=device)


class FakeTransformer:
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.rope = FakeRope()
        self.audio_rope = FakeRope()
        self.last_hidden_shape = None
        self.last_timestep = None

    def to(self, _device):
        self.device = torch.device("cpu")
        return self

    def __call__(self, **kwargs):
        self.last_hidden_shape = kwargs["hidden_states"].shape
        self.last_timestep = kwargs["timestep"].detach().clone()
        return kwargs["hidden_states"], kwargs["audio_hidden_states"]


class FakePipeline:
    transformer_spatial_patch_size = 1
    transformer_temporal_patch_size = 1
    audio_sampling_rate = 16000
    audio_hop_length = 160
    audio_vae_temporal_compression_ratio = 1
    audio_vae = SimpleNamespace(config=SimpleNamespace(mel_bins=8, latent_channels=2))

    @staticmethod
    def _pack_latents(latents, patch_size=1, patch_size_t=1):
        batch, channels, frames, height, width = latents.shape
        return latents.permute(0, 2, 3, 4, 1).reshape(batch, frames * height * width, channels)

    @staticmethod
    def _unpack_latents(latents, num_frames, height, width, patch_size=1, patch_size_t=1):
        batch, _seq, channels = latents.shape
        return latents.reshape(batch, num_frames, height, width, channels).permute(0, 4, 1, 2, 3)

    def prepare_audio_latents(self, batch_size, num_channels_latents, audio_latent_length, num_mel_bins, **_kwargs):
        return torch.zeros((batch_size, audio_latent_length, num_channels_latents), dtype=torch.float32)


class FakeConnectors:
    device = torch.device("cpu")

    def to(self, _device):
        return self

    def __call__(self, text_embeds, attention_mask, padding_side="left"):
        return text_embeds, text_embeds, attention_mask


def count_media_files(root: str):
    exts = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".wmv", ".m4v", ".flv"}
    count = 0
    for current_root, _dirs, files in os.walk(root):
        if os.path.basename(current_root) in {"cache", "cache_ref", ".thumbs"}:
            continue
        count += sum(1 for file in files if os.path.splitext(file)[1].lower() in exts)
    return count


def validate_dataset_pairing():
    cfg = DatasetConfig(
        folder_path=TARGET_DIR,
        reference_path=REFERENCE_DIR,
        reference_cache_path=REFERENCE_CACHE_DIR,
        require_reference=True,
        cache_latents_to_disk=False,
        caption_ext="txt",
        resolution=720,
        buckets=False,
        num_frames=81,
        fps=24,
        do_audio=False,
        do_i2v=False,
    )
    dataset = AiToolkitDataset(cfg, batch_size=1, sd=FakeSD())
    expected = count_media_files(TARGET_DIR)
    matched = sum(1 for item in dataset.file_list if item.reference_path)
    assert len(dataset.file_list) == expected, f"Expected {expected} targets, got {len(dataset.file_list)}"
    assert matched == expected, f"Expected all targets to have references, got {matched}/{expected}"
    assert all(os.path.exists(item.reference_path) for item in dataset.file_list), "Reference path missing on disk"
    print(f"dataset_pairing_ok targets={expected} matched={matched}")


def validate_batch_reference_latents():
    batch = DataLoaderBatchDTO(file_items=[FakeFileItem(1), FakeFileItem(2)])
    assert batch.latents.shape == (2, 4, 2, 4, 4), batch.latents.shape
    assert batch.reference_latents.shape == (2, 4, 1, 4, 4), batch.reference_latents.shape
    assert torch.equal(batch.reference_latents[0], torch.full((4, 1, 4, 4), 11.0))
    assert torch.equal(batch.reference_latents[1], torch.full((4, 1, 4, 4), 12.0))
    batch.cleanup()
    print("batch_reference_latents_ok")


def validate_reference_latent_cache():
    with tempfile.TemporaryDirectory() as temp_dir:
        target_dir = os.path.join(temp_dir, "target")
        reference_dir = os.path.join(temp_dir, "reference")
        reference_cache_dir = os.path.join(temp_dir, "cache_ref")
        os.makedirs(target_dir)
        os.makedirs(reference_dir)
        Image.new("RGB", (64, 64), "white").save(os.path.join(target_dir, "sample.png"))
        Image.new("RGB", (64, 64), "black").save(os.path.join(reference_dir, "sample.png"))

        cfg = DatasetConfig(
            folder_path=target_dir,
            reference_path=reference_dir,
            reference_cache_path=reference_cache_dir,
            require_reference=True,
            cache_latents_to_disk=True,
            caption_ext="txt",
            resolution=64,
            buckets=True,
            num_frames=1,
        )
        dataset = AiToolkitDataset(cfg, batch_size=1, sd=FakeSD())
        item = dataset.file_list[0]
        assert item.is_latent_cached
        assert item.get_reference_latent_path() is not None
        assert os.path.exists(item.get_reference_latent_path()), item.get_reference_latent_path()
        batch = DataLoaderBatchDTO(file_items=[item])
        assert batch.reference_latents is not None
        assert batch.reference_latents.shape[0] == 1
        batch.cleanup()
        print("reference_latent_cache_ok")


def validate_arch_tag_normalization():
    cfg = ModelConfig(arch="ltx2.3:ic_v2v", name_or_path="dummy", model_kwargs={"ic_lora_strategy": "v2v"})
    assert cfg.arch == "ltx2.3", cfg.arch
    assert cfg.model_kwargs["ic_lora_strategy"] == "v2v"
    print("arch_tag_normalization_ok")


def validate_ltx2_v2v_forward_shapes():
    model = object.__new__(LTX2Model)
    model.model = FakeModule()
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.pipeline = FakePipeline()
    model.transformer = FakeTransformer()
    model.pipeline.connectors = FakeConnectors()
    model.model_config = SimpleNamespace(model_kwargs={"ic_lora_strategy": "v2v"})
    model.ltx_version = "2.3"

    batch = SimpleNamespace(
        reference_latents=torch.ones((2, 4, 1, 4, 5), dtype=torch.float32),
        audio_latents=None,
        audio_tensor=None,
        audio_data=None,
        audio_target=None,
        audio_pred=None,
        first_frame_latents=None,
        tensor=None,
        num_frames=3,
        dataset_config=SimpleNamespace(do_i2v=False, fps=24, reference_downscale=1),
    )
    embeds = PromptEmbeds(
        torch.zeros((2, 4, 8), dtype=torch.float32),
        attention_mask=torch.ones((2, 4), dtype=torch.float32),
    )
    target_latents = torch.randn((2, 4, 3, 4, 5), dtype=torch.float32)
    output = LTX2Model.get_noise_prediction(
        model,
        target_latents,
        torch.tensor([0.25, 0.75], dtype=torch.float32),
        embeds,
        batch=batch,
    )
    assert output.shape == target_latents.shape, output.shape
    assert model.transformer.last_hidden_shape[1] == 80, model.transformer.last_hidden_shape
    assert torch.all(model.transformer.last_timestep[:, :20] == 0), model.transformer.last_timestep[:, :20]
    print("ltx2_v2v_forward_shapes_ok")


if __name__ == "__main__":
    validate_dataset_pairing()
    validate_reference_latent_cache()
    validate_batch_reference_latents()
    validate_arch_tag_normalization()
    validate_ltx2_v2v_forward_shapes()
