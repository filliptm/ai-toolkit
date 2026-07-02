import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions_built_in.diffusion_models.wan_vace.wan_vace_model import WanVACEModel
from toolkit.config_modules import ModelConfig
from toolkit.prompt_utils import PromptEmbeds


class _FakeLatentDist:
    def __init__(self, tensor):
        self.tensor = tensor

    def mode(self):
        batch, _channels, frames, height, width = self.tensor.shape
        return torch.zeros(
            (batch, 16, (frames - 1) // 4 + 1, height // 8, width // 8),
            dtype=self.tensor.dtype,
            device=self.tensor.device,
        )


class _FakeEncodeOutput:
    def __init__(self, tensor):
        self.latent_dist = _FakeLatentDist(tensor)


class _FakeVAE:
    temperal_downsample = [False, True, True]
    device = torch.device("cpu")
    dtype = torch.float32
    config = SimpleNamespace(
        z_dim=16,
        scale_factor_spatial=8,
        latents_mean=[0.0] * 16,
        latents_std=[1.0] * 16,
    )

    def to(self, *args, **kwargs):
        return self

    def encode(self, tensor):
        return _FakeEncodeOutput(tensor)


class _FakeVACETransformer(torch.nn.Module):
    config = SimpleNamespace(vace_layers=[0, 5, 10, 15, 20, 25, 30, 35], patch_size=(1, 2, 2))

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))

    def __call__(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        control_hidden_states,
        control_hidden_states_scale,
        return_dict=False,
        **kwargs,
    ):
        assert hidden_states.shape == (2, 16, 1, 8, 8)
        assert control_hidden_states.shape == (2, 96, 1, 8, 8)
        assert control_hidden_states_scale.shape == (8,)
        return (hidden_states * self.scale,)


def _fake_model():
    model = WanVACEModel.__new__(WanVACEModel)
    model.vae = _FakeVAE()
    model.model = _FakeVACETransformer()
    model.vae_device_torch = torch.device("cpu")
    model.vae_torch_dtype = torch.float32
    model.device_torch = torch.device("cpu")
    model.torch_dtype = torch.float32
    model.vace_default_mask = "full"
    model.vace_require_mask = False
    model.vace_conditioning_scale = [1.0] * 8
    return model


def test_vace_conditioning_latent_shape():
    model = _fake_model()
    batch = SimpleNamespace(
        control_tensor=torch.rand(2, 3, 64, 64),
        mask_tensor=None,
        clip_image_tensor=None,
    )
    latent_model_input = torch.randn(2, 16, 1, 8, 8)

    conditioning = model._build_training_conditioning(batch, latent_model_input)

    assert conditioning.shape == (2, 96, 1, 8, 8)


def test_vace_reference_conditioning_prepends_reference_latent():
    model = _fake_model()
    batch = SimpleNamespace(
        control_tensor=torch.rand(2, 3, 64, 64),
        mask_tensor=torch.ones(2, 1, 64, 64),
        clip_image_tensor=torch.rand(2, 3, 64, 64),
    )
    latent_model_input = torch.randn(2, 16, 1, 8, 8)

    conditioning = model._build_training_conditioning(batch, latent_model_input)

    assert conditioning.shape == (2, 96, 2, 8, 8)


def test_vace_noise_prediction_receives_control_stream():
    model = _fake_model()
    batch = SimpleNamespace(
        control_tensor=torch.rand(2, 3, 64, 64),
        mask_tensor=torch.ones(2, 1, 64, 64),
        clip_image_tensor=None,
    )
    text_embeddings = PromptEmbeds(torch.randn(2, 4, 32))

    prediction = model.get_noise_prediction(
        torch.randn(2, 16, 1, 8, 8),
        torch.ones(2),
        text_embeddings,
        batch=batch,
    )

    assert prediction.shape == (2, 16, 1, 8, 8)


def test_vace_synthetic_training_step_updates_parameter():
    model = _fake_model()
    optimizer = torch.optim.SGD(model.model.parameters(), lr=0.1)
    batch = SimpleNamespace(
        control_tensor=torch.rand(2, 3, 64, 64),
        mask_tensor=torch.ones(2, 1, 64, 64),
        clip_image_tensor=None,
    )
    text_embeddings = PromptEmbeds(torch.randn(2, 4, 32))
    noisy_latents = torch.randn(2, 16, 1, 8, 8)
    target = torch.zeros_like(noisy_latents)
    before = model.model.scale.detach().clone()

    prediction = model.get_noise_prediction(noisy_latents, torch.ones(2), text_embeddings, batch=batch)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    optimizer.step()

    assert model.model.scale.detach() != before


def test_vace_lora_save_uses_comfy_projection_names():
    state_dict = {
        "transformer.vace_blocks.0.proj_in.lora_A.weight": torch.zeros(1, 1),
        "transformer.vace_blocks.0.proj_out.lora_B.weight": torch.zeros(1, 1),
        "transformer.vace_blocks.0.attn1.to_q.lora_A.weight": torch.zeros(1, 1),
    }

    saved = WanVACEModel.convert_lora_weights_before_save(None, state_dict)

    assert "diffusion_model.vace_blocks.0.before_proj.lora_A.weight" in saved
    assert "diffusion_model.vace_blocks.0.after_proj.lora_B.weight" in saved
    assert "diffusion_model.vace_blocks.0.proj_in.lora_A.weight" not in saved
    assert "diffusion_model.vace_blocks.0.proj_out.lora_B.weight" not in saved

    loaded = WanVACEModel.convert_lora_weights_before_load(None, saved)

    assert "transformer.vace_blocks.0.proj_in.lora_A.weight" in loaded
    assert "transformer.vace_blocks.0.proj_out.lora_B.weight" in loaded
    assert "transformer.vace_blocks.0.attn1.to_q.lora_A.weight" in loaded


def test_vace_tagged_arch_normalizes_to_model_arch():
    config = ModelConfig(arch="wan21_vace:14b", name_or_path="Wan-AI/Wan2.1-VACE-14B-diffusers")

    assert config.arch == "wan21_vace"


if __name__ == "__main__":
    test_vace_conditioning_latent_shape()
    test_vace_reference_conditioning_prepends_reference_latent()
    test_vace_noise_prediction_receives_control_stream()
    test_vace_synthetic_training_step_updates_parameter()
    test_vace_lora_save_uses_comfy_projection_names()
    test_vace_tagged_arch_normalizes_to_model_arch()
    print("Wan VACE conditioning tests passed")
