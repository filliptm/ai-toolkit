import os
from typing import List, Optional

import huggingface_hub
import torch
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.memory_management import MemoryManager
from safetensors.torch import load_file
from diffusers.utils.torch_utils import randn_tensor

from transformers import AutoTokenizer, Qwen3ForCausalLM
from diffusers import AutoencoderKL

try:
    from diffusers import ZImagePipeline, ZImageControlNetInpaintPipeline
    from diffusers.models.controlnets import ZImageControlNetModel
    from diffusers.models.transformers import ZImageTransformer2DModel
except ImportError:
    raise ImportError(
        "Diffusers is out of date. Update diffusers to the latest version by doing pip uninstall diffusers and then pip install -r requirements.txt"
    )

try:
    import torch.nn.functional as F
    from PIL import Image
except ImportError:
    F = None
    Image = None


scheduler_config = {
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}


class ZImageModel(BaseModel):
    arch = "zimage"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["ZImageTransformer2DModel"]
        self.controlnet = None

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16 * 2  # 16 for the VAE, 2 for patch size

    def load_training_adapter(self, transformer: ZImageTransformer2DModel):
        self.print_and_status_update("Loading assistant LoRA")
        lora_path = self.model_config.assistant_lora_path
        if not os.path.exists(lora_path):
            # assume it is a hub path
            lora_splits = lora_path.split("/")
            if len(lora_splits) != 3:
                raise ValueError(
                    f"Assistant LoRA path {lora_path} is not a valid local path or hub path."
                )
            repo_id = "/".join(lora_splits[:2])
            filename = lora_splits[2]
            try:
                lora_path = huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                )
                # upgrade path to
                self.model_config.assistant_lora_path = lora_path
            except Exception as e:
                raise ValueError(
                    f"Failed to download assistant LoRA from {lora_path}: {e}"
                )
        # load the adapter and merge it in. We will inference with a -1.0 multiplier so the adapter effects only work during training.
        lora_state_dict = load_file(lora_path)
        dim = int(
            lora_state_dict[
                "diffusion_model.layers.0.attention.to_k.lora_A.weight"
            ].shape[0]
        )

        new_sd = {}
        for key, value in lora_state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        lora_state_dict = new_sd

        network_config = {
            "type": "lora",
            "linear": dim,
            "linear_alpha": dim,
            "transformer_only": True,
        }

        network_config = NetworkConfig(**network_config)
        LoRASpecialNetwork.LORA_PREFIX_UNET = "lora_transformer"
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=transformer,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=True,
            target_lin_modules=self.target_lora_modules,
            is_assistant_adapter=True,
            is_ara=True,
        )
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        self.print_and_status_update("Merging in assistant LoRA")
        network.force_to(self.device_torch, dtype=self.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_state_dict)

        network.merge_in(merge_weight=1.0)

        # mark it as not merged so inference ignores it.
        network.is_merged_in = False

        # add the assistant so sampler will activate it while sampling
        self.assistant_lora: LoRASpecialNetwork = network

        # deactivate lora during training
        self.assistant_lora.multiplier = -1.0
        self.assistant_lora.is_active = False

        # tell the model to invert assistant on inference since we want remove lora effects
        self.invert_assistant_lora = True

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading ZImage model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path

        self.print_and_status_update("Loading transformer")

        transformer_path = model_path
        transformer_subfolder = "transformer"
        if os.path.exists(transformer_path):
            transformer_subfolder = None
            transformer_path = os.path.join(transformer_path, "transformer")
            # check if the path is a full checkpoint.
            te_folder_path = os.path.join(model_path, "text_encoder")
            # if we have the te, this folder is a full checkpoint, use it as the base
            if os.path.exists(te_folder_path):
                base_model_path = model_path

        transformer = ZImageTransformer2DModel.from_pretrained(
            transformer_path, subfolder=transformer_subfolder, torch_dtype=dtype
        )

        # load assistant lora if specified
        if self.model_config.assistant_lora_path is not None:
            self.load_training_adapter(transformer)
            # set qtype to be float8 if it is qfloat8
            if self.model_config.qtype == "qfloat8":
                self.model_config.qtype = "float8"

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[
                    transformer.x_pad_token,
                    transformer.cap_pad_token,
                ]
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")

        flush()

        self.print_and_status_update("Text Encoder")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer", torch_dtype=dtype
        )
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype
        )

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        self.print_and_status_update("Loading VAE")
        vae = AutoencoderKL.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype
        )

        self.noise_scheduler = ZImageModel.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        kwargs = {}

        pipe: ZImagePipeline = ZImagePipeline(
            scheduler=self.noise_scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
            **kwargs,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        # leave it on cpu for now
        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # just to make sure everything is on the right device and dtype
        text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        flush()

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = ZImageModel.get_train_scheduler()

        if self.controlnet is not None:
            pipeline: ZImageControlNetInpaintPipeline = ZImageControlNetInpaintPipeline(
                scheduler=scheduler,
                text_encoder=unwrap_model(self.text_encoder[0]),
                tokenizer=self.tokenizer[0],
                vae=unwrap_model(self.vae),
                transformer=unwrap_model(self.transformer),
                controlnet=unwrap_model(self.controlnet),
            )

            pipeline = pipeline.to(self.device_torch)

            return pipeline

        pipeline: ZImagePipeline = ZImagePipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: ZImagePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if self.model.device != self.device_torch:
            self.model.to(self.device_torch, dtype=self.torch_dtype)
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)
        if isinstance(pipeline, ZImageControlNetInpaintPipeline):
            if gen_config.ctrl_img is None:
                raise ValueError("Z-Image ControlNet inpaint/outpaint samples require sample.ctrl_img")
            if gen_config.mask_img is None:
                raise ValueError("Z-Image ControlNet inpaint/outpaint samples require sample.mask_img")
            if Image is None:
                raise ImportError("Pillow is required for Z-Image ControlNet sampling")

            init_img = Image.open(gen_config.ctrl_img).convert("RGB")
            mask_img = Image.open(gen_config.mask_img).convert("L")
            if init_img.size != (gen_config.width, gen_config.height):
                init_img = init_img.resize((gen_config.width, gen_config.height), Image.BILINEAR)
            if mask_img.size != (gen_config.width, gen_config.height):
                mask_img = mask_img.resize((gen_config.width, gen_config.height), Image.NEAREST)

            latents, preserve_latents, edit_mask = self._prepare_masked_sample_latents(
                pipeline=pipeline,
                image=init_img,
                mask_image=mask_img,
                width=gen_config.width,
                height=gen_config.height,
                generator=generator,
                latents=gen_config.latents,
            )

            def preserve_known_latents(_pipeline, _step_index, _timestep, callback_kwargs):
                step_latents = callback_kwargs["latents"]
                callback_kwargs["latents"] = step_latents * edit_mask + preserve_latents * (1 - edit_mask)
                return callback_kwargs

            img = pipeline(
                image=init_img,
                mask_image=mask_img,
                control_image=init_img,
                prompt_embeds=conditional_embeds.text_embeds,
                negative_prompt_embeds=unconditional_embeds.text_embeds,
                height=gen_config.height,
                width=gen_config.width,
                num_inference_steps=gen_config.num_inference_steps,
                guidance_scale=gen_config.guidance_scale,
                controlnet_conditioning_scale=gen_config.adapter_conditioning_scale,
                latents=latents,
                generator=generator,
                callback_on_step_end=preserve_known_latents,
                **extra,
            ).images[0]
            img = Image.composite(img, init_img, mask_img)
        else:
            img = pipeline(
                prompt_embeds=conditional_embeds.text_embeds,
                negative_prompt_embeds=unconditional_embeds.text_embeds,
                height=gen_config.height,
                width=gen_config.width,
                num_inference_steps=gen_config.num_inference_steps,
                guidance_scale=gen_config.guidance_scale,
                latents=gen_config.latents,
                generator=generator,
                **extra,
            ).images[0]
        return img

    def attach_controlnet(self, controlnet: ZImageControlNetModel):
        self.controlnet = ZImageControlNetModel.from_transformer(controlnet, unwrap_model(self.transformer))
        if not hasattr(self.controlnet, "gradient_checkpointing"):
            self.controlnet.gradient_checkpointing = False
        self.adapter = self.controlnet

    def _get_mask_for_batch(self, batch, latents: torch.Tensor):
        mask = None
        if getattr(batch, "mask_tensor", None) is not None:
            mask = batch.mask_tensor
        elif getattr(batch, "inpaint_tensor", None) is not None:
            inpaint_tensor = batch.inpaint_tensor
            if inpaint_tensor.shape[1] == 4:
                # Existing inpaint tensors use alpha 1 as keep area, so invert for
                # Diffusers' inpaint convention where 1 is the generated area.
                mask = 1 - inpaint_tensor[:, 3:4, :, :]
            else:
                mask = inpaint_tensor[:, :1, :, :]

        if mask is None:
            mask = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], device=latents.device)

        mask = mask.to(latents.device, dtype=latents.dtype)
        if mask.shape[-2:] != latents.shape[-2:]:
            mask = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")
        return mask

    def _compose_masked_latents(self, noisy_latents: torch.Tensor, batch):
        if F is None:
            raise ImportError("torch.nn.functional is required for Z-Image masked latent conditioning")
        if self.controlnet is None:
            return noisy_latents
        if getattr(batch, "latents", None) is None:
            return noisy_latents
        if getattr(batch, "mask_tensor", None) is None and getattr(batch, "inpaint_tensor", None) is None:
            return noisy_latents

        edit_mask = self._get_mask_for_batch(batch, noisy_latents)
        clean_latents = batch.latents.to(noisy_latents.device, dtype=noisy_latents.dtype)
        if clean_latents.shape[-2:] != noisy_latents.shape[-2:]:
            clean_latents = F.interpolate(
                clean_latents,
                size=noisy_latents.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return noisy_latents * edit_mask + clean_latents * (1 - edit_mask)

    def condition_noisy_latents(self, latents: torch.Tensor, batch):
        return self._compose_masked_latents(latents, batch)

    def _prepare_masked_sample_latents(
        self,
        pipeline: ZImageControlNetInpaintPipeline,
        image: Image.Image,
        mask_image: Image.Image,
        width: int,
        height: int,
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        device = self.device_torch
        init_image = pipeline.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=1,
            num_images_per_prompt=1,
            device=device,
            dtype=self.vae.dtype,
        )
        mask_condition = pipeline.mask_processor.preprocess(mask_image, height=height, width=width)
        mask_condition = mask_condition.to(device=device, dtype=init_image.dtype)
        init_image = init_image * (mask_condition < 0.5)

        self.vae.to(self.vae_device_torch, dtype=self.torch_dtype)
        init_image = init_image.to(self.vae_device_torch, dtype=self.torch_dtype)
        encoded = self.vae.encode(init_image)
        if hasattr(encoded, "latent_dist"):
            preserve_latents = encoded.latent_dist.mode()
        elif hasattr(encoded, "latents"):
            preserve_latents = encoded.latents
        else:
            raise AttributeError("Could not access latents of provided encoder output")
        preserve_latents = (preserve_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        preserve_latents = preserve_latents.to(device=device, dtype=torch.float32)

        edit_mask = F.interpolate(
            mask_condition[:, :1].to(device=device, dtype=torch.float32),
            size=preserve_latents.shape[-2:],
            mode="nearest",
        )

        if latents is None:
            latents = randn_tensor(preserve_latents.shape, generator=generator, device=device, dtype=torch.float32)
        else:
            if latents.shape != preserve_latents.shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {preserve_latents.shape}")
            latents = latents.to(device=device, dtype=torch.float32)

        latents = latents * edit_mask + preserve_latents * (1 - edit_mask)
        return latents, preserve_latents, edit_mask

    def _encode_control_tensor(self, tensor: torch.Tensor, target_hw, device, dtype):
        self.vae.to(self.vae_device_torch, dtype=self.torch_dtype)
        tensor = tensor.to(self.vae_device_torch, dtype=self.torch_dtype)
        if tensor.shape[-2:] != target_hw:
            tensor = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
        tensor = tensor * 2 - 1
        latent = self.encode_images(tensor).to(device, dtype=dtype)
        return latent

    def build_controlnet_condition(self, latents: torch.Tensor, batch):
        if F is None:
            raise ImportError("torch.nn.functional is required for Z-Image ControlNet conditioning")

        with torch.no_grad():
            target_hw = None
            if getattr(batch, "tensor", None) is not None:
                target_hw = batch.tensor.shape[-2:]
            else:
                target_hw = (batch.file_items[0].crop_height, batch.file_items[0].crop_width)

            control_tensor = getattr(batch, "control_tensor", None)
            if control_tensor is None:
                masked_image_latents = latents
            else:
                control_tensor = control_tensor[:, :3, :, :]
                masked_image_latents = self._encode_control_tensor(
                    control_tensor,
                    target_hw,
                    latents.device,
                    latents.dtype,
                )

            control_latents = masked_image_latents.unsqueeze(2)

            mask = self._get_mask_for_batch(batch, latents)
            # Diffusers ZImageControlNetInpaintPipeline inverts the mask before
            # passing it to ControlNet: 1 means keep/preserved region here.
            keep_mask = 1 - mask
            keep_mask = F.interpolate(keep_mask, size=masked_image_latents.shape[-2:], mode="nearest")
            keep_mask = keep_mask.to(latents.device, dtype=latents.dtype).unsqueeze(2)

            masked_image_latents = (masked_image_latents * keep_mask.squeeze(2)).unsqueeze(2)

            return torch.cat([control_latents, keep_mask, masked_image_latents], dim=1)

    def get_controlnet_block_samples(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: PromptEmbeds,
        batch,
        controlnet: ZImageControlNetModel,
        conditioning_scale: float = 1.0,
    ):
        control_dtype = controlnet.dtype
        latent_model_input = self._compose_masked_latents(latent_model_input, batch)
        control_image_input = self.build_controlnet_condition(latent_model_input, batch).to(
            self.device_torch, dtype=control_dtype
        )
        latent_model_input = latent_model_input.to(self.device_torch, dtype=control_dtype).unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))
        timestep_model_input = (1000 - timestep) / 1000
        return controlnet(
            latent_model_input_list,
            timestep_model_input,
            text_embeddings.text_embeds,
            control_image_input,
            conditioning_scale=conditioning_scale,
        )

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        controlnet_block_samples=None,
        **kwargs,
    ):
        latent_model_input = latent_model_input.unsqueeze(2)
        latent_model_input_list = list(latent_model_input.unbind(dim=0))

        timestep_model_input = (1000 - timestep) / 1000

        model_out_list = self.transformer(
            latent_model_input_list,
            timestep_model_input,
            text_embeddings.text_embeds,
            controlnet_block_samples=controlnet_block_samples,
        )[0]

        noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred

        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt,
            do_classifier_free_guidance=False,
            device=self.device_torch,
        )
        pe = PromptEmbeds([prompt_embeds, None])
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer: ZImageTransformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "zimage"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd
