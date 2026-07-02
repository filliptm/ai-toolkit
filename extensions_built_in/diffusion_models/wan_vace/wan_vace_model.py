import os
from functools import partial
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from diffusers import WanVACEPipeline, WanVACETransformer3DModel
from PIL import Image
import cv2

from extensions_built_in.diffusion_models.wan22.wan22_5b_model import time_text_monkeypatch
from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.models.wan21.wan21 import Wan21
from toolkit.models.wan21.wan_lora_convert import convert_to_diffusers, convert_to_original
from toolkit.prompt_utils import PromptEmbeds
from toolkit.util.quantize import quantize_model


class WanVACEModel(Wan21):
    arch = "wan21_vace"
    _wan_vae_path = None

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
            device=device,
            model_config=model_config,
            dtype=dtype,
            custom_pipeline=custom_pipeline,
            noise_scheduler=noise_scheduler,
            **kwargs,
        )
        self.target_lora_modules = ["WanVACETransformer3DModel"]
        self.vace_task = self.model_config.model_kwargs.get("vace_task", "edit")
        self.vace_conditioning_scale = self.model_config.model_kwargs.get("conditioning_scale", 1.0)
        self.vace_default_mask = self.model_config.model_kwargs.get("default_mask", "full")
        self.vace_require_mask = self.model_config.model_kwargs.get("require_mask", False)

    def get_bucket_divisibility(self):
        return 16

    def load_wan_transformer(self, transformer_path, subfolder=None):
        self.print_and_status_update("Loading Wan VACE transformer")
        dtype = self.torch_dtype
        disable_mmap = self.model_config.model_kwargs.get("disable_mmap", os.name == "nt")
        transformer = WanVACETransformer3DModel.from_pretrained(
            transformer_path,
            subfolder=subfolder,
            torch_dtype=dtype,
            disable_mmap=disable_mmap,
        ).to(dtype=dtype)

        if self.model_config.split_model_over_gpus:
            raise ValueError("Splitting model over gpus is not supported for Wan VACE models")

        if self.model_config.assistant_lora_path is not None or self.model_config.inference_lora_path is not None:
            raise ValueError("Assistant LoRA is not supported for Wan VACE models currently")

        if self.model_config.lora_path is not None:
            raise ValueError("Loading LoRA through model.lora_path is not supported for Wan VACE models currently")

        if self.model_config.low_vram:
            transformer.to("cpu", dtype=dtype)
        else:
            transformer.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Wan VACE transformer")
            quantize_model(self, transformer)
            flush()

        if self.model_config.low_vram:
            self.print_and_status_update("Moving Wan VACE transformer to CPU")
            transformer.to("cpu")

        return transformer

    def load_model(self):
        super().load_model()
        # Diffusers' Wan condition embedder has historically needed this patch for
        # batch-shaped timestep tensors during training. Keep VACE aligned with Wan2.2.
        self.model.condition_embedder.forward = partial(time_text_monkeypatch, self.model.condition_embedder)

    def get_generation_pipeline(self):
        scheduler = self.get_train_scheduler()
        pipeline = WanVACEPipeline(
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.model),
            text_encoder=unwrap_model(self.text_encoder),
            tokenizer=self.tokenizer,
            scheduler=scheduler,
        )
        pipeline = pipeline.to(self.device_torch)
        return pipeline

    def _conditioning_scale_tensor(self, device=None, dtype=None):
        transformer = unwrap_model(self.model)
        scale = self.vace_conditioning_scale
        if isinstance(scale, torch.Tensor):
            return scale.to(device=device or self.device_torch, dtype=dtype or self.torch_dtype)
        num_layers = len(transformer.config.vace_layers)
        if isinstance(scale, list):
            if len(scale) != num_layers:
                raise ValueError(
                    f"model.model_kwargs.conditioning_scale has {len(scale)} values, expected {num_layers}"
                )
            return torch.tensor(scale, device=device or self.device_torch, dtype=dtype or self.torch_dtype)
        return torch.full((num_layers,), float(scale), device=device or self.device_torch, dtype=dtype or self.torch_dtype)

    def _loaded_vae_dtype(self):
        if self.torch_dtype in [torch.float16, torch.bfloat16, torch.float32]:
            return self.torch_dtype
        try:
            return self.vae.encoder.conv_in.bias.dtype
        except AttributeError:
            pass
        try:
            return next(self.vae.parameters()).dtype
        except (AttributeError, StopIteration):
            return getattr(self.vae, "dtype", None) or self.vae_torch_dtype

    def _normalize_video_tensor(self, tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        # Dataloader image controls are B,C,H,W in [0,1]. Video tensors may be
        # B,T,C,H,W or B,C,T,H,W. VACE wants B,C,T,H,W in [-1,1].
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(2)
        elif tensor.ndim == 5 and tensor.shape[1] != 3 and tensor.shape[2] == 3:
            tensor = tensor.permute(0, 2, 1, 3, 4)
        elif tensor.ndim != 5:
            raise ValueError(f"Unsupported VACE conditioning tensor shape: {tensor.shape}")

        tensor = tensor.to(self.vae_device_torch, dtype=self._loaded_vae_dtype())
        if tensor.min() >= 0:
            tensor = tensor * 2.0 - 1.0

        b, c, t, h, w = tensor.shape
        if h != target_h or w != target_w:
            tensor = tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            tensor = F.interpolate(tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
            tensor = tensor.view(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4)
        return tensor

    def _normalize_mask_tensor(
        self,
        mask: Optional[torch.Tensor],
        batch_size: int,
        num_frames: int,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        if mask is None:
            fill = 1.0 if self.vace_default_mask == "full" else 0.0
            return torch.full(
                (batch_size, 1, num_frames, target_h, target_w),
                fill,
                device=self.vae_device_torch,
                dtype=self._loaded_vae_dtype(),
            )
        if mask.ndim == 4:
            mask = mask.unsqueeze(2)
        elif mask.ndim == 5 and mask.shape[1] != 1 and mask.shape[2] == 1:
            mask = mask.permute(0, 2, 1, 3, 4)
        elif mask.ndim != 5:
            raise ValueError(f"Unsupported VACE mask tensor shape: {mask.shape}")

        mask = mask.to(self.vae_device_torch, dtype=self._loaded_vae_dtype())
        if mask.shape[1] != 1:
            mask = mask[:, :1]
        b, c, t, h, w = mask.shape
        if h != target_h or w != target_w:
            mask = mask.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            mask = F.interpolate(mask, size=(target_h, target_w), mode="nearest")
            mask = mask.view(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4)
        return torch.where(mask > 0.5, torch.ones_like(mask), torch.zeros_like(mask))

    def _prepare_masks(
        self,
        mask: torch.Tensor,
        num_reference_images: int = 0,
    ) -> torch.Tensor:
        patch_size = unwrap_model(self.model).config.patch_size[1]
        vae_scale = self.vae.config.scale_factor_spatial
        temporal_scale = 2 ** sum(self.vae.temperal_downsample)
        mask_list = []
        for mask_item in mask:
            _, num_frames, height, width = mask_item.shape
            new_num_frames = (num_frames + temporal_scale - 1) // temporal_scale
            new_height = height // (vae_scale * patch_size) * patch_size
            new_width = width // (vae_scale * patch_size) * patch_size
            mask_item = mask_item[0, :, :, :]
            mask_item = mask_item.view(num_frames, new_height, vae_scale, new_width, vae_scale)
            mask_item = mask_item.permute(2, 4, 0, 1, 3).flatten(0, 1)
            mask_item = F.interpolate(
                mask_item.unsqueeze(0),
                size=(new_num_frames, new_height, new_width),
                mode="nearest-exact",
            ).squeeze(0)
            if num_reference_images > 0:
                mask_padding = torch.zeros_like(mask_item[:, :num_reference_images, :, :])
                mask_item = torch.cat([mask_padding, mask_item], dim=1)
            mask_list.append(mask_item)
        return torch.stack(mask_list)

    def _encode_vace_conditioning(
        self,
        video: torch.Tensor,
        mask: torch.Tensor,
        reference_images: Optional[List[List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        self.vae.to(self.vae_device_torch)
        vae_dtype = self._loaded_vae_dtype()
        video = video.to(self.vae_device_torch, dtype=vae_dtype)
        mask = mask.to(self.vae_device_torch, dtype=vae_dtype)
        mask = torch.where(mask > 0.5, torch.ones_like(mask), torch.zeros_like(mask))

        latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=self.vae_device_torch,
            dtype=torch.float32,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            self.vae.config.latents_std,
            device=self.vae_device_torch,
            dtype=torch.float32,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)

        inactive = video * (1 - mask)
        reactive = video * mask
        inactive_latents = self.vae.encode(inactive).latent_dist.mode()
        reactive_latents = self.vae.encode(reactive).latent_dist.mode()
        inactive_latents = ((inactive_latents.float() - latents_mean) * latents_std).to(vae_dtype)
        reactive_latents = ((reactive_latents.float() - latents_mean) * latents_std).to(vae_dtype)
        latents = torch.cat([inactive_latents, reactive_latents], dim=1)

        if reference_images:
            latent_list = []
            for latent, reference_batch in zip(latents, reference_images):
                for reference_image in reference_batch:
                    if reference_image is None:
                        continue
                    reference_image = reference_image.to(self.vae_device_torch, dtype=vae_dtype)
                    if reference_image.ndim == 3:
                        reference_image = reference_image.unsqueeze(0)
                    reference_image = reference_image.unsqueeze(2)
                    reference_latent = self.vae.encode(reference_image).latent_dist.mode()
                    reference_latent = ((reference_latent.float() - latents_mean) * latents_std).to(vae_dtype)
                    reference_latent = reference_latent.squeeze(0)
                    reference_latent = torch.cat([reference_latent, torch.zeros_like(reference_latent)], dim=0)
                    latent = torch.cat([reference_latent, latent], dim=1)
                latent_list.append(latent)
            latents = torch.stack(latent_list)

        mask_latents = self._prepare_masks(mask, len(reference_images[0]) if reference_images else 0)
        return torch.cat([latents, mask_latents], dim=1)

    def _build_training_conditioning(self, batch: DataLoaderBatchDTO, latent_model_input: torch.Tensor) -> torch.Tensor:
        if batch.control_tensor is None:
            raise ValueError(
                "Wan VACE training requires datasets.control_path with source A images/videos matching target B filenames"
            )
        if self.vace_require_mask and batch.mask_tensor is None:
            raise ValueError(
                "Wan VACE outpaint training requires datasets.mask_path with white editable regions and black preserved regions"
            )

        _, _, latent_frames, latent_h, latent_w = latent_model_input.shape
        temporal_scale = 2 ** sum(self.vae.temperal_downsample)
        num_frames = (latent_frames - 1) * temporal_scale + 1
        target_h = latent_h * self.vae.config.scale_factor_spatial
        target_w = latent_w * self.vae.config.scale_factor_spatial

        video = self._normalize_video_tensor(batch.control_tensor, target_h, target_w)
        if video.shape[2] != num_frames:
            if video.shape[2] == 1:
                video = video.expand(-1, -1, num_frames, -1, -1)
            else:
                video = F.interpolate(video, size=(num_frames, target_h, target_w), mode="trilinear", align_corners=False)

        mask = self._normalize_mask_tensor(batch.mask_tensor, video.shape[0], num_frames, target_h, target_w)
        reference_images = None
        if batch.clip_image_tensor is not None:
            refs = batch.clip_image_tensor.to(self.vae_device_torch, dtype=self._loaded_vae_dtype())
            if refs.ndim == 4:
                refs = refs * 2.0 - 1.0 if refs.min() >= 0 else refs
                reference_images = [[ref] for ref in refs]
            elif refs.ndim == 5:
                refs = refs * 2.0 - 1.0 if refs.min() >= 0 else refs
                reference_images = [[ref_item for ref_item in ref_batch] for ref_batch in refs]
            else:
                raise ValueError(f"Unsupported VACE reference tensor shape: {refs.shape}")
        return self._encode_vace_conditioning(video, mask, reference_images).to(
            latent_model_input.device, dtype=latent_model_input.dtype
        )

    def _load_pil_sequence(self, image_path: str, width: int, height: int, num_frames: int) -> List[Image.Image]:
        if os.path.splitext(image_path)[1].lower() in [".mp4", ".avi", ".mov", ".webm", ".mkv", ".wmv", ".m4v", ".flv"]:
            cap = cv2.VideoCapture(image_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open VACE sample control video: {image_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frame_index = max(total_frames - 1, 0)
            interval = max_frame_index / (num_frames - 1) if num_frames > 1 else 0
            frames_to_extract = [min(int(round(i * interval)), max_frame_index) for i in range(num_frames)]
            frames = []
            try:
                for frame_idx in frames_to_extract:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        raise ValueError(f"Could not read frame {frame_idx} from VACE sample control video: {image_path}")
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame).resize((width, height), Image.LANCZOS))
            finally:
                cap.release()
            return frames
        img = Image.open(image_path).convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)
        return [img.copy() for _ in range(num_frames)]

    def _load_mask_sequence(self, mask_path: Optional[str], width: int, height: int, num_frames: int) -> List[Image.Image]:
        if mask_path is None:
            color = 255 if self.vace_default_mask == "full" else 0
            mask = Image.new("L", (width, height), color)
        elif os.path.splitext(mask_path)[1].lower() in [".mp4", ".avi", ".mov", ".webm", ".mkv", ".wmv", ".m4v", ".flv"]:
            cap = cv2.VideoCapture(mask_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open VACE sample mask video: {mask_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frame_index = max(total_frames - 1, 0)
            interval = max_frame_index / (num_frames - 1) if num_frames > 1 else 0
            frames_to_extract = [min(int(round(i * interval)), max_frame_index) for i in range(num_frames)]
            masks = []
            try:
                for frame_idx in frames_to_extract:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        raise ValueError(f"Could not read frame {frame_idx} from VACE sample mask video: {mask_path}")
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    masks.append(Image.fromarray(frame).resize((width, height), Image.NEAREST))
            finally:
                cap.release()
            return masks
        else:
            mask = Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST)
        return [mask.copy() for _ in range(num_frames)]

    def _load_reference_images(self, reference_img: Optional[Union[str, List[str]]]) -> Optional[List[Image.Image]]:
        if reference_img is None:
            return None
        if isinstance(reference_img, str):
            reference_img = [reference_img]
        return [Image.open(path).convert("RGB") for path in reference_img]

    def generate_single_image(
        self,
        pipeline: WanVACEPipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        pipeline.set_progress_bar_config(disable=False)
        if gen_config.ctrl_img is None:
            raise ValueError("Wan VACE samples require sample.ctrl_img")
        if self.vace_require_mask and gen_config.mask_img is None:
            raise ValueError("Wan VACE outpaint samples require sample.mask_img")

        num_frames = ((gen_config.num_frames - 1) // 4) * 4 + 1
        gen_config.num_frames = num_frames
        d = self.get_bucket_divisibility()
        height = gen_config.height // d * d
        width = gen_config.width // d * d
        video = self._load_pil_sequence(gen_config.ctrl_img, width, height, num_frames)
        mask = self._load_mask_sequence(gen_config.mask_img, width, height, num_frames)
        reference_images = self._load_reference_images(gen_config.reference_img)

        output = pipeline(
            video=video,
            mask=mask,
            reference_images=reference_images,
            prompt_embeds=conditional_embeds.text_embeds.to(self.device_torch, dtype=self.torch_dtype),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(self.device_torch, dtype=self.torch_dtype),
            height=height,
            width=width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            conditioning_scale=self._conditioning_scale_tensor(self.device_torch, self.torch_dtype),
            generator=generator,
            return_dict=False,
            output_type="pil",
            **extra,
        )[0]

        batch_item = output[0]
        if gen_config.num_frames > 1:
            return batch_item
        return batch_item[0]

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: PromptEmbeds,
        batch: DataLoaderBatchDTO,
        **kwargs,
    ):
        with torch.no_grad():
            conditioning_latents = self._build_training_conditioning(batch, latent_model_input)

        noise_pred = self.model(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            control_hidden_states=conditioning_latents,
            control_hidden_states_scale=self._conditioning_scale_tensor(latent_model_input.device, latent_model_input.dtype),
            return_dict=False,
            **kwargs,
        )[0]
        return noise_pred

    def get_base_model_version(self):
        return "wan_2.1_vace"

    def get_transformer_block_names(self):
        return ["blocks", "vace_blocks"]

    def convert_lora_weights_before_save(self, state_dict):
        state_dict = convert_to_original(state_dict)
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key
            if ".vace_blocks." in new_key:
                new_key = new_key.replace(".proj_in.", ".before_proj.")
                new_key = new_key.replace(".proj_out.", ".after_proj.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key
            if ".vace_blocks." in new_key:
                new_key = new_key.replace(".before_proj.", ".proj_in.")
                new_key = new_key.replace(".after_proj.", ".proj_out.")
            new_sd[new_key] = value
        return convert_to_diffusers(new_sd)
