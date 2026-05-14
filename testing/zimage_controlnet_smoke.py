import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers.models.controlnets import ZImageControlNetModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel
from toolkit.config_modules import GenerateImageConfig, ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlnet", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--prompt", default="extend the cozy illustrated room naturally to the right")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = ModelConfig(
        name_or_path="Tongyi-MAI/Z-Image-Turbo",
        arch="zimage",
        quantize=True,
        quantize_te=True,
        low_vram=False,
        qtype="qfloat8",
        qtype_te="qfloat8",
    )
    model = ZImageModel(device=device, model_config=model_config, dtype="bf16")
    model.load_model()

    controlnet = ZImageControlNetModel.from_single_file(
        args.controlnet,
        torch_dtype=model.torch_dtype,
        control_in_dim=33,
        low_cpu_mem_usage=False,
    )
    model.attach_controlnet(controlnet)

    pipeline = model.get_generation_pipeline()
    prompt_embeds = model.get_prompt_embeds(args.prompt)
    negative_embeds = model.get_prompt_embeds("")

    generator = torch.Generator(device=device).manual_seed(1234)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gen_config = GenerateImageConfig(
        prompt=args.prompt,
        negative_prompt="",
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=1.0,
        seed=1234,
        output_path=str(output_path),
        ctrl_img=os.path.abspath(args.source),
        mask_img=os.path.abspath(args.mask),
        adapter_conditioning_scale=0.85,
    )
    image = model.generate_single_image(
        pipeline=pipeline,
        gen_config=gen_config,
        conditional_embeds=prompt_embeds,
        unconditional_embeds=negative_embeds,
        generator=generator,
        extra={},
    )
    image.save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
