from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def get_step(checkpoint_dir: Path) -> int:
    match = re.search(r"_(\d{9})$", checkpoint_dir.name)
    if match:
        return int(match.group(1))

    meta_path = checkpoint_dir / "aitk_meta.yaml"
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("step:"):
                return int(line.split(":", 1)[1].strip())

    return -1


def find_latest_checkpoint(job_output_dir: Path) -> Path:
    candidates = []
    for checkpoint_dir in job_output_dir.iterdir():
        if not checkpoint_dir.is_dir():
            continue
        weights_path = checkpoint_dir / "diffusion_pytorch_model.safetensors"
        if weights_path.exists():
            candidates.append(checkpoint_dir)

    if not candidates:
        raise FileNotFoundError(f"No diffusers checkpoint folders found in {job_output_dir}")

    return max(candidates, key=lambda path: (get_step(path), path.stat().st_mtime))


def export_control_tensors(source: Path, destination: Path, dtype: torch.dtype | None) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)

    tensors = {}
    with safe_open(str(source), framework="pt", device="cpu") as source_file:
        for key in source_file.keys():
            if not key.startswith("control_"):
                continue
            tensor = source_file.get_tensor(key)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            tensors[key] = tensor

    if not tensors:
        raise RuntimeError(f"No control_* tensors found in {source}")

    save_file(tensors, str(destination), metadata={"format": "pt", "source": "ai-toolkit-zimage-controlnet"})
    return len(tensors)


def parse_dtype(value: str) -> torch.dtype | None:
    if value == "keep":
        return None
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an ai-toolkit Z-Image ControlNet checkpoint as a ComfyUI model patch.")
    parser.add_argument(
        "--job-output-dir",
        type=Path,
        default=Path("output/zimage_green_outpaint_4k"),
        help="Training output folder containing checkpoint directories.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Specific checkpoint directory to export.")
    parser.add_argument("--output", type=Path, default=None, help="Destination .safetensors path.")
    parser.add_argument("--copy-to", type=Path, default=None, help="Optional folder to copy the exported file into.")
    parser.add_argument("--dtype", choices=["keep", "fp16", "bf16"], default="fp16")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or find_latest_checkpoint(args.job_output_dir)
    source = checkpoint_dir / "diffusion_pytorch_model.safetensors"
    if not source.exists():
        raise FileNotFoundError(source)

    step = get_step(checkpoint_dir)
    output = args.output
    if output is None:
        suffix = f"_{step:09d}" if step >= 0 else ""
        output = args.job_output_dir / "comfyui" / f"{args.job_output_dir.name}{suffix}_model_patch.safetensors"

    count = export_control_tensors(source, output, parse_dtype(args.dtype))
    print(f"Exported {count} control tensors")
    print(f"Source: {source}")
    print(f"Output: {output}")

    if args.copy_to is not None:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        copied_path = args.copy_to / output.name
        shutil.copy2(output, copied_path)
        print(f"Copied: {copied_path}")


if __name__ == "__main__":
    main()
