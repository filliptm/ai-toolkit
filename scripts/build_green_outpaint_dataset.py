import argparse
import csv
import html
import re
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
GREEN = (0, 255, 0)


def clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"^File:", "", value).strip()
    value = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", value).strip()
    return value


def first_sentence(value: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", value.strip(), maxsplit=1)
    return parts[0].strip() if parts else value.strip()


def load_attribution(source_dir: Path) -> dict[str, dict[str, str]]:
    attribution_path = source_dir / "attribution.csv"
    if not attribution_path.exists():
        return {}
    with attribution_path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["filename"]: row for row in csv.DictReader(f)}


def source_caption(image_path: Path, attribution: dict[str, dict[str, str]]) -> str:
    row = attribution.get(image_path.name, {})
    description = clean_text(row.get("description", ""))
    title = clean_text(row.get("title", "")) or clean_text(image_path.stem)
    if description:
        base = description
    elif title and not title.lower().startswith("picsum_"):
        base = title
    else:
        base = "A high resolution photographic scene with natural composition and realistic detail"
    base = first_sentence(base)
    if not base.endswith("."):
        base += "."
    return base


def center_crop_box(width: int, height: int, aspect_width: int, aspect_height: int) -> tuple[int, int, int, int]:
    target_ratio = aspect_width / aspect_height
    current_ratio = width / height
    if current_ratio > target_ratio:
        crop_w = round(height * target_ratio)
        left = (width - crop_w) // 2
        return left, 0, left + crop_w, height
    crop_h = round(width / target_ratio)
    top = (height - crop_h) // 2
    return 0, top, width, top + crop_h


def edge_fractions(index: int, portrait: bool) -> tuple[float, float, float, float]:
    # Deterministic variety without requiring a sidecar random state.
    if portrait:
        left = 0.14 + ((index * 17) % 10) / 100
        right = 0.14 + ((index * 23 + 3) % 10) / 100
        top = 0.12 + ((index * 29 + 5) % 11) / 100
        bottom = 0.12 + ((index * 31 + 7) % 11) / 100
    else:
        left = 0.16 + ((index * 13) % 12) / 100
        right = 0.16 + ((index * 19 + 4) % 12) / 100
        top = 0.10 + ((index * 11 + 2) % 10) / 100
        bottom = 0.10 + ((index * 7 + 6) % 10) / 100
    return left, right, top, bottom


def make_pair(
    target: Image.Image,
    index: int,
    portrait: bool,
    side_mode: str,
) -> tuple[Image.Image, Image.Image, dict[str, int]]:
    width, height = target.size
    left_f, right_f, top_f, bottom_f = edge_fractions(index, portrait)
    left = max(32, round(width * left_f))
    right = max(32, round(width * right_f))
    top = max(32, round(height * top_f))
    bottom = max(32, round(height * bottom_f))

    control = target.copy()
    mask = Image.new("L", (width, height), 0)

    if side_mode == "all":
        bands = [
            (0, 0, left, height),
            (width - right, 0, width, height),
            (0, 0, width, top),
            (0, height - bottom, width, height),
        ]
        active_sides = "left,right,top,bottom"
    elif side_mode == "two":
        if portrait:
            bands = [
                (0, 0, width, top),
                (0, height - bottom, width, height),
            ]
            active_sides = "top,bottom"
        else:
            bands = [
                (0, 0, left, height),
                (width - right, 0, width, height),
            ]
            active_sides = "left,right"
    else:
        raise ValueError(f"Unknown side_mode {side_mode}")

    for band in bands:
        control.paste(GREEN, band)
        mask.paste(255, band)

    return control, mask, {
        "left": left if "left" in active_sides else 0,
        "right": right if "right" in active_sides else 0,
        "top": top if "top" in active_sides else 0,
        "bottom": bottom if "bottom" in active_sides else 0,
        "active_sides": active_sides,
    }


def render_variant(
    image: Image.Image,
    source_path: Path,
    out_root: Path,
    sample_id: str,
    index: int,
    portrait: bool,
    side_mode: str,
    caption_base: str,
) -> dict[str, str | int]:
    if portrait:
        box = center_crop_box(image.width, image.height, 9, 16)
        target_size = (2160, 3840)
        variant = "portrait_9x16" if side_mode == "all" else "portrait_9x16_top_bottom"
        orientation_caption = (
            "The training target is a vertical 9:16 outpaint with neon green border regions to replace."
        )
    else:
        box = center_crop_box(image.width, image.height, 16, 9)
        target_size = (3840, 2160)
        variant = "landscape_16x9" if side_mode == "all" else "landscape_16x9_left_right"
        orientation_caption = "The training target is a wide 16:9 outpaint with neon green border regions to replace."

    target = image.crop(box).resize(target_size, Image.Resampling.LANCZOS)
    control, mask, border = make_pair(target, index=index, portrait=portrait, side_mode=side_mode)

    stem = f"{sample_id}_{variant}"
    target_path = out_root / "target" / f"{stem}.jpg"
    control_path = out_root / "control" / f"{stem}.png"
    mask_path = out_root / "mask" / f"{stem}.png"
    caption_path = out_root / "target" / f"{stem}.txt"

    target.save(target_path, quality=95, optimize=True)
    control.save(control_path, optimize=True)
    mask.save(mask_path, optimize=True)

    caption = (
        f"{caption_base} "
        "Fill only the neon green areas with coherent continuation of the scene, matching perspective, lighting, texture, and detail. "
        f"{orientation_caption}"
    )
    caption_path.write_text(caption + "\n", encoding="utf-8")

    return {
        "sample": stem,
        "variant": variant,
        "source": str(source_path),
        "target": str(target_path),
        "control": str(control_path),
        "mask": str(mask_path),
        "caption": str(caption_path),
        "width": target_size[0],
        "height": target_size[1],
        "side_mode": side_mode,
        "crop_left": box[0],
        "crop_top": box[1],
        "crop_right": box[2],
        "crop_bottom": box[3],
        **border,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--variant-set",
        choices=["all", "all_edges", "two_side"],
        default="all",
        help="Which edge-mask variants to generate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source)
    out_root = Path(args.output)
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    for child in ["target", "control", "mask"]:
        (out_root / child).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    attribution = load_attribution(source_dir)
    existing_rows = []
    manifest_path = out_root / "manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            existing_rows = list(csv.DictReader(f))
    existing_samples = {row.get("sample") for row in existing_rows}

    if args.variant_set == "all":
        variant_modes = [False, True]
        side_modes = ["all", "two"]
    elif args.variant_set == "all_edges":
        variant_modes = [False, True]
        side_modes = ["all"]
    else:
        variant_modes = [False, True]
        side_modes = ["two"]

    manifest_rows = []
    failures = []
    for index, image_path in enumerate(image_paths):
        sample_id = f"{index + 1:04d}"
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                caption_base = source_caption(image_path, attribution)
                for portrait in variant_modes:
                    for side_mode in side_modes:
                        preview_variant = (
                            "portrait_9x16"
                            if portrait and side_mode == "all"
                            else "portrait_9x16_top_bottom"
                            if portrait
                            else "landscape_16x9"
                            if side_mode == "all"
                            else "landscape_16x9_left_right"
                        )
                        if f"{sample_id}_{preview_variant}" in existing_samples:
                            continue
                        manifest_rows.append(
                            render_variant(
                                img,
                                image_path,
                                out_root,
                                sample_id,
                                index,
                                portrait,
                                side_mode,
                                caption_base,
                            )
                        )
        except Exception as exc:
            failures.append({"source": str(image_path), "error": str(exc)})

    combined_rows = existing_rows + manifest_rows
    if combined_rows:
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = []
            for row in combined_rows:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)

    failures_path = out_root / "failures.csv"
    if failures:
        with failures_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "error"])
            writer.writeheader()
            writer.writerows(failures)

    readme = out_root / "README.md"
    readme.write_text(
        "# Z-Image Green Outpaint Dataset\n\n"
        f"Source folder: `{source_dir}`\n\n"
        f"Validated source images: {len(image_paths) - len(failures)}\n\n"
        f"Generated paired samples: {len(combined_rows)}\n\n"
        "- `target/`: completed training targets with `.txt` captions.\n"
        "- `control/`: matching RGB PNGs with exact `(0, 255, 0)` neon green border regions.\n"
        "- `mask/`: matching grayscale PNG masks where white is the generated/outpaint region.\n"
        "- Each source image can get 16:9 landscape and 9:16 portrait samples.\n"
        "- Variants include all-edge masks plus two-side masks: left/right for 16:9 and top/bottom for 9:16.\n",
        encoding="utf-8",
    )

    print(f"source_images={len(image_paths)}")
    print(f"new_samples={len(manifest_rows)}")
    print(f"total_samples={len(combined_rows)}")
    print(f"failures={len(failures)}")
    print(f"output={out_root}")


if __name__ == "__main__":
    main()
