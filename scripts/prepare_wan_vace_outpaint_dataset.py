import argparse
import csv
import html
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ORDERED_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]
WHITE = (255, 255, 255)
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
        attribution_path = source_dir / "source_attribution.csv"
    if not attribution_path.exists():
        return {}
    with attribution_path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["filename"]: row for row in csv.DictReader(f) if row.get("filename")}


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


def outpaint_caption(base: str, orientation_caption: str = "") -> str:
    base = clean_text(base)
    base = re.sub(
        r"\s*Fill only the neon green areas.*$",
        "",
        base,
        flags=re.IGNORECASE,
    ).strip()
    base = re.sub(r"\bneon\s+green\b|\bgreen\b", "white", base, flags=re.IGNORECASE)
    if base and not base.endswith("."):
        base += "."
    caption = (
        f"{base} "
        "Extend the scene only inside the white masked area, matching the existing perspective, lighting, "
        "texture, color, and detail."
    ).strip()
    if orientation_caption:
        caption = f"{caption} {orientation_caption}"
    return caption.strip() + "\n"


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


def side_bands(width: int, height: int, index: int, portrait: bool, side_mode: str) -> tuple[list[tuple[int, int, int, int]], dict[str, int | str]]:
    left_f, right_f, top_f, bottom_f = edge_fractions(index, portrait)
    left = max(32, round(width * left_f))
    right = max(32, round(width * right_f))
    top = max(32, round(height * top_f))
    bottom = max(32, round(height * bottom_f))
    band_map = {
        "left": (0, 0, left, height),
        "right": (width - right, 0, width, height),
        "top": (0, 0, width, top),
        "bottom": (0, height - bottom, width, height),
    }
    if side_mode == "all":
        sides = ["left", "right", "top", "bottom"]
    elif side_mode == "two":
        sides = ["top", "bottom"] if portrait else ["left", "right"]
    elif side_mode in band_map:
        sides = [side_mode]
    elif side_mode == "left_right":
        sides = ["left", "right"]
    elif side_mode == "top_bottom":
        sides = ["top", "bottom"]
    else:
        raise ValueError(f"Unknown side mode: {side_mode}")

    return [band_map[side] for side in sides], {
        "left": left if "left" in sides else 0,
        "right": right if "right" in sides else 0,
        "top": top if "top" in sides else 0,
        "bottom": bottom if "bottom" in sides else 0,
        "active_sides": ",".join(sides),
    }


def make_pair(target: Image.Image, index: int, portrait: bool, side_mode: str) -> tuple[Image.Image, Image.Image, dict[str, int | str]]:
    width, height = target.size
    control = target.copy()
    mask = Image.new("L", (width, height), 0)
    bands, border = side_bands(width, height, index, portrait, side_mode)
    for band in bands:
        control.paste(WHITE, band)
        mask.paste(255, band)
    return control, mask, border


def variant_name(portrait: bool, side_mode: str) -> str:
    prefix = "portrait_9x16" if portrait else "landscape_16x9"
    if side_mode == "all":
        return prefix
    if side_mode == "two":
        return f"{prefix}_{'top_bottom' if portrait else 'left_right'}"
    return f"{prefix}_{side_mode}"


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
        orientation_caption = "The training target is a vertical 9:16 VACE outpaint."
    else:
        box = center_crop_box(image.width, image.height, 16, 9)
        target_size = (3840, 2160)
        orientation_caption = "The training target is a wide 16:9 VACE outpaint."

    target = image.crop(box).resize(target_size, Image.Resampling.LANCZOS)
    control, mask, border = make_pair(target, index=index, portrait=portrait, side_mode=side_mode)

    variant = variant_name(portrait, side_mode)
    stem = f"{sample_id}_{variant}"
    target_path = out_root / "target" / f"{stem}.jpg"
    control_path = out_root / "control" / f"{stem}.png"
    mask_path = out_root / "mask" / f"{stem}.png"
    caption_path = out_root / "target" / f"{stem}.txt"

    target.save(target_path, quality=95, optimize=True)
    control.save(control_path, optimize=True)
    mask.save(mask_path, optimize=True)
    caption_path.write_text(outpaint_caption(caption_base, orientation_caption), encoding="utf-8")

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
        "fill_color": "white",
        "mask_convention": "white_edit_black_preserve",
        **border,
    }


def green_mask_from_control(control: Image.Image, threshold: int = 80) -> Image.Image:
    arr = np.asarray(control.convert("RGB"))
    green = (arr[:, :, 1] >= 255 - threshold) & (arr[:, :, 0] <= threshold) & (arr[:, :, 2] <= threshold)
    return Image.fromarray((green.astype(np.uint8) * 255), mode="L")


def binary_mask(mask: Image.Image) -> Image.Image:
    gray = mask.convert("L")
    return gray.point(lambda value: 255 if value > 127 else 0, mode="L")


def mask_has_white(mask: Image.Image) -> bool:
    return mask.convert("L").getbbox() is not None


def apply_white_fill(control: Image.Image, mask: Image.Image) -> Image.Image:
    result = control.convert("RGB")
    white = Image.new("RGB", result.size, WHITE)
    result.paste(white, mask=mask.convert("L"))
    return result


def convert_green_dataset(
    dataset: Path,
    output: Path | None,
    overwrite: bool,
    validate: bool = True,
    use_existing_masks: bool = False,
    control_ext: str = "png",
) -> dict[str, int | str]:
    source_root = dataset
    out_root = output or dataset
    if output is not None:
        if out_root.exists():
            if not overwrite:
                raise FileExistsError(f"{out_root} already exists. Pass --overwrite to replace it.")
            shutil.rmtree(out_root)
        shutil.copytree(source_root, out_root)

    for child in ["target", "control", "mask"]:
        (out_root / child).mkdir(parents=True, exist_ok=True)

    target_files = sorted(p for p in (out_root / "target").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    control_files = sorted(p for p in (out_root / "control").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    converted = 0
    skipped_already_converted = 0
    used_existing_mask = 0
    missing_mask_source = 0
    if use_existing_masks:
        control_suffix = "." + control_ext.lstrip(".").lower()
        for target_path in target_files:
            mask_path = find_match(out_root / "mask", target_path.stem)
            if mask_path is None:
                missing_mask_source += 1
                continue
            with Image.open(target_path) as target, Image.open(mask_path) as mask_img:
                mask = binary_mask(mask_img)
                white_control = apply_white_fill(target, mask)
                output_control = out_root / "control" / f"{target_path.stem}{control_suffix}"
                if control_suffix in [".jpg", ".jpeg"]:
                    white_control.save(output_control, quality=95, optimize=True)
                else:
                    white_control.save(output_control, optimize=True)
                mask.save(out_root / "mask" / f"{target_path.stem}.png", optimize=True)
                converted += 1

            caption_path = out_root / "target" / f"{target_path.stem}.txt"
            if caption_path.exists():
                caption_path.write_text(outpaint_caption(caption_path.read_text(encoding="utf-8")), encoding="utf-8")
    else:
        for control_path in control_files:
            with Image.open(control_path) as control:
                mask = green_mask_from_control(control)
                if not mask_has_white(mask):
                    existing_mask = next(
                        (out_root / "mask" / f"{control_path.stem}{ext}" for ext in IMAGE_EXTS if (out_root / "mask" / f"{control_path.stem}{ext}").exists()),
                        None,
                    )
                    if existing_mask is not None:
                        skipped_already_converted += 1
                        continue
                    missing_mask_source += 1
                else:
                    mask = binary_mask(mask)
                white_control = apply_white_fill(control, mask)
                white_control.save(control_path, optimize=True)
                (out_root / "mask" / f"{control_path.stem}.png").parent.mkdir(parents=True, exist_ok=True)
                mask.save(out_root / "mask" / f"{control_path.stem}.png", optimize=True)
                converted += 1

            caption_path = out_root / "target" / f"{control_path.stem}.txt"
            if caption_path.exists():
                caption_path.write_text(outpaint_caption(caption_path.read_text(encoding="utf-8")), encoding="utf-8")

    manifest_path = out_root / "manifest.csv"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["fill_color"] = "white"
            row["mask_convention"] = "white_edit_black_preserve"
        write_manifest(manifest_path, rows)

    write_readme(out_root, "Converted Wan VACE White Outpaint Dataset", converted)
    make_contact_sheet(out_root)
    stats = validate_dataset(out_root) if validate else {"target_images": 0, "valid_triplets": 0, "problems": []}
    stats.update(
        {
            "converted_controls": converted,
            "skipped_already_converted": skipped_already_converted,
            "used_existing_masks": used_existing_mask,
            "missing_mask_source": missing_mask_source,
            "output": str(out_root),
        }
    )
    write_validation(out_root, stats)
    return stats


def build_dataset(source: Path, output: Path, limit: int | None, variant_set: str, overwrite: bool) -> dict[str, int | str]:
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists() and overwrite:
        shutil.rmtree(output)
    for child in ["target", "control", "mask"]:
        (output / child).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if limit is not None:
        image_paths = image_paths[:limit]
    attribution = load_attribution(source)

    if variant_set == "all":
        portrait_modes = [False, True]
        side_modes = ["all", "two"]
    elif variant_set == "all_edges":
        portrait_modes = [False, True]
        side_modes = ["all"]
    elif variant_set == "two_side":
        portrait_modes = [False, True]
        side_modes = ["two"]
    elif variant_set == "single_side":
        portrait_modes = [False, True]
        side_modes = ["left", "right", "top", "bottom"]
    else:
        raise ValueError(f"Unknown variant set: {variant_set}")

    existing_rows = []
    manifest_path = output / "manifest.csv"
    if manifest_path.exists() and not overwrite:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
            existing_rows = list(csv.DictReader(f))
    existing_samples = {row.get("sample") for row in existing_rows}

    manifest_rows = []
    failures = []
    for index, image_path in enumerate(image_paths):
        sample_id = f"{index + 1:04d}"
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                caption_base = source_caption(image_path, attribution)
                for portrait in portrait_modes:
                    for side_mode in side_modes:
                        stem = f"{sample_id}_{variant_name(portrait, side_mode)}"
                        if stem in existing_samples:
                            continue
                        manifest_rows.append(
                            render_variant(img, image_path, output, sample_id, index, portrait, side_mode, caption_base)
                        )
        except Exception as exc:
            failures.append({"source": str(image_path), "error": str(exc)})

    combined_rows = existing_rows + manifest_rows
    if combined_rows:
        write_manifest(manifest_path, combined_rows)
    if failures:
        with (output / "failures.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "error"])
            writer.writeheader()
            writer.writerows(failures)

    write_readme(output, "Wan VACE White Outpaint Dataset", len(combined_rows))
    make_contact_sheet(output)
    stats = validate_dataset(output)
    stats.update({"source_images": len(image_paths), "new_samples": len(manifest_rows), "failures": len(failures), "output": str(output)})
    write_validation(output, stats)
    return stats


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_match(folder: Path, stem: str) -> Path | None:
    for ext in ORDERED_IMAGE_EXTS:
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def validate_dataset(root: Path) -> dict[str, int | str | list[str]]:
    target_dir = root / "target"
    control_dir = root / "control"
    mask_dir = root / "mask"
    target_files = sorted(p for p in target_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    problems = []
    valid = 0
    empty_masks = 0
    all_white_masks = 0
    caption_green_refs = 0
    for target_path in target_files:
        control_path = find_match(control_dir, target_path.stem)
        mask_path = find_match(mask_dir, target_path.stem)
        caption_path = target_dir / f"{target_path.stem}.txt"
        if control_path is None:
            problems.append(f"missing control: {target_path.name}")
            continue
        if mask_path is None:
            problems.append(f"missing mask: {target_path.name}")
            continue
        with Image.open(target_path) as target, Image.open(control_path) as control, Image.open(mask_path) as mask:
            if target.size != control.size or target.size != mask.size:
                problems.append(f"dimension mismatch: {target_path.name}")
                continue
            mask_l = binary_mask(mask)
            hist = mask_l.histogram()
            black = hist[0]
            white = hist[255]
            if white == 0:
                empty_masks += 1
                problems.append(f"empty mask: {target_path.name}")
            if black == 0:
                all_white_masks += 1
                problems.append(f"all-white mask: {target_path.name}")
            valid += 1
        if caption_path.exists() and re.search(r"\bgreen\b", caption_path.read_text(encoding="utf-8"), re.IGNORECASE):
            caption_green_refs += 1
            problems.append(f"caption references green: {caption_path.name}")
    return {
        "target_images": len(target_files),
        "valid_triplets": valid,
        "empty_masks": empty_masks,
        "all_white_masks": all_white_masks,
        "caption_green_refs": caption_green_refs,
        "problems": problems[:100],
    }


def write_validation(root: Path, stats: dict) -> None:
    (root / "validation.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def write_readme(root: Path, title: str, sample_count: int) -> None:
    (root / "README.md").write_text(
        f"# {title}\n\n"
        f"Generated paired samples: {sample_count}\n\n"
        "- `target/`: completed training targets with `.txt` captions.\n"
        "- `control/`: matching RGB inputs with white regions to outpaint.\n"
        "- `mask/`: matching grayscale masks where white is editable/reactive and black is preserved/inactive.\n"
        "- Designed for Wan VACE edit/outpaint training.\n",
        encoding="utf-8",
    )


def make_contact_sheet(root: Path, limit: int = 8, thumb_width: int = 240) -> None:
    target_files = sorted(p for p in (root / "target").iterdir() if p.suffix.lower() in IMAGE_EXTS)[:limit]
    if not target_files:
        return
    rows = []
    for target_path in target_files:
        control_path = find_match(root / "control", target_path.stem)
        mask_path = find_match(root / "mask", target_path.stem)
        if control_path is None or mask_path is None:
            continue
        thumbs = []
        for path in [target_path, control_path, mask_path]:
            with Image.open(path) as img:
                img = img.convert("RGB")
                scale = thumb_width / img.width
                thumb = img.resize((thumb_width, max(1, round(img.height * scale))), Image.Resampling.LANCZOS)
                thumbs.append(thumb)
        row_h = max(t.height for t in thumbs) + 24
        row = Image.new("RGB", (thumb_width * 3, row_h), (24, 24, 24))
        draw = ImageDraw.Draw(row)
        for idx, (label, thumb) in enumerate(zip(["target", "control", "mask"], thumbs)):
            x = idx * thumb_width
            row.paste(thumb, (x, 24))
            draw.text((x + 6, 6), label, fill=(255, 255, 255))
        rows.append(row)
    if not rows:
        return
    sheet = Image.new("RGB", (thumb_width * 3, sum(row.height for row in rows)), (16, 16, 16))
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(root / "preview_contact_sheet.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Wan VACE white-mask outpaint datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert-green", help="Convert an existing green outpaint dataset to Wan VACE white-mask format.")
    convert_parser.add_argument("--dataset", required=True)
    convert_parser.add_argument("--output", default=None)
    convert_parser.add_argument("--overwrite", action="store_true")
    convert_parser.add_argument("--no-validate", action="store_true")
    convert_parser.add_argument("--use-existing-masks", action="store_true")
    convert_parser.add_argument("--control-ext", choices=["png", "jpg"], default="png")

    build_parser = subparsers.add_parser("build", help="Build a Wan VACE white-mask outpaint dataset from source images.")
    build_parser.add_argument("--source", required=True)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--limit", type=int, default=None)
    build_parser.add_argument("--variant-set", choices=["all", "all_edges", "two_side", "single_side"], default="all")
    build_parser.add_argument("--overwrite", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate a target/control/mask outpaint dataset.")
    validate_parser.add_argument("--dataset", required=True)

    args = parser.parse_args()
    if args.command == "convert-green":
        stats = convert_green_dataset(
            Path(args.dataset),
            Path(args.output) if args.output else None,
            args.overwrite,
            validate=not args.no_validate,
            use_existing_masks=args.use_existing_masks,
            control_ext=args.control_ext,
        )
    elif args.command == "build":
        stats = build_dataset(Path(args.source), Path(args.output), args.limit, args.variant_set, args.overwrite)
    else:
        root = Path(args.dataset)
        stats = validate_dataset(root)
        make_contact_sheet(root)
        write_validation(root, stats)

    for key, value in stats.items():
        if key == "problems":
            print(f"{key}={len(value)}")
        else:
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
