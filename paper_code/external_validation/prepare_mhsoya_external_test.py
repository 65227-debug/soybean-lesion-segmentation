#!/usr/bin/env python3
"""Build a reproducible 160-image external segmentation test subset.

The selector deliberately does not use predictions from any trained model. It
only applies minimal image-quality checks, removes exact/perceptual duplicates,
groups short capture bursts, and samples across capture dates. Ground-truth
pixel masks still need to be annotated manually after selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SEED = 20260803
TARGET_PER_CLASS = 40
BURST_GAP_SECONDS = 30
TRAIN_PHASH_DISTANCE = 4
SELECTED_PHASH_DISTANCE = 4


@dataclass
class Candidate:
    class_name: str
    prefix: str
    zip_path: str
    member: str
    source_name: str
    captured_at: str
    date_group: str
    scene_id: str
    width: int
    height: int
    sha256: str
    phash64: str
    brightness_mean: float
    contrast_std: float
    dark_clip_fraction: float
    bright_clip_fraction: float
    blur_score: float
    quality_ok: bool = True
    rejection_reason: str = ""
    selected: bool = False
    output_name: str = ""

    @property
    def phash_int(self) -> int:
        return int(self.phash64, 16)


CLASS_SPECS = {
    "Healthy": ("Healthy_Soyabean.zip", "H"),
    "Frogeye_leaf_spot": ("Soyabean_Frog_Leaf_Eye.zip", "F"),
    "Rust": ("Soyabean_Rust.zip", "R"),
    "Septoria_brown_spot": ("Soyabean_Spectoria_Brown_Spot.zip", "B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manual-exclude", type=Path)
    parser.add_argument("--reference-root", type=Path, required=True)
    return parser.parse_args()


def parse_capture_time(name: str) -> datetime | None:
    stem = Path(name).stem
    match = re.match(r"^(\d{8})_(\d{6})", stem)
    if match:
        token = f"{match.group(1)}_{match.group(2)}"
    else:
        # Some released files use 2024102 for 2024-10-02.
        match = re.match(r"^(\d{7})_(\d{6})", stem)
        if not match:
            return None
        short_date = match.group(1)
        token = f"{short_date[:6]}0{short_date[6:]}_{match.group(2)}"
    try:
        return datetime.strptime(token, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def phash64(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))[:8, :8]
    values = dct.flatten()
    median = float(np.median(values[1:]))
    bits = values > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def image_metrics(data: bytes) -> tuple[int, int, int, dict[str, float]]:
    with Image.open(BytesIO(data)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        thumb = image.copy()
        thumb.thumbnail((384, 384), Image.Resampling.LANCZOS)
        rgb = np.asarray(thumb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    metrics = {
        "brightness_mean": float(gray.mean()),
        "contrast_std": float(gray.std()),
        "dark_clip_fraction": float((gray < 10).mean()),
        "bright_clip_fraction": float((gray > 245).mean()),
        "blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }
    return width, height, phash64(gray), metrics


def load_manual_excludes(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    values = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token and not token.startswith("#"):
            values.add(token)
    return values


def load_known_sha256(reference_root: Path) -> set[str]:
    known: set[str] = set()
    metadata_dir = reference_root / "data" / "dadou" / "metadata"
    for name in ("标注总清单.csv", "精标总清单.csv", "ASDID弱监督池.csv"):
        path = metadata_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                digest = (row.get("sha256") or "").strip().lower()
                if len(digest) == 64:
                    known.add(digest)
    return known


def reference_image_paths(reference_root: Path) -> list[Path]:
    roots = [
        reference_root / "data" / "新建文件夹" / "frogeye",
        reference_root / "data" / "新建文件夹" / "target_spot",
        reference_root / "data" / "dadou" / "01_训练集_其余127张" / "images",
        reference_root / "data" / "dadou" / "02_验证集_42张" / "images",
        reference_root / "data" / "dadou" / "03_测试集_49张_请最后标" / "images",
    ]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
    return sorted(set(paths))


def load_reference_phashes(paths: list[Path], cache_path: Path | None = None) -> list[tuple[int, str]]:
    if cache_path is not None and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        expected_paths = [str(path) for path in paths]
        if cached.get("paths") == expected_paths:
            print(f"loaded {len(cached['values'])} cached reference perceptual hashes", flush=True)
            return [(int(value, 16), path) for value, path in cached["values"]]
    values: list[tuple[int, str]] = []
    for index, path in enumerate(paths, start=1):
        try:
            data = path.read_bytes()
            _, _, value, _ = image_metrics(data)
            values.append((value, str(path)))
        except Exception:
            continue
        if index % 500 == 0:
            print(f"reference perceptual hashes: {index}/{len(paths)}", flush=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "paths": [str(path) for path in paths],
            "values": [[f"{value:016x}", path] for value, path in values],
        }), encoding="utf-8")
    return values


def read_candidates(dataset_root: Path) -> tuple[list[Candidate], dict[tuple[str, str], bytes]]:
    candidates: list[Candidate] = []
    cache: dict[tuple[str, str], bytes] = {}
    for class_name, (zip_name, prefix) in CLASS_SPECS.items():
        zip_path = dataset_root / "Soyabean_Leaf_Image_Dataset" / zip_name
        with zipfile.ZipFile(zip_path) as archive:
            entries = [
                item for item in archive.infolist()
                if not item.is_dir() and Path(item.filename).suffix.lower() in IMAGE_SUFFIXES
            ]
            for index, item in enumerate(entries, start=1):
                data = archive.read(item)
                cache[(str(zip_path), item.filename)] = data
                digest = hashlib.sha256(data).hexdigest()
                try:
                    width, height, value, metrics = image_metrics(data)
                except Exception as exc:
                    candidates.append(Candidate(
                        class_name=class_name,
                        prefix=prefix,
                        zip_path=str(zip_path),
                        member=item.filename,
                        source_name=Path(item.filename).name,
                        captured_at="",
                        date_group="unreadable",
                        scene_id="unreadable",
                        width=0,
                        height=0,
                        sha256=digest,
                        phash64="0" * 16,
                        brightness_mean=0.0,
                        contrast_std=0.0,
                        dark_clip_fraction=0.0,
                        bright_clip_fraction=0.0,
                        blur_score=0.0,
                        quality_ok=False,
                        rejection_reason=f"unreadable:{type(exc).__name__}",
                    ))
                    continue
                captured = parse_capture_time(Path(item.filename).name)
                candidates.append(Candidate(
                    class_name=class_name,
                    prefix=prefix,
                    zip_path=str(zip_path),
                    member=item.filename,
                    source_name=Path(item.filename).name,
                    captured_at=captured.isoformat() if captured else "",
                    date_group=captured.strftime("%Y%m%d") if captured else "",
                    scene_id="",
                    width=width,
                    height=height,
                    sha256=digest,
                    phash64=f"{value:016x}",
                    brightness_mean=metrics["brightness_mean"],
                    contrast_std=metrics["contrast_std"],
                    dark_clip_fraction=metrics["dark_clip_fraction"],
                    bright_clip_fraction=metrics["bright_clip_fraction"],
                    blur_score=metrics["blur_score"],
                ))
                if index % 100 == 0:
                    print(f"{class_name}: {index}/{len(entries)}", flush=True)
    return candidates, cache


def assign_scene_ids(candidates: list[Candidate]) -> None:
    by_class: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_class[candidate.class_name].append(candidate)
    for class_name, class_items in by_class.items():
        timed = sorted(
            (datetime.fromisoformat(item.captured_at), item)
            for item in class_items if item.captured_at
        )
        scene_index = 0
        previous_time: datetime | None = None
        for captured, item in timed:
            if (
                previous_time is None
                or captured.date() != previous_time.date()
                or (captured - previous_time).total_seconds() > BURST_GAP_SECONDS
            ):
                scene_index += 1
            item.scene_id = f"{class_name}_{captured:%Y%m%d}_burst_{scene_index:03d}"
            previous_time = captured
        for item in class_items:
            if item.captured_at:
                continue
            stem = Path(item.source_name).stem
            match = re.match(r"^([A-Za-z]+\d+)[-_]", stem)
            group = match.group(1) if match else stem
            item.date_group = group
            item.scene_id = f"{class_name}_{group}"


def hamming(a: int, b: int) -> int:
    # Compatible with the project's older Python runtime.
    return bin(a ^ b).count("1")


def nearest_phash(value: int, references: list[tuple[int, str]], limit: int) -> tuple[int, str] | None:
    best_distance = 65
    best_path = ""
    for reference, path in references:
        distance = hamming(value, reference)
        if distance < best_distance:
            best_distance = distance
            best_path = path
            if distance == 0:
                break
    if best_distance <= limit:
        return best_distance, best_path
    return None


def mark_rejections(
    candidates: list[Candidate],
    known_sha: set[str],
    reference_phashes: list[tuple[int, str]],
    manual_excludes: set[str],
) -> dict[str, float]:
    valid_blur = [item.blur_score for item in candidates if item.quality_ok]
    blur_floor = max(6.0, float(np.percentile(valid_blur, 5)))
    sha_classes: dict[str, set[str]] = defaultdict(set)
    for item in candidates:
        sha_classes[item.sha256].add(item.class_name)

    for item in candidates:
        if not item.quality_ok:
            continue
        reasons: list[str] = []
        if item.source_name in manual_excludes or item.member in manual_excludes:
            reasons.append("manual_exclude")
        if min(item.width, item.height) < 1000:
            reasons.append("resolution_below_1000")
        if not 30.0 <= item.brightness_mean <= 230.0:
            reasons.append("extreme_brightness")
        if item.contrast_std < 15.0:
            reasons.append("very_low_contrast")
        if item.dark_clip_fraction > 0.45 or item.bright_clip_fraction > 0.45:
            reasons.append("severe_clipping")
        if item.blur_score < blur_floor:
            reasons.append("bottom_5pct_sharpness")
        if item.sha256 in known_sha:
            reasons.append("exact_overlap_with_training_pool")
        if len(sha_classes[item.sha256]) > 1:
            reasons.append("same_file_in_multiple_classes")
        if reference_phashes:
            match = nearest_phash(item.phash_int, reference_phashes, TRAIN_PHASH_DISTANCE)
            if match is not None:
                reasons.append(f"near_overlap_training_phash_d{match[0]}")
        if reasons:
            item.quality_ok = False
            item.rejection_reason = ";".join(reasons)
    return {"blur_floor": blur_floor}


def choose_subset(candidates: list[Candidate]) -> list[Candidate]:
    selected: list[Candidate] = []
    selected_hashes: list[int] = []
    selected_sha: set[str] = set()

    for class_index, class_name in enumerate(CLASS_SPECS):
        class_items = [
            item for item in candidates
            if item.class_name == class_name and item.quality_ok
        ]
        if len(class_items) < TARGET_PER_CLASS:
            raise RuntimeError(f"Only {len(class_items)} valid candidates for {class_name}")

        by_scene: dict[str, list[Candidate]] = defaultdict(list)
        for item in class_items:
            by_scene[item.scene_id].append(item)

        class_blur = np.array([item.blur_score for item in class_items], dtype=np.float64)
        median_blur = float(np.median(class_blur))
        for scene_items in by_scene.values():
            # Prefer an annotatable representative within a burst, without using
            # lesion extent or any model output as a selection signal.
            scene_items.sort(key=lambda item: (
                abs(np.log1p(item.blur_score) - np.log1p(median_blur)),
                abs(item.brightness_mean - 128.0),
                item.source_name,
            ))

        by_date: dict[str, list[str]] = defaultdict(list)
        for scene_id, scene_items in by_scene.items():
            by_date[scene_items[0].date_group].append(scene_id)
        rng = random.Random(SEED + class_index * 1009)
        date_groups = sorted(by_date)
        rng.shuffle(date_groups)
        for date_group in date_groups:
            rng.shuffle(by_date[date_group])

        scene_offsets = defaultdict(int)
        class_selected: list[Candidate] = []
        while len(class_selected) < TARGET_PER_CLASS:
            progress = False
            for date_group in date_groups:
                scenes = by_date[date_group]
                while scene_offsets[date_group] < len(scenes):
                    scene_id = scenes[scene_offsets[date_group]]
                    scene_offsets[date_group] += 1
                    picked = None
                    for item in by_scene[scene_id]:
                        if item.sha256 in selected_sha:
                            continue
                        if any(
                            hamming(item.phash_int, existing) <= SELECTED_PHASH_DISTANCE
                            for existing in selected_hashes
                        ):
                            continue
                        picked = item
                        break
                    if picked is not None:
                        class_selected.append(picked)
                        selected.append(picked)
                        selected_sha.add(picked.sha256)
                        selected_hashes.append(picked.phash_int)
                        progress = True
                        break
                if len(class_selected) >= TARGET_PER_CLASS:
                    break
            if not progress:
                raise RuntimeError(f"Could not select {TARGET_PER_CLASS} diverse images for {class_name}")

        class_selected.sort(key=lambda item: (item.date_group, item.captured_at, item.source_name))
        prefix = CLASS_SPECS[class_name][1]
        for index, item in enumerate(class_selected, start=1):
            item.selected = True
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", item.source_name)
            item.output_name = f"{prefix}{index:03d}_{safe_name}"
    return selected


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_contact_sheet(class_name: str, items: list[Candidate], cache: dict[tuple[str, str], bytes], path: Path) -> None:
    columns, rows = 5, 8
    tile_w, tile_h = 300, 250
    header_h = 56
    sheet = Image.new("RGB", (columns * tile_w, header_h + rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(26)
    label_font = load_font(15)
    draw.text((12, 12), f"{class_name} - {len(items)} selected images", fill="black", font=title_font)
    for index, item in enumerate(items):
        row, column = divmod(index, columns)
        x, y = column * tile_w, header_h + row * tile_h
        with Image.open(BytesIO(cache[(item.zip_path, item.member)])) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((tile_w - 8, tile_h - 42), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile_w - 8, tile_h - 42), (230, 230, 230))
        px = (canvas.width - image.width) // 2
        py = (canvas.height - image.height) // 2
        canvas.paste(image, (px, py))
        sheet.paste(canvas, (x + 4, y + 4))
        label = f"{item.output_name[:28]}  {item.date_group}"
        draw.text((x + 6, y + tile_h - 32), label, fill="black", font=label_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def write_outputs(
    output: Path,
    candidates: list[Candidate],
    selected: list[Candidate],
    cache: dict[tuple[str, str], bytes],
    selection_settings: dict[str, float],
    reference_count: int,
) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metadata_dir = output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_SPECS:
        class_dir = output / "images" / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        class_items = sorted(
            (item for item in selected if item.class_name == class_name),
            key=lambda item: item.output_name,
        )
        for item in class_items:
            (class_dir / item.output_name).write_bytes(cache[(item.zip_path, item.member)])
        make_contact_sheet(
            class_name,
            class_items,
            cache,
            metadata_dir / "contact_sheets" / f"{class_name}.jpg",
        )

    fieldnames = list(asdict(candidates[0]).keys())
    with (metadata_dir / "candidate_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(item) for item in candidates)
    with (metadata_dir / "selection_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(item) for item in sorted(selected, key=lambda x: (x.class_name, x.output_name)))

    summary = {
        "seed": SEED,
        "target_per_class": TARGET_PER_CLASS,
        "burst_gap_seconds": BURST_GAP_SECONDS,
        "training_phash_distance": TRAIN_PHASH_DISTANCE,
        "selected_phash_distance": SELECTED_PHASH_DISTANCE,
        "reference_images_hashed": reference_count,
        "selection_settings": selection_settings,
        "candidate_counts": dict(Counter(item.class_name for item in candidates)),
        "valid_candidate_counts": dict(Counter(item.class_name for item in candidates if item.quality_ok)),
        "selected_counts": dict(Counter(item.class_name for item in selected)),
        "selected_date_counts": {
            class_name: dict(Counter(item.date_group for item in selected if item.class_name == class_name))
            for class_name in CLASS_SPECS
        },
        "rejection_reason_counts": dict(Counter(
            reason
            for item in candidates if item.rejection_reason
            for reason in item.rejection_reason.split(";")
        )),
    }
    (metadata_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = f"""# MH-SoyaHealthVision 外部测试子集（待标注）

本目录含 160 张原始分辨率图像，每类 40 张：Healthy、Frogeye leaf spot、Rust、Septoria brown spot。

选择过程未使用任何现有模型的预测结果。筛选仅使用最低质量检查、与现有精标/弱监督池的哈希排重、30 秒连续拍摄分组和跨日期分层抽样。随机种子为 {SEED}。

## 重要说明

- 当前目录只有图像级类别，尚无像素级真值，不能直接计算分割 mIoU。
- 后续人工掩膜建议使用：0=背景，1=健康叶片组织，2=病斑，255=不确定/忽略。
- 这些图像不得用于训练、阈值选择、校准或挑选最佳模型；标注完成后应冻结模型并只评估一次。
- 主实验建议使用叶片近景；UAV 图像属于不同尺度任务，不与本子集混合计分。
- 数据来源：MH-SoyaHealthVision V1, DOI 10.17632/hkbgh5s3b7.1, CC BY 4.0。

## 文件说明

- `images/`：按四个图像级类别分目录保存的原始图像。
- `metadata/selection_manifest.csv`：160 张图像与原压缩包成员的对应关系。
- `metadata/candidate_audit.csv`：所有候选图像的质量指标和排除原因。
- `metadata/selection_summary.json`：选择参数与统计汇总。
- `metadata/contact_sheets/`：人工复核联系表。
"""
    (output / "README_CN.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(args.dataset_root)
    known_sha = load_known_sha256(args.reference_root)
    reference_paths = reference_image_paths(args.reference_root)
    print(f"known SHA-256 values: {len(known_sha)}", flush=True)
    print(f"reference images for perceptual deduplication: {len(reference_paths)}", flush=True)
    reference_phashes = load_reference_phashes(
        reference_paths,
        args.reference_root / "work_dirs" / "mhsoya_external_selection_cache" / "reference_phashes.json",
    )
    candidates, cache = read_candidates(args.dataset_root)
    assign_scene_ids(candidates)
    settings = mark_rejections(
        candidates,
        known_sha,
        reference_phashes,
        load_manual_excludes(args.manual_exclude),
    )
    selected = choose_subset(candidates)
    write_outputs(
        args.output,
        candidates,
        selected,
        cache,
        settings,
        len(reference_phashes),
    )
    print(json.dumps(Counter(item.class_name for item in selected), ensure_ascii=False), flush=True)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
