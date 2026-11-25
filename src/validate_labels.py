"""Валидатор изображений и YOLO-разметки (включая проверку после экспорта из CVAT).

Поддерживает как плоскую структуру (`images/`, `labels/`, `classes.txt`),
так и сплитованную структуру (`images/{train,val,test}`, `labels/{train,val,test}`, `dataset.yaml`).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.utils import (
    list_image_files,
    list_label_files,
    load_classes,
    load_yaml,
    normalize_stem,
    save_json,
    setup_logging,
    sha256_file,
)

logger = logging.getLogger("sts_cv.validate")

EXIF_ORIENTATION_TAG = 274
BOUNDS_EPS = 1e-4


@dataclass(frozen=True)
class BBox:
    """Нормализованный bounding box в формате Ultralytics YOLO."""

    class_id: int
    xc: float
    yc: float
    w: float
    h: float
    line_no: int = 0

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Преобразует (xc, yc, w, h) в (x1, y1, x2, y2)."""
        half_w = self.w / 2.0
        half_h = self.h / 2.0
        return (self.xc - half_w, self.yc - half_h, self.xc + half_w, self.yc + half_h)

    def area(self) -> float:
        """Возвращает нормализованную площадь бокса."""
        return max(0.0, self.w) * max(0.0, self.h)

    def iou(self, other: BBox) -> float:
        """Вычисляет Intersection over Union двух боксов."""
        ax1, ay1, ax2, ay2 = self.to_xyxy()
        bx1, by1, bx2, by2 = other.to_xyxy()
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        union = self.area() + other.area() - inter
        return inter / union if union > 0.0 else 0.0

    def clip_to_unit(self) -> tuple[BBox, bool]:
        """Обрезает бокс в диапазон [0, 1] с корректным пересчётом центра и размеров."""
        x1, y1, x2, y2 = self.to_xyxy()
        cx1 = min(max(x1, 0.0), 1.0)
        cy1 = min(max(y1, 0.0), 1.0)
        cx2 = min(max(x2, 0.0), 1.0)
        cy2 = min(max(y2, 0.0), 1.0)
        changed = any(
            abs(a - b) > 1e-7 for a, b in ((x1, cx1), (y1, cy1), (x2, cx2), (y2, cy2))
        )
        nw = max(0.0, cx2 - cx1)
        nh = max(0.0, cy2 - cy1)
        nxc = (cx1 + cx2) / 2.0
        nyc = (cy1 + cy2) / 2.0
        return BBox(self.class_id, nxc, nyc, nw, nh, self.line_no), changed

    def to_yolo_line(self) -> str:
        """Форматирует бокс в строку стандарта YOLO."""
        return f"{self.class_id} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class ValidationIssue:
    """Единичная найденная проблема валидации."""

    severity: str  # "ERROR" | "WARNING"
    code: str
    file: str
    line_no: int | None
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageRecord:
    """Диагностическая карточка одного изображения и его аннотаций."""

    filename: str
    stem: str
    split: str
    image_path: str
    label_path: str | None
    is_valid_image: bool
    width: int = 0
    height: int = 0
    img_format: str = ""
    mode: str = ""
    file_size_bytes: int = 0
    sha256: str = ""
    dhash: int = 0
    exif_orientation: int = 1
    black_mask_ratio: float = 0.0
    has_label_file: bool = False
    raw_line_count: int = 0
    boxes: list[BBox] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Итоговый структурированный отчёт валидации датасета."""

    dataset_dir: str
    classes: list[str]
    num_images: int
    num_valid_images: int
    num_label_files: int
    num_missing_label_files: int
    num_orphan_label_files: int
    num_empty_label_files: int
    num_total_boxes: int
    class_counts: dict[str, int]
    issues: list[ValidationIssue] = field(default_factory=list)
    records: list[ImageRecord] = field(default_factory=list)
    near_duplicate_pairs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    def counts_by_code(self) -> dict[str, int]:
        return dict(Counter(i.code for i in self.issues))

    def is_valid(self, strict: bool = False) -> bool:
        if self.errors:
            return False
        return not (strict and self.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": self.dataset_dir,
            "classes": self.classes,
            "num_images": self.num_images,
            "num_valid_images": self.num_valid_images,
            "num_label_files": self.num_label_files,
            "num_missing_label_files": self.num_missing_label_files,
            "num_orphan_label_files": self.num_orphan_label_files,
            "num_empty_label_files": self.num_empty_label_files,
            "num_total_boxes": self.num_total_boxes,
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
            "counts_by_code": self.counts_by_code(),
            "class_counts": self.class_counts,
            "near_duplicate_pairs": self.near_duplicate_pairs,
            "issues": [asdict(i) for i in self.issues],
        }


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Вычисляет 64-битный разностный перцептивный хэш (dHash) изображения."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.flatten()
    val = 0
    for bit in bits:
        val = (val << 1) | int(bit)
    return val


def hamming_distance(a: int, b: int) -> int:
    """Расстояние Хэмминга между двумя целочисленными хэшами."""
    return (a ^ b).bit_count()


def detect_magic_format(header: bytes) -> str:
    """Определяет реальный формат изображения по сигнатуре (magic bytes)."""
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if header.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if header.startswith(b"BM"):
        return "BMP"
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
        return "WEBP"
    return "UNKNOWN"


def verify_class_schema(
    actual_classes: list[str],
    reference_classes: list[str],
) -> list[ValidationIssue]:
    """Сверяет порядок и ID классов (защита от сдвига ID после экспорта из CVAT)."""
    issues: list[ValidationIssue] = []
    if actual_classes == reference_classes:
        return issues

    if len(actual_classes) != len(reference_classes):
        issues.append(
            ValidationIssue(
                severity="ERROR",
                code="CLASS_MAPPING_MISMATCH",
                file="classes.txt",
                line_no=None,
                message=(
                    f"Несовпадение числа классов: получено {len(actual_classes)}, "
                    f"ожидалось {len(reference_classes)}"
                ),
                details={"actual": actual_classes, "expected": reference_classes},
            )
        )
        return issues

    mismatches = [
        (idx, a, b)
        for idx, (a, b) in enumerate(zip(actual_classes, reference_classes, strict=True))
        if a != b
    ]
    issues.append(
        ValidationIssue(
            severity="ERROR",
            code="CLASS_MAPPING_MISMATCH",
            file="classes.txt",
            line_no=None,
            message=f"Сдвиг или несовпадение ID классов: {mismatches}",
            details={"mismatches": mismatches},
        )
    )
    return issues


def parse_yolo_label_file(
    label_path: Path,
    num_classes: int,
    rel_name: str,
    tiny_area_threshold: float = 1e-5,
    dup_iou_threshold: float = 0.85,
) -> tuple[list[BBox], list[ValidationIssue], int]:
    """Парсит и валидирует один .txt файл разметки YOLO."""
    issues: list[ValidationIssue] = []
    boxes: list[BBox] = []

    raw_text = label_path.read_text(encoding="utf-8", errors="replace")
    lines = raw_text.splitlines()
    non_empty_lines = [ln for ln in lines if ln.strip()]
    if not non_empty_lines:
        issues.append(
            ValidationIssue(
                severity="WARNING",
                code="EMPTY_LABEL_FILE",
                file=rel_name,
                line_no=None,
                message="Файл разметки пуст (0 боксов; убедитесь, что кадр действительно фоновый).",
            )
        )
        return boxes, issues, 0

    seen_exact: set[str] = set()
    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    code="BLANK_LINE",
                    file=rel_name,
                    line_no=line_no,
                    message="Пустая строка внутри файла разметки.",
                )
            )
            continue

        if stripped in seen_exact:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    code="EXACT_DUPLICATE_BOX",
                    file=rel_name,
                    line_no=line_no,
                    message=f"Точный дубликат строки разметки: '{stripped}'",
                )
            )
        seen_exact.add(stripped)

        parts = stripped.split()
        if len(parts) != 5:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="BAD_TOKEN_COUNT",
                    file=rel_name,
                    line_no=line_no,
                    message=f"Ожидалось 5 полей (cls xc yc w h), получено {len(parts)}: '{stripped}'",
                )
            )
            continue

        try:
            class_id = int(parts[0])
            if str(class_id) != parts[0]:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="INVALID_CLASS_FORMAT",
                        file=rel_name,
                        line_no=line_no,
                        message=f"Некорректный формат class_id: '{parts[0]}'",
                    )
                )
                continue
        except ValueError:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="INVALID_CLASS_FORMAT",
                    file=rel_name,
                    line_no=line_no,
                    message=f"class_id не является целым числом: '{parts[0]}'",
                )
            )
            continue

        try:
            xc, yc, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="NON_FINITE_COORD",
                    file=rel_name,
                    line_no=line_no,
                    message=f"Координаты не являются числами: '{stripped}'",
                )
            )
            continue

        if not all(math.isfinite(v) for v in (xc, yc, w, h)):
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="NON_FINITE_COORD",
                    file=rel_name,
                    line_no=line_no,
                    message=f"Координаты содержат NaN/Inf: '{stripped}'",
                )
            )
            continue

        row_has_error = False
        if class_id < 0 or class_id >= num_classes:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="UNKNOWN_CLASS_ID",
                    file=rel_name,
                    line_no=line_no,
                    message=(
                        f"Неизвестный class_id={class_id} "
                        f"(допустимый диапазон 0..{num_classes - 1})"
                    ),
                    details={"class_id": class_id, "line": stripped},
                )
            )
            row_has_error = True

        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="CENTER_OUT_OF_RANGE",
                    file=rel_name,
                    line_no=line_no,
                    message=f"Центр бокса вне [0, 1]: xc={xc:.6f}, yc={yc:.6f}",
                    details={"xc": xc, "yc": yc},
                )
            )
            row_has_error = True

        if w <= 0.0 or h <= 0.0:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="SIZE_NON_POSITIVE",
                    file=rel_name,
                    line_no=line_no,
                    message=f"Неположительные размеры бокса: w={w:.6f}, h={h:.6f}",
                )
            )
            row_has_error = True
        elif w > 1.0 + BOUNDS_EPS or h > 1.0 + BOUNDS_EPS:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="SIZE_OUT_OF_RANGE",
                    file=rel_name,
                    line_no=line_no,
                    message=f"Размеры бокса превышают 1.0: w={w:.6f}, h={h:.6f}",
                )
            )
            row_has_error = True

        box = BBox(class_id=class_id, xc=xc, yc=yc, w=w, h=h, line_no=line_no)
        x1, y1, x2, y2 = box.to_xyxy()
        if (
            x1 < -BOUNDS_EPS
            or y1 < -BOUNDS_EPS
            or x2 > 1.0 + BOUNDS_EPS
            or y2 > 1.0 + BOUNDS_EPS
        ):
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="BOX_EXCEEDS_BOUNDS",
                    file=rel_name,
                    line_no=line_no,
                    message=(
                        f"Границы бокса выходят за [0, 1]: "
                        f"x1={x1:.4f}, y1={y1:.4f}, x2={x2:.4f}, y2={y2:.4f}"
                    ),
                    details={"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_id": class_id},
                )
            )
            row_has_error = True

        if not row_has_error and box.area() < tiny_area_threshold:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    code="TINY_BOX",
                    file=rel_name,
                    line_no=line_no,
                    message=f"Подозрительно малая площадь бокса: area={box.area():.6f}",
                )
            )

        if not row_has_error:
            boxes.append(box)

    # Проверка почти-дублирующихся боксов одного класса (IoU >= dup_iou_threshold)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i].class_id == boxes[j].class_id:
                iou_val = boxes[i].iou(boxes[j])
                if iou_val >= dup_iou_threshold:
                    issues.append(
                        ValidationIssue(
                            severity="WARNING",
                            code="NEAR_DUPLICATE_BOX",
                            file=rel_name,
                            line_no=boxes[j].line_no,
                            message=(
                                f"Почти дублирующийся бокс класса {boxes[j].class_id} "
                                f"(строки {boxes[i].line_no} и {boxes[j].line_no}, IoU={iou_val:.3f})"
                            ),
                            details={
                                "line_a": boxes[i].line_no,
                                "line_b": boxes[j].line_no,
                                "iou": round(iou_val, 4),
                            },
                        )
                    )

    return boxes, issues, len(non_empty_lines)


def _load_classes_from_dataset(dataset_dir: Path, classes_path: Path | None) -> list[str]:
    """Определяет список классов из classes.txt или dataset.yaml."""
    if classes_path is not None:
        return load_classes(classes_path)

    txt_candidate = dataset_dir / "classes.txt"
    if txt_candidate.exists():
        return load_classes(txt_candidate)

    yaml_candidate = dataset_dir / "dataset.yaml"
    if yaml_candidate.exists():
        cfg = load_yaml(yaml_candidate)
        names = cfg.get("names")
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names.keys(), key=int)]
        if isinstance(names, list):
            return [str(x) for x in names]

    raise FileNotFoundError(
        f"Не найден ни classes.txt, ни dataset.yaml с полем names в {dataset_dir}"
    )


def _discover_splits(dataset_dir: Path) -> list[tuple[str, Path, Path]]:
    """Находит пары (split_name, images_dir, labels_dir) для плоской или сплитованной структуры."""
    images_root = dataset_dir / "images"
    labels_root = dataset_dir / "labels"
    if not images_root.exists():
        raise FileNotFoundError(f"Отсутствует каталог images/: {images_root}")

    split_subdirs = [
        name for name in ("train", "val", "test") if (images_root / name).is_dir()
    ]
    if split_subdirs:
        return [
            (split, images_root / split, labels_root / split) for split in split_subdirs
        ]
    return [("all", images_root, labels_root)]


def validate_dataset(
    dataset_dir: Path,
    classes_path: Path | None = None,
    reference_classes_path: Path | None = None,
    near_dup_hamming_threshold: int = 8,
    check_black_mask_boxes: bool = True,
    missing_label_severity: str = "ERROR",
) -> ValidationReport:
    """Выполняет полную проверку изображений, YOLO-разметки, дубликатов и схемы классов."""
    dataset_dir = dataset_dir.resolve()
    classes = _load_classes_from_dataset(dataset_dir, classes_path)
    issues: list[ValidationIssue] = []

    if reference_classes_path is not None:
        ref_classes = load_classes(reference_classes_path)
        issues.extend(verify_class_schema(classes, ref_classes))

    splits = _discover_splits(dataset_dir)
    records: list[ImageRecord] = []
    sha_map: dict[str, list[str]] = defaultdict(list)
    class_counter: Counter[int] = Counter()

    total_label_files = 0
    missing_label_count = 0
    orphan_label_count = 0
    empty_label_count = 0

    for split_name, images_dir, labels_dir in splits:
        img_files = list_image_files(images_dir)
        lbl_files = list_label_files(labels_dir)
        total_label_files += len(lbl_files)

        img_by_stem: dict[str, list[Path]] = defaultdict(list)
        for p in img_files:
            img_by_stem[normalize_stem(p.stem)].append(p)

        for stem, paths in img_by_stem.items():
            if len(paths) > 1:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="IMAGE_DUPLICATE_STEM",
                        file=f"{split_name}/{stem}",
                        line_no=None,
                        message=f"Несколько изображений с одинаковым stem '{stem}': {[x.name for x in paths]}",
                    )
                )

        lbl_by_stem: dict[str, Path] = {normalize_stem(p.stem): p for p in lbl_files}

        # Осиротевшие .txt без изображения
        for stem, lbl_path in sorted(lbl_by_stem.items()):
            if stem not in img_by_stem:
                orphan_label_count += 1
                rel_lbl = f"{split_name}/{lbl_path.name}" if split_name != "all" else lbl_path.name
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="ORPHAN_LABEL_FILE",
                        file=rel_lbl,
                        line_no=None,
                        message=f"Файл разметки без соответствующего изображения: {rel_lbl}",
                    )
                )

        # Проверка каждого изображения и его .txt
        for img_path in img_files:
            stem = normalize_stem(img_path.stem)
            rel_img = f"{split_name}/{img_path.name}" if split_name != "all" else img_path.name
            file_size = img_path.stat().st_size
            file_sha = sha256_file(img_path)
            sha_map[file_sha].append(rel_img)

            lbl_path = lbl_by_stem.get(stem)
            rec = ImageRecord(
                filename=img_path.name,
                stem=stem,
                split=split_name,
                image_path=str(img_path),
                label_path=str(lbl_path) if lbl_path else None,
                is_valid_image=False,
                file_size_bytes=file_size,
                sha256=file_sha,
                has_label_file=lbl_path is not None,
            )

            # 1. Проверка файла изображения
            if file_size == 0:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="IMAGE_EMPTY",
                        file=rel_img,
                        line_no=None,
                        message="Файл изображения имеет нулевой размер (0 байт).",
                    )
                )
                records.append(rec)
                continue

            with img_path.open("rb") as f:
                header = f.read(16)
            magic_fmt = detect_magic_format(header)

            rgb_arr: np.ndarray | None = None
            try:
                with Image.open(img_path) as im_verify:
                    im_verify.verify()
                with Image.open(img_path) as im:
                    exif = im.getexif()
                    orientation = int(exif.get(EXIF_ORIENTATION_TAG, 1)) if exif else 1
                    rec.exif_orientation = orientation
                    if orientation != 1:
                        issues.append(
                            ValidationIssue(
                                severity="WARNING",
                                code="IMAGE_EXIF_ORIENTATION",
                                file=rel_img,
                                line_no=None,
                                message=(
                                    f"Нестандартная EXIF-ориентация ({orientation} != 1): "
                                    "возможен сдвиг координат между PIL и OpenCV."
                                ),
                                details={"orientation": orientation},
                            )
                        )
                    im_norm = ImageOps.exif_transpose(im)
                    im_norm.load()
                    rec.width, rec.height = im_norm.size
                    rec.img_format = im.format or magic_fmt
                    rec.mode = im_norm.mode
                    rec.dhash = compute_dhash(im_norm)
                    rgb_arr = np.asarray(im_norm.convert("RGB"))
                    black_mask = (rgb_arr[:, :, 0] <= 2) & (rgb_arr[:, :, 1] <= 2) & (rgb_arr[:, :, 2] <= 2)
                    rec.black_mask_ratio = float(black_mask.mean())
                    rec.is_valid_image = True

                    ext_expected = {
                        ".jpg": "JPEG",
                        ".jpeg": "JPEG",
                        ".png": "PNG",
                        ".bmp": "BMP",
                        ".webp": "WEBP",
                    }.get(img_path.suffix.lower(), "UNKNOWN")
                    if magic_fmt != "UNKNOWN" and ext_expected != magic_fmt:
                        issues.append(
                            ValidationIssue(
                                severity="WARNING",
                                code="IMAGE_FORMAT_MISMATCH",
                                file=rel_img,
                                line_no=None,
                                message=(
                                    f"Расширение '{img_path.suffix}' не совпадает с фактическим "
                                    f"форматом файла ({magic_fmt})."
                                ),
                            )
                        )
            except Exception as exc:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        code="IMAGE_CORRUPT",
                        file=rel_img,
                        line_no=None,
                        message=f"Битое или нечитаемое изображение: {exc}",
                    )
                )
                records.append(rec)
                continue

            # 2. Проверка файла разметки
            if lbl_path is None:
                missing_label_count += 1
                issues.append(
                    ValidationIssue(
                        severity=missing_label_severity,
                        code="MISSING_LABEL_FILE",
                        file=rel_img,
                        line_no=None,
                        message=(
                            "Отсутствует соответствующий .txt файл разметки "
                            "(не считать автоматически фоном без визуальной проверки)."
                        ),
                    )
                )
            else:
                rel_lbl = f"{split_name}/{lbl_path.name}" if split_name != "all" else lbl_path.name
                boxes, lbl_issues, raw_lines = parse_yolo_label_file(
                    label_path=lbl_path,
                    num_classes=len(classes),
                    rel_name=rel_lbl,
                )
                rec.boxes = boxes
                rec.raw_line_count = raw_lines
                if raw_lines == 0:
                    empty_label_count += 1
                issues.extend(lbl_issues)

                for b in boxes:
                    class_counter[b.class_id] += 1

                # Проверка попадания боксов в чёрную ROI-маску
                if check_black_mask_boxes and rgb_arr is not None and boxes:
                    h_img, w_img = rgb_arr.shape[:2]
                    for b in boxes:
                        x1, y1, x2, y2 = b.to_xyxy()
                        px1 = max(0, min(w_img - 1, round(x1 * w_img)))
                        py1 = max(0, min(h_img - 1, round(y1 * h_img)))
                        px2 = max(px1 + 1, min(w_img, round(x2 * w_img)))
                        py2 = max(py1 + 1, min(h_img, round(y2 * h_img)))
                        crop = rgb_arr[py1:py2, px1:px2]
                        if crop.size > 0:
                            black_frac = float(
                                ((crop[:, :, 0] <= 2) & (crop[:, :, 1] <= 2) & (crop[:, :, 2] <= 2)).mean()
                            )
                            if black_frac >= 0.85:
                                issues.append(
                                    ValidationIssue(
                                        severity="WARNING",
                                        code="BOX_IN_BLACK_MASK",
                                        file=rel_lbl,
                                        line_no=b.line_no,
                                        message=(
                                            f"Бокс класса {b.class_id} на {black_frac*100:.1f}% "
                                            "попадает в чёрную маску кадра."
                                        ),
                                    )
                                )

            records.append(rec)

    # 3. Точные дубликаты изображений (SHA-256)
    for digest, files in sha_map.items():
        if len(files) > 1:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    code="IMAGE_EXACT_DUPLICATE",
                    file=", ".join(files),
                    line_no=None,
                    message=f"Обнаружены побайтовые дубликаты изображений (sha256={digest[:12]}): {files}",
                )
            )

    # 4. Почти-дубликаты / кадры одной камеры по dHash
    near_dups: list[dict[str, Any]] = []
    valid_recs = [r for r in records if r.is_valid_image]
    for i in range(len(valid_recs)):
        for j in range(i + 1, len(valid_recs)):
            dist = hamming_distance(valid_recs[i].dhash, valid_recs[j].dhash)
            if dist <= near_dup_hamming_threshold:
                near_dups.append(
                    {
                        "image_a": valid_recs[i].filename,
                        "image_b": valid_recs[j].filename,
                        "split_a": valid_recs[i].split,
                        "split_b": valid_recs[j].split,
                        "hamming_distance": dist,
                    }
                )
                if valid_recs[i].split != valid_recs[j].split:
                    issues.append(
                        ValidationIssue(
                            severity="WARNING",
                            code="IMAGE_NEAR_DUPLICATE_CROSS_SPLIT",
                            file=f"{valid_recs[i].filename} <-> {valid_recs[j].filename}",
                            line_no=None,
                            message=(
                                f"Визуально близкие кадры (dHash dist={dist}) попали в разные "
                                f"сплиты: {valid_recs[i].split} и {valid_recs[j].split}."
                            ),
                        )
                    )

    class_counts = {name: int(class_counter.get(idx, 0)) for idx, name in enumerate(classes)}
    return ValidationReport(
        dataset_dir=str(dataset_dir),
        classes=classes,
        num_images=len(records),
        num_valid_images=len(valid_recs),
        num_label_files=total_label_files,
        num_missing_label_files=missing_label_count,
        num_orphan_label_files=orphan_label_count,
        num_empty_label_files=empty_label_count,
        num_total_boxes=sum(class_counts.values()),
        class_counts=class_counts,
        issues=issues,
        records=records,
        near_duplicate_pairs=near_dups,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Валидация изображений и YOLO-разметки (включая проверку после экспорта из CVAT)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Каталог датасета (с images/, labels/ и classes.txt или dataset.yaml).",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=None,
        help="Необязательный путь к classes.txt (если лежит отдельно).",
    )
    parser.add_argument(
        "--reference-classes",
        type=Path,
        default=None,
        help="Эталонный classes.txt для сверки соответствия ID классов после экспорта из CVAT.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Возвращать ненулевой код выхода не только при ERROR, но и при WARNING.",
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Считать отсутствие .txt предупреждением вместо ошибки.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Сохранить полный отчёт валидации в JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    report = validate_dataset(
        dataset_dir=args.input,
        classes_path=args.classes,
        reference_classes_path=args.reference_classes,
        missing_label_severity="WARNING" if args.allow_missing_labels else "ERROR",
    )

    logger.info(
        "Проверено изображений: %d (валидных: %d), файлов разметки: %d, боксов: %d",
        report.num_images,
        report.num_valid_images,
        report.num_label_files,
        report.num_total_boxes,
    )
    logger.info(
        "Ошибки (ERROR): %d | Предупреждения (WARNING): %d | По кодам: %s",
        len(report.errors),
        len(report.warnings),
        report.counts_by_code(),
    )

    for issue in report.issues[:30]:
        loc = f"{issue.file}:{issue.line_no}" if issue.line_no else issue.file
        lvl = logging.ERROR if issue.severity == "ERROR" else logging.WARNING
        logger.log(lvl, "[%s] %s — %s", issue.code, loc, issue.message)
    if len(report.issues) > 30:
        logger.info("... и ещё %d записей (сохраните --json-out для полного списка).", len(report.issues) - 30)

    if args.json_out is not None:
        save_json(report.to_dict(), args.json_out)
        logger.info("JSON-отчёт сохранён: %s", args.json_out)

    return 0 if report.is_valid(strict=args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
