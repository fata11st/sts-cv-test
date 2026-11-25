"""Юнит-тесты валидатора изображений и YOLO-разметки (src.validate_labels)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.validate_labels import BBox, validate_dataset, verify_class_schema


def _create_dummy_image(path: Path, size: tuple[int, int] = (120, 80), color: tuple[int, int, int] = (80, 120, 160)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    # Добавляем простую текстуру, чтобы dHash отличался от чистого фона
    arr[10:30, 20:50] = (200, 180, 60)
    Image.fromarray(arr, mode="RGB").save(path)


def _setup_mini_dataset(tmp_path: Path, classes: list[str] | None = None) -> Path:
    ds = tmp_path / "mini_ds"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    cls_list = classes or ["bicycle", "car", "bus", "pedestrian"]
    (ds / "classes.txt").write_text("\n".join(cls_list) + "\n", encoding="utf-8")
    return ds


def test_valid_dataset_with_cyrillic_and_spaces(tmp_path: Path) -> None:
    """Корректная разметка с кириллическими именами и пробелами проходит строгую валидацию."""
    ds = _setup_mini_dataset(tmp_path)
    stem = "Снимок экрана от 2023-06-05 14-24-28"
    _create_dummy_image(ds / "images" / f"{stem}.png")
    (ds / "labels" / f"{stem}.txt").write_text(
        "0 0.500000 0.500000 0.200000 0.300000\n"
        "3 0.200000 0.700000 0.100000 0.200000\n",
        encoding="utf-8",
    )

    report = validate_dataset(ds)
    assert report.is_valid(strict=True)
    assert report.num_images == 1
    assert report.num_total_boxes == 2
    assert report.class_counts["bicycle"] == 1
    assert report.class_counts["pedestrian"] == 1


def test_detects_coordinate_and_class_errors(tmp_path: Path) -> None:
    """Валидатор выявляет выход за [0,1], неизвестный класс, NaN, неверное число полей и <=0 размеры."""
    ds = _setup_mini_dataset(tmp_path)
    _create_dummy_image(ds / "images" / "sample_bad.jpg")
    (ds / "labels" / "sample_bad.txt").write_text(
        "\n".join(
            [
                "99 0.5 0.5 0.2 0.2",        # UNKNOWN_CLASS_ID
                "1 0.95 0.5 0.2 0.2",        # BOX_EXCEEDS_BOUNDS (x2=1.05)
                "1 0.5 -0.1 0.2 0.2",        # CENTER_OUT_OF_RANGE + BOX_EXCEEDS_BOUNDS
                "1 0.5 0.5 0.0 0.2",         # SIZE_NON_POSITIVE
                "1 0.5 nan 0.2 0.2",         # NON_FINITE_COORD
                "1 0.5 0.5 0.2",             # BAD_TOKEN_COUNT
                "1.5 0.5 0.5 0.2 0.2",       # INVALID_CLASS_FORMAT
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_dataset(ds)
    codes = set(report.counts_by_code().keys())
    assert not report.is_valid()
    assert "UNKNOWN_CLASS_ID" in codes
    assert "BOX_EXCEEDS_BOUNDS" in codes
    assert "CENTER_OUT_OF_RANGE" in codes
    assert "SIZE_NON_POSITIVE" in codes
    assert "NON_FINITE_COORD" in codes
    assert "BAD_TOKEN_COUNT" in codes
    assert "INVALID_CLASS_FORMAT" in codes
    assert report.num_total_boxes == 0


def test_detects_corrupt_and_empty_images_and_missing_orphan_labels(tmp_path: Path) -> None:
    """Валидатор выявляет битые и нулевые изображения, отсутствующие и осиротевшие .txt."""
    ds = _setup_mini_dataset(tmp_path)
    (ds / "images" / "corrupt.jpg").write_bytes(b"not a real jpeg file")
    (ds / "images" / "empty.png").write_bytes(b"")
    _create_dummy_image(ds / "images" / "no_label.png")
    (ds / "labels" / "orphan_only.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    report = validate_dataset(ds)
    codes = set(report.counts_by_code().keys())
    assert not report.is_valid()
    assert "IMAGE_CORRUPT" in codes
    assert "IMAGE_EMPTY" in codes
    assert "MISSING_LABEL_FILE" in codes
    assert "ORPHAN_LABEL_FILE" in codes


def test_detects_duplicate_and_near_duplicate_boxes(tmp_path: Path) -> None:
    """Валидатор предупреждает о точных и сильно перекрывающихся (IoU >= 0.85) дубликатах боксов."""
    ds = _setup_mini_dataset(tmp_path)
    _create_dummy_image(ds / "images" / "dup.png")
    (ds / "labels" / "dup.txt").write_text(
        "1 0.500000 0.500000 0.200000 0.200000\n"
        "1 0.500000 0.500000 0.200000 0.200000\n"
        "1 0.502000 0.502000 0.200000 0.200000\n",
        encoding="utf-8",
    )

    report = validate_dataset(ds)
    codes = set(report.counts_by_code().keys())
    assert "EXACT_DUPLICATE_BOX" in codes
    assert "NEAR_DUPLICATE_BOX" in codes
    assert not report.is_valid(strict=True)


def test_cvat_class_schema_mismatch_detection() -> None:
    """Проверяет защиту от сдвига или пересортировки ID классов после экспорта из CVAT."""
    ref = ["bicycle", "car", "motorcycle", "bus"]
    reordered = sorted(ref)
    issues = verify_class_schema(reordered, ref)
    assert len(issues) == 1
    assert issues[0].code == "CLASS_MAPPING_MISMATCH"
    assert verify_class_schema(ref, ref) == []


def test_bbox_clip_to_unit_preserves_valid_bounds() -> None:
    """Клиппинг корректно пересчитывает (xc, yc, w, h) без выхода за [0, 1]."""
    raw = BBox(class_id=1, xc=0.95, yc=0.02, w=0.20, h=0.10)
    clipped, changed = raw.clip_to_unit()
    assert changed is True
    x1, y1, x2, y2 = clipped.to_xyxy()
    assert x1 >= 0.0 and y1 >= 0.0 and x2 <= 1.0 + 1e-7 and y2 <= 1.0 + 1e-7
    assert pytest.approx(x1, abs=1e-6) == 0.85
    assert pytest.approx(x2, abs=1e-6) == 1.0
    assert pytest.approx(y1, abs=1e-6) == 0.0
    assert pytest.approx(y2, abs=1e-6) == 0.07


def test_confusion_matrix_and_tp_fp_fn_matching() -> None:
    """Проверяет расчёт confusion matrix (TP, FP, FN и перепутывание классов)."""
    from src.tensorboard_logger import DetectionPred, compute_confusion_matrix, match_predictions_to_gt

    gt = [
        BBox(1, 0.2, 0.2, 0.1, 0.1),  # car (TP)
        BBox(7, 0.5, 0.5, 0.1, 0.1),  # police -> predicted as car (class swap)
        BBox(10, 0.8, 0.8, 0.05, 0.1),  # pedestrian (FN)
    ]
    preds = [
        DetectionPred(1, 0.90, BBox(1, 0.2, 0.2, 0.1, 0.1)),   # TP car
        DetectionPred(1, 0.80, BBox(1, 0.5, 0.5, 0.1, 0.1)),   # confused police -> car
        DetectionPred(4, 0.60, BBox(4, 0.9, 0.1, 0.1, 0.1)),   # FP truck on background
        DetectionPred(1, 0.10, BBox(1, 0.1, 0.9, 0.1, 0.1)),   # below conf_threshold=0.25
    ]

    tps, fps, fns = match_predictions_to_gt(gt, preds, iou_threshold=0.5, conf_threshold=0.25)
    assert len(tps) == 1
    assert len(fps) == 2
    assert len(fns) == 2

    cm = compute_confusion_matrix([gt], [preds], num_classes=11, iou_threshold=0.5, conf_threshold=0.25)
    assert cm.shape == (12, 12)
    assert cm[1, 1] == 1.0   # true car -> pred car
    assert cm[1, 7] == 1.0   # true police -> pred car
    assert cm[4, 11] == 1.0  # background -> pred truck (FP)
    assert cm[11, 10] == 1.0 # true pedestrian -> background (FN)
