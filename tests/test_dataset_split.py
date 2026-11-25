"""Юнит-тесты воспроизводимости, отсутствия утечки камер и целостности manifest (src.prepare_dataset)."""

from __future__ import annotations

import json
from pathlib import Path

from src.prepare_dataset import (
    PreparedSample,
    assign_scene_groups,
    prepare_dataset,
    stratified_group_split,
)
from src.utils import PROJECT_ROOT, load_yaml, verify_manifest
from src.validate_labels import BBox, validate_dataset


def _make_synthetic_samples() -> list[PreparedSample]:
    """Создаёт синтетический набор из 20 изображений с 3 мультикадровыми камерами и редким классом."""
    samples: list[PreparedSample] = []
    # Камера 1: 3 похожих кадра (dHash одинаковый)
    for i in range(3):
        samples.append(
            PreparedSample(
                filename=f"camA_{i}.png",
                stem=f"camA_{i}",
                source_name="synth",
                image_path=Path(f"camA_{i}.png"),
                width=1280,
                height=720,
                dhash=0b1111000011110000,
                exif_orientation=1,
                black_mask_ratio=0.0,
                boxes=[BBox(1, 0.5, 0.5, 0.1, 0.1)],
            )
        )
    # Камера 2: 2 похожих кадра (dHash diff = 1 бит)
    for i, dh in enumerate([0b0000111100001111, 0b0000111100001110]):
        samples.append(
            PreparedSample(
                filename=f"camB_{i}.png",
                stem=f"camB_{i}",
                source_name="synth",
                image_path=Path(f"camB_{i}.png"),
                width=1280,
                height=720,
                dhash=dh,
                exif_orientation=1,
                black_mask_ratio=0.0,
                boxes=[BBox(1, 0.4, 0.5, 0.1, 0.1), BBox(2, 0.7, 0.6, 0.1, 0.1)],
            )
        )
    # 15 уникальных сцен с хорошо разнесёнными dHash
    for i in range(15):
        dh = (i * 0x0F0F0F0F0F0F0F0F) ^ (0x55AA55AA << (i % 17))
        boxes = [BBox(1, 0.5, 0.5, 0.1, 0.1)]
        if i % 4 == 0:
            boxes.append(BBox(0, 0.2, 0.3, 0.1, 0.1))  # редкий класс 0 (4 сцены)
        if i % 5 == 0:
            boxes.append(BBox(3, 0.8, 0.3, 0.1, 0.1))  # редкий класс 3 (3 сцены)
        samples.append(
            PreparedSample(
                filename=f"single_{i:02d}.png",
                stem=f"single_{i:02d}",
                source_name="synth",
                image_path=Path(f"single_{i:02d}.png"),
                width=640,
                height=480,
                dhash=dh & ((1 << 64) - 1),
                exif_orientation=1,
                black_mask_ratio=0.0,
                boxes=boxes,
            )
        )
    return samples


def test_group_split_zero_leakage_and_disjoint() -> None:
    """Сплит гарантирует непересечение файлов и 0% утечки кадров одной камеры между сплитами."""
    samples = _make_synthetic_samples()
    assign_scene_groups(samples, dhash_threshold=4)
    splits = stratified_group_split(samples, num_classes=4, seed=42)

    train_f = {s.filename for s in splits["train"]}
    val_f = {s.filename for s in splits["val"]}
    test_f = {s.filename for s in splits["test"]}

    assert train_f.isdisjoint(val_f)
    assert train_f.isdisjoint(test_f)
    assert val_f.isdisjoint(test_f)
    assert len(train_f) + len(val_f) + len(test_f) == len(samples)

    train_g = {s.group_id for s in splits["train"]}
    val_g = {s.group_id for s in splits["val"]}
    test_g = {s.group_id for s in splits["test"]}
    assert train_g.isdisjoint(val_g)
    assert train_g.isdisjoint(test_g)
    assert val_g.isdisjoint(test_g)

    # Все 4 класса должны присутствовать в train
    train_classes = {b.class_id for s in splits["train"] for b in s.boxes}
    assert train_classes == {0, 1, 2, 3}


def test_group_split_is_deterministic() -> None:
    """Одинаковый seed гарантирует побайтово идентичный состав train/val/test."""
    s1 = _make_synthetic_samples()
    assign_scene_groups(s1, dhash_threshold=4)
    r1 = stratified_group_split(s1, num_classes=4, seed=123)

    s2 = _make_synthetic_samples()
    assign_scene_groups(s2, dhash_threshold=4)
    r2 = stratified_group_split(s2, num_classes=4, seed=123)

    for sp in ("train", "val", "test"):
        assert [x.filename for x in r1[sp]] == [x.filename for x in r2[sp]]


def test_prepare_dataset_reproducibility_and_manifest(tmp_path: Path) -> None:
    """Двойной прогон prepare_dataset на реальной мини-выборке даёт идентичные сплиты,
    валидный dataset.yaml без абсолютных путей и проверяемый manifest.sha256."""
    raw_dir = PROJECT_ROOT / "test_data" / "traffic_v5_mini"
    out_a = tmp_path / "run_a" / "processed"
    rep_a = tmp_path / "run_a" / "reports"
    out_b = tmp_path / "run_b" / "processed"
    rep_b = tmp_path / "run_b" / "reports"

    res_a = prepare_dataset(input_dir=raw_dir, output_dir=out_a, reports_dir=rep_a, seed=42)
    res_b = prepare_dataset(input_dir=raw_dir, output_dir=out_b, reports_dir=rep_b, seed=42)

    assert res_a["post_errors"] == 0
    assert res_a["post_warnings"] == 0
    assert res_a["num_images"] == 40
    assert res_a["split_counts"] == res_b["split_counts"]

    # Сплиты полностью совпадают между двумя запусками
    meta_a = json.loads((out_a / "split_metadata.json").read_text(encoding="utf-8"))
    meta_b = json.loads((out_b / "split_metadata.json").read_text(encoding="utf-8"))
    assert meta_a == meta_b

    # dataset.yaml содержит только относительные пути и все 11 классов
    ds_yaml = load_yaml(out_a / "dataset.yaml")
    assert "path" not in ds_yaml or not Path(str(ds_yaml["path"])).is_absolute()
    assert not Path(ds_yaml["train"]).is_absolute()
    assert not Path(ds_yaml["val"]).is_absolute()
    assert not Path(ds_yaml["test"]).is_absolute()
    assert ds_yaml["nc"] == 11

    # Повторная строгая валидация обработанного датасета
    post = validate_dataset(out_a, reference_classes_path=raw_dir / "classes.txt")
    assert post.is_valid(strict=True)

    # Проверка целостности manifest и реакция на повреждение файла
    manifest_a = rep_a / "manifest.sha256"
    ok, errs = verify_manifest(manifest_a, out_a.parent)
    assert ok, f"Ошибки manifest: {errs}"

    # Портим один файл и убеждаемся, что verify_manifest это ловит
    (out_a / "dataset.yaml").write_text("corrupted: true\n", encoding="utf-8")
    ok_after, errs_after = verify_manifest(manifest_a, out_a.parent)
    assert not ok_after
    assert any("dataset.yaml" in e for e in errs_after)
