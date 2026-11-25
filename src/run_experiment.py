"""CLI-раннер воспроизводимого эксперимента детекции объектов Ultralytics YOLO + TensorBoard.

Пример dry-run запуска:
    python -m src.run_experiment \
      --config configs/yolo.yaml \
      --data data/processed/dataset.yaml \
      --dry-run

Пример полноценного короткого обучения:
    python -m src.run_experiment \
      --config configs/yolo.yaml \
      --data data/processed/dataset.yaml
"""

from __future__ import annotations

import argparse
import copy
import logging
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.tensorboard_logger import (
    DetectionPred,
    ExperimentTensorBoardLogger,
    attach_ultralytics_callbacks,
    render_confusion_matrix_figure,
    render_gt_vs_error_analysis,
    render_pr_f1_curves,
)
from src.utils import (
    PROJECT_ROOT,
    get_best_device,
    get_git_metadata,
    get_runtime_metadata,
    list_image_files,
    load_yaml,
    normalize_stem,
    save_json,
    save_yaml,
    set_seed,
    setup_logging,
    sha256_file,
)
from src.validate_labels import BBox, parse_yolo_label_file

logger = logging.getLogger("sts_cv.experiment")


def _create_isolated_run_dir(runs_root: Path, exp_name: str, mode: str, short_sha: str) -> Path:
    """Создаёт уникальную изолированную папку запуска в runs/."""
    runs_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{ts}_{exp_name}_{mode}_{short_sha}"
    candidate = runs_root / base_name
    suffix = 1
    while candidate.exists():
        candidate = runs_root / f"{base_name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _resolve_dataset_splits(data_yaml_path: Path) -> tuple[list[str], dict[str, tuple[Path, Path]]]:
    """Проверяет dataset.yaml и возвращает классы и каталоги (images, labels) для train/val/test."""
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"Файл датасета не найден: {data_yaml_path}. "
            "Сначала выполните: python -m src.prepare_dataset --input test_data/traffic_v5_mini --output data/processed --seed 42"
        )
    cfg = load_yaml(data_yaml_path)
    names_raw = cfg.get("names")
    if isinstance(names_raw, dict):
        classes = [str(names_raw[k]) for k in sorted(names_raw.keys(), key=int)]
    elif isinstance(names_raw, list):
        classes = [str(x) for x in names_raw]
    else:
        raise ValueError(f"Поле 'names' отсутствует или некорректно в {data_yaml_path}")

    raw_path = cfg.get("path")
    if raw_path is None or str(raw_path).strip() in ("", "."):
        dataset_root = data_yaml_path.parent.resolve()
    else:
        p = Path(str(raw_path))
        dataset_root = p if p.is_absolute() else (data_yaml_path.parent / p).resolve()

    splits: dict[str, tuple[Path, Path]] = {}
    for sp in ("train", "val", "test"):
        rel_img = cfg.get(sp)
        if not rel_img:
            continue
        img_dir = (dataset_root / str(rel_img)).resolve()
        lbl_dir = (dataset_root / str(rel_img).replace("images", "labels", 1)).resolve()
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Не найден каталог изображений сплита '{sp}': {img_dir}")
        splits[sp] = (img_dir, lbl_dir)

    if "train" not in splits or "val" not in splits:
        raise ValueError(f"dataset.yaml должен содержать как минимум train и val: {data_yaml_path}")
    return classes, splits


def _load_split_gt(
    images_dir: Path,
    labels_dir: Path,
    num_classes: int,
) -> tuple[list[Path], list[list[BBox]]]:
    """Загружает пары (image_path, gt_boxes) для указанного сплита."""
    img_paths = list_image_files(images_dir)
    all_gt: list[list[BBox]] = []
    for img_p in img_paths:
        lbl_p = labels_dir / f"{normalize_stem(img_p.stem)}.txt"
        if not lbl_p.exists():
            # fallback на оригинальный stem без перекодировки
            lbl_p = labels_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            boxes, _, _ = parse_yolo_label_file(lbl_p, num_classes=num_classes, rel_name=lbl_p.name)
            all_gt.append(boxes)
        else:
            all_gt.append([])
    return img_paths, all_gt


def _extract_val_summary(val_results: Any, classes: list[str], gt_counts: Counter[int]) -> tuple[dict[str, float], list[dict[str, Any]], np.ndarray | None]:
    """Извлекает агрегированные и поклассовые метрики + confusion matrix из результата model.val()."""
    box = getattr(val_results, "box", None)
    precision = float(getattr(box, "mp", 0.0)) if box is not None else 0.0
    recall = float(getattr(box, "mr", 0.0)) if box is not None else 0.0
    map50 = float(getattr(box, "map50", 0.0)) if box is not None else 0.0
    map50_95 = float(getattr(box, "map", 0.0)) if box is not None else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    summary = {
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "mAP50": round(map50, 5),
        "mAP50_95": round(map50_95, 5),
        "F1": round(f1, 5),
    }

    ap_class_index = list(getattr(box, "ap_class_index", [])) if box is not None else []
    p_arr = getattr(box, "p", []) if box is not None else []
    r_arr = getattr(box, "r", []) if box is not None else []
    ap50_arr = getattr(box, "ap50", []) if box is not None else []
    ap_arr = getattr(box, "ap", []) if box is not None else []

    by_cid: dict[int, dict[str, float]] = {}
    for pos, cid in enumerate(ap_class_index):
        by_cid[int(cid)] = {
            "precision": float(p_arr[pos]) if pos < len(p_arr) else 0.0,
            "recall": float(r_arr[pos]) if pos < len(r_arr) else 0.0,
            "ap50": float(ap50_arr[pos]) if pos < len(ap50_arr) else 0.0,
            "ap50_95": float(ap_arr[pos]) if pos < len(ap_arr) else 0.0,
        }

    per_class_rows: list[dict[str, Any]] = []
    for cid, cname in enumerate(classes):
        m = by_cid.get(cid, {"precision": 0.0, "recall": 0.0, "ap50": 0.0, "ap50_95": 0.0})
        per_class_rows.append(
            {
                "class_id": cid,
                "class_name": cname,
                "gt_count": int(gt_counts.get(cid, 0)),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "ap50": round(m["ap50"], 4),
                "ap50_95": round(m["ap50_95"], 4),
            }
        )

    cm_obj = getattr(val_results, "confusion_matrix", None)
    cm_mat = getattr(cm_obj, "matrix", None) if cm_obj is not None else None
    return summary, per_class_rows, cm_mat


def _evaluate_and_log_split(
    model: Any,
    data_yaml_path: Path,
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    classes: list[str],
    imgsz: int,
    device: str,
    conf_thr: float,
    iou_thr: float,
    max_vis_samples: int,
    tb_logger: ExperimentTensorBoardLogger,
    artifacts_dir: Path,
    step: int,
    ultra_eval_dir: Path,
) -> dict[str, Any]:
    """Запускает model.val(), строит confusion matrix, PR/F1 кривые и Top-K Error Analysis (TP/FP/FN)."""
    img_paths, all_gt = _load_split_gt(images_dir, labels_dir, num_classes=len(classes))
    gt_counts = Counter(b.class_id for gts in all_gt for b in gts)

    val_res = model.val(
        data=str(data_yaml_path),
        split=split_name,
        imgsz=imgsz,
        device=device,
        batch=min(8, max(1, len(img_paths))),
        workers=0,
        project=str(ultra_eval_dir),
        name=f"eval_{split_name}",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    summary, per_class_rows, cm_mat = _extract_val_summary(val_res, classes, gt_counts)

    # Логируем метрики test в TensorBoard
    if split_name == "test":
        for k, v in summary.items():
            tb_logger.log_scalar(f"test/{k}", v, step)
    tb_logger.log_per_class_metrics(split_name, per_class_rows, step=step)

    # Матрицы ошибок -> PNG в artifacts/ + TensorBoard Images
    if cm_mat is not None:
        cm_norm_img = render_confusion_matrix_figure(
            cm_mat,
            classes,
            title=f"Confusion Matrix — {split_name} (нормализованная)",
            normalize=True,
        )
        cm_norm_img.save(artifacts_dir / f"confusion_matrix_{split_name}_normalized.png")
        tb_logger.log_image(f"eval/confusion_matrix_{split_name}_normalized", cm_norm_img, step=step)

        if split_name == "val":
            cm_raw_img = render_confusion_matrix_figure(
                cm_mat,
                classes,
                title="Confusion Matrix — val (абсолютные значения)",
                normalize=False,
            )
            cm_raw_img.save(artifacts_dir / "confusion_matrix_val.png")
            tb_logger.log_image("eval/confusion_matrix_val", cm_raw_img, step=step)

    # Инференс на всех изображениях сплита для PR/F1 и TP/FP/FN Error Analysis
    all_preds: list[list[DetectionPred]] = []
    for img_p in img_paths:
        pred_res = model.predict(
            source=str(img_p),
            imgsz=imgsz,
            conf=0.05,
            iou=0.60,
            device=device,
            verbose=False,
        )[0]
        preds_i: list[DetectionPred] = []
        if pred_res.boxes is not None and len(pred_res.boxes) > 0:
            xywhn = pred_res.boxes.xywhn.cpu().numpy()
            cids = pred_res.boxes.cls.cpu().numpy().astype(int)
            confs = pred_res.boxes.conf.cpu().numpy()
            for (xc, yc, bw, bh), cid, cf in zip(xywhn, cids, confs, strict=True):
                if 0 <= int(cid) < len(classes):
                    b, _ = BBox(int(cid), float(xc), float(yc), float(bw), float(bh)).clip_to_unit()
                    preds_i.append(DetectionPred(class_id=int(cid), conf=float(cf), box=b))
        all_preds.append(preds_i)

    pr_img, opt_thr = render_pr_f1_curves(all_gt, all_preds, iou_threshold=iou_thr, split_name=split_name)
    pr_img.save(artifacts_dir / f"pr_f1_curves_{split_name}.png")
    tb_logger.log_image(f"eval/pr_f1_curves_{split_name}", pr_img, step=step)

    # Сортируем кадры по числу ошибок (FP + FN) от худших к лучшим (Top-K Error Analysis)
    frame_errors: list[tuple[int, int, Path, list[BBox], list[DetectionPred]]] = []
    for img_p, gts, preds_i in zip(img_paths, all_gt, all_preds, strict=True):
        _, stats = render_gt_vs_error_analysis(
            img_p, gts, preds_i, classes, iou_threshold=iou_thr, conf_threshold=conf_thr
        )
        err_score = stats["fp"] + stats["fn"]
        frame_errors.append((err_score, len(gts), img_p, gts, preds_i))

    frame_errors.sort(key=lambda x: (-x[0], -x[1], x[2].name))
    error_records: list[dict[str, Any]] = []
    for idx, (_, _, img_p, gts, preds_i) in enumerate(frame_errors[:max_vis_samples], start=1):
        combo, stats = render_gt_vs_error_analysis(
            img_p, gts, preds_i, classes, iou_threshold=iou_thr, conf_threshold=conf_thr
        )
        combo.save(artifacts_dir / f"{split_name}_error_analysis_{idx:02d}.jpg", quality=88)
        tb_logger.log_image(f"predictions/{split_name}_error_analysis_{idx:02d}", combo, step=step)
        error_records.append({"rank": idx, "image": img_p.name, **stats})

    return {
        "metrics": summary,
        "optimal_threshold": opt_thr,
        "per_class": per_class_rows,
        "top_error_frames": error_records,
    }


def _benchmark_inference(
    model: Any,
    sample_images: list[Path],
    imgsz: int,
    device: str,
    iters: int,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    """Измеряет latency (mean/p50/p95 ms/img), FPS и размер checkpoint."""
    if not sample_images:
        return {}
    # Warmup
    for img_p in sample_images[: min(2, len(sample_images))]:
        model.predict(source=str(img_p), imgsz=imgsz, device=device, verbose=False)

    latencies_ms: list[float] = []
    for i in range(max(1, iters)):
        img_p = sample_images[i % len(sample_images)]
        t0 = time.perf_counter()
        model.predict(source=str(img_p), imgsz=imgsz, device=device, verbose=False)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    p50_ms = float(np.percentile(arr, 50))
    p95_ms = float(np.percentile(arr, 95))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    ckpt_mb = (
        round(checkpoint_path.stat().st_size / (1024.0 * 1024.0), 2)
        if checkpoint_path is not None and checkpoint_path.exists()
        else None
    )
    return {
        "device": device,
        "imgsz": imgsz,
        "iterations": len(latencies_ms),
        "latency_ms_mean": round(mean_ms, 2),
        "latency_ms_p50": round(p50_ms, 2),
        "latency_ms_p95": round(p95_ms, 2),
        "fps": round(fps, 2),
        "checkpoint_size_mb": ckpt_mb,
    }


def run_experiment(
    config_path: Path,
    data_yaml_path: Path,
    dry_run: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Выполняет воспроизводимый запуск эксперимента Ultralytics YOLO с логированием в TensorBoard."""
    cfg = copy.deepcopy(load_yaml(config_path))
    overrides = overrides or {}

    exp_cfg = cfg.setdefault("experiment", {})
    model_cfg = cfg.setdefault("model", {})
    train_cfg = cfg.setdefault("training", {})
    aug_cfg = cfg.setdefault("augmentations", {})
    eval_cfg = cfg.setdefault("evaluation", {})
    dry_cfg = cfg.setdefault("dry_run", {})

    # Применяем CLI-оверрайды
    if overrides.get("name"):
        exp_cfg["name"] = overrides["name"]
    if overrides.get("seed") is not None:
        exp_cfg["seed"] = int(overrides["seed"])
    if overrides.get("runs_dir"):
        exp_cfg["runs_dir"] = str(overrides["runs_dir"])
    if overrides.get("weights"):
        model_cfg["weights"] = overrides["weights"]
    if overrides.get("imgsz") is not None:
        model_cfg["imgsz"] = int(overrides["imgsz"])
    if overrides.get("device"):
        model_cfg["device"] = overrides["device"]
    if overrides.get("epochs") is not None:
        train_cfg["epochs"] = int(overrides["epochs"])
    if overrides.get("batch") is not None:
        train_cfg["batch"] = int(overrides["batch"])
    if overrides.get("lr0") is not None:
        train_cfg["lr0"] = float(overrides["lr0"])
    if overrides.get("skip_test"):
        eval_cfg["run_test_split"] = False
    if overrides.get("skip_benchmark"):
        eval_cfg["benchmark_inference"] = False

    if dry_run:
        train_cfg["epochs"] = int(dry_cfg.get("epochs", 1))
        model_cfg["imgsz"] = int(dry_cfg.get("imgsz", 320))
        train_cfg["batch"] = int(dry_cfg.get("batch", 4))
        eval_cfg["max_vis_samples"] = int(dry_cfg.get("max_vis_samples", 3))
        eval_cfg["benchmark_iters"] = int(dry_cfg.get("benchmark_iters", 5))

    seed = int(exp_cfg.get("seed", 42))
    deterministic = bool(exp_cfg.get("deterministic", True))
    set_seed(seed, deterministic=deterministic)

    device = get_best_device(str(model_cfg.get("device", "auto")))
    amp_cfg = train_cfg.get("amp", "auto")
    use_amp = device not in ("cpu", "mps") if str(amp_cfg).lower() == "auto" else bool(amp_cfg)
    train_cfg["amp"] = use_amp
    model_cfg["resolved_device"] = device

    # Проверка датасета и метаданные версий
    data_yaml_resolved = data_yaml_path.resolve()
    classes, splits = _resolve_dataset_splits(data_yaml_resolved)
    git_meta = get_git_metadata(PROJECT_ROOT)
    runtime_meta = get_runtime_metadata(device)

    dataset_yaml_sha = sha256_file(data_yaml_resolved)
    manifest_rel = Path(str(exp_cfg.get("manifest_path", "reports/manifest.sha256")))
    manifest_path = manifest_rel if manifest_rel.is_absolute() else (PROJECT_ROOT / manifest_rel)
    manifest_sha = sha256_file(manifest_path) if manifest_path.exists() else "missing"

    runs_rel = Path(str(exp_cfg.get("runs_dir", "runs")))
    runs_root = runs_rel if runs_rel.is_absolute() else (PROJECT_ROOT / runs_rel)
    mode_tag = "dryrun" if dry_run else "train"
    run_dir = _create_isolated_run_dir(
        runs_root=runs_root,
        exp_name=str(exp_cfg.get("name", "yolo_exp")),
        mode=mode_tag,
        short_sha=git_meta["short_commit"],
    )
    checkpoints_dir = run_dir / "checkpoints"
    artifacts_dir = run_dir / "artifacts"
    ultra_dir = run_dir / "ultralytics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=run_dir / "experiment.log")
    logger.info("Старт эксперимента: %s (mode=%s, device=%s, seed=%d)", run_dir.name, mode_tag, device, seed)
    logger.info("Датасет: %s (sha256=%s, classes=%d)", data_yaml_path, dataset_yaml_sha[:12], len(classes))

    # Сохраняем resolved YAML конфиг
    resolved_yaml_path = run_dir / "config_resolved.yaml"
    save_yaml(cfg, resolved_yaml_path)
    resolved_yaml_text = resolved_yaml_path.read_text(encoding="utf-8")

    # Инициализируем TensorBoard SummaryWriter
    tb_logger = ExperimentTensorBoardLogger(log_dir=run_dir)
    tb_logger.log_metadata(
        run_info={
            "run_id": run_dir.name,
            "mode": mode_tag,
            "weights": model_cfg.get("weights"),
            "imgsz": model_cfg.get("imgsz"),
            "epochs": train_cfg.get("epochs"),
            "batch": train_cfg.get("batch"),
            "seed": seed,
            "device": device,
            "gpu_hardware": runtime_meta["hardware"].get("gpu_name"),
            "git_commit": git_meta["commit"],
            "git_branch": git_meta["branch"],
            "git_dirty": git_meta["dirty"],
            "dataset_yaml_sha256": dataset_yaml_sha,
            "manifest_sha256": manifest_sha,
            "split_sizes": {sp: len(list_image_files(dirs[0])) for sp, dirs in splits.items()},
            "python": runtime_meta["versions"].get("python"),
            "torch": runtime_meta["versions"].get("torch"),
            "ultralytics": runtime_meta["versions"].get("ultralytics"),
        },
        resolved_config_yaml=resolved_yaml_text,
    )

    # Если уже есть EDA график распределения классов, логируем его в TensorBoard
    eda_chart = PROJECT_ROOT / "reports" / "samples" / "eda_class_distribution.png"
    if eda_chart.exists():
        with Image.open(eda_chart) as im_eda:
            tb_logger.log_image("dataset/class_distribution", im_eda.convert("RGB"), step=0)

    # Отключаем встроенный TB Ultralytics, чтобы в runs/<run_id> был единый чистый SummaryWriter
    from ultralytics import YOLO
    from ultralytics import settings as ultra_settings

    try:
        ultra_settings.update({"tensorboard": False})
    except (OSError, RuntimeError, KeyError, ValueError):
        logger.debug("Не удалось обновить настройки Ultralytics tensorboard=False.")

    # Храним загруженные базовые веса в кэше .cache/weights (вне git), чтобы не мусорить в корне проекта
    weights_spec = str(model_cfg.get("weights", "yolo11n.pt"))
    weights_path = Path(weights_spec)
    if not weights_path.exists() and not weights_path.is_absolute():
        weights_cache = PROJECT_ROOT / ".cache" / "weights"
        weights_cache.mkdir(parents=True, exist_ok=True)
        weights_path = weights_cache / weights_spec

    model = YOLO(str(weights_path))
    attach_ultralytics_callbacks(model, tb_logger)

    # Запуск обучения
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml_resolved),
        "epochs": int(train_cfg.get("epochs", 10)),
        "imgsz": int(model_cfg.get("imgsz", 640)),
        "batch": int(train_cfg.get("batch", 8)),
        "workers": int(train_cfg.get("workers", 0)),
        "device": device,
        "seed": seed,
        "deterministic": deterministic,
        "optimizer": str(train_cfg.get("optimizer", "AdamW")),
        "lr0": float(train_cfg.get("lr0", 0.002)),
        "lrf": float(train_cfg.get("lrf", 0.05)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.0005)),
        "warmup_epochs": float(train_cfg.get("warmup_epochs", 2.0)),
        "patience": int(train_cfg.get("patience", 20)),
        "cos_lr": bool(train_cfg.get("cos_lr", True)),
        "close_mosaic": int(train_cfg.get("close_mosaic", 5)),
        "amp": use_amp,
        "project": str(ultra_dir),
        "name": "train",
        "exist_ok": True,
        "save": True,
        "plots": False,
        "val": True,
        "verbose": False,
        # Аугментации
        "mosaic": float(aug_cfg.get("mosaic", 0.8)),
        "mixup": float(aug_cfg.get("mixup", 0.05)),
        "copy_paste": float(aug_cfg.get("copy_paste", 0.1)),
        "degrees": float(aug_cfg.get("degrees", 0.0)),
        "translate": float(aug_cfg.get("translate", 0.1)),
        "scale": float(aug_cfg.get("scale", 0.4)),
        "fliplr": float(aug_cfg.get("fliplr", 0.5)),
        "flipud": float(aug_cfg.get("flipud", 0.0)),
        "hsv_h": float(aug_cfg.get("hsv_h", 0.015)),
        "hsv_s": float(aug_cfg.get("hsv_s", 0.6)),
        "hsv_v": float(aug_cfg.get("hsv_v", 0.4)),
    }
    freeze_layers = int(train_cfg.get("freeze", 0))
    if freeze_layers > 0:
        train_kwargs["freeze"] = freeze_layers

    t_train_start = time.perf_counter()
    model.train(**train_kwargs)
    train_elapsed = time.perf_counter() - t_train_start
    logger.info("Обучение завершено за %.1f сек.", train_elapsed)

    # Копируем checkpoints (best.pt и last.pt) в run_dir/checkpoints/
    ultra_weights_dir = ultra_dir / "train" / "weights"
    best_ckpt: Path | None = None
    last_ckpt: Path | None = None
    for ckpt_name in ("best.pt", "last.pt"):
        src_ckpt = ultra_weights_dir / ckpt_name
        if src_ckpt.exists():
            dst_ckpt = checkpoints_dir / ckpt_name
            shutil.copy2(src_ckpt, dst_ckpt)
            if ckpt_name == "best.pt":
                best_ckpt = dst_ckpt
            else:
                last_ckpt = dst_ckpt

    eval_ckpt = best_ckpt or last_ckpt
    if eval_ckpt is None:
        # Гарантируем сохранение checkpoint даже при нестандартном завершении dry-run
        fallback_ckpt = checkpoints_dir / "last.pt"
        model.save(str(fallback_ckpt))
        eval_ckpt = fallback_ckpt
    logger.info("Чекпоинт для оценки: %s", eval_ckpt)

    eval_model = YOLO(str(eval_ckpt))
    final_step = int(train_cfg.get("epochs", 1))
    conf_thr = float(eval_cfg.get("conf_threshold", 0.25))
    iou_thr = float(eval_cfg.get("iou_threshold", 0.50))
    max_vis = int(eval_cfg.get("max_vis_samples", 6))
    imgsz = int(model_cfg.get("imgsz", 640))

    # Финальная оценка на val: confusion matrix, PR/F1, Error Analysis (TP/FP/FN)
    val_eval = _evaluate_and_log_split(
        model=eval_model,
        data_yaml_path=data_yaml_resolved,
        split_name="val",
        images_dir=splits["val"][0],
        labels_dir=splits["val"][1],
        classes=classes,
        imgsz=imgsz,
        device=device,
        conf_thr=conf_thr,
        iou_thr=iou_thr,
        max_vis_samples=max_vis,
        tb_logger=tb_logger,
        artifacts_dir=artifacts_dir,
        step=final_step,
        ultra_eval_dir=ultra_dir,
    )
    logger.info("Итоговые метрики VAL: %s", val_eval["metrics"])

    # Независимая оценка на test
    test_eval: dict[str, Any] | None = None
    if bool(eval_cfg.get("run_test_split", True)) and "test" in splits:
        test_eval = _evaluate_and_log_split(
            model=eval_model,
            data_yaml_path=data_yaml_resolved,
            split_name="test",
            images_dir=splits["test"][0],
            labels_dir=splits["test"][1],
            classes=classes,
            imgsz=imgsz,
            device=device,
            conf_thr=conf_thr,
            iou_thr=iou_thr,
            max_vis_samples=max_vis,
            tb_logger=tb_logger,
            artifacts_dir=artifacts_dir,
            step=final_step,
            ultra_eval_dir=ultra_dir,
        )
        logger.info("Итоговые метрики TEST: %s", test_eval["metrics"])

    # Бенчмарк скорости инференса (ms/img, FPS)
    bench_metrics: dict[str, Any] = {}
    if bool(eval_cfg.get("benchmark_inference", True)):
        val_imgs = list_image_files(splits["val"][0])
        bench_metrics = _benchmark_inference(
            model=eval_model,
            sample_images=val_imgs,
            imgsz=imgsz,
            device=device,
            iters=int(eval_cfg.get("benchmark_iters", 20)),
            checkpoint_path=eval_ckpt,
        )
        if bench_metrics:
            tb_logger.log_scalar("benchmark/latency_ms_mean", bench_metrics["latency_ms_mean"], final_step)
            tb_logger.log_scalar("benchmark/latency_ms_p95", bench_metrics["latency_ms_p95"], final_step)
            tb_logger.log_scalar("benchmark/fps", bench_metrics["fps"], final_step)
            bench_md = "\n".join(
                [
                    "| Метрика | Значение |",
                    "|---|---|",
                    *[f"| `{k}` | `{v}` |" for k, v in bench_metrics.items()],
                ]
            )
            tb_logger.log_text("eval/inference_benchmark", bench_md, step=final_step)
            logger.info(
                "Бенчмарк инференса (%s, imgsz=%d): mean=%.2f ms, p95=%.2f ms, FPS=%.1f",
                device,
                imgsz,
                bench_metrics["latency_ms_mean"],
                bench_metrics["latency_ms_p95"],
                bench_metrics["fps"],
            )

    # Логируем HParams для удобного сравнения экспериментов в TensorBoard
    hparam_metrics = {
        "mAP50": val_eval["metrics"]["mAP50"],
        "mAP50_95": val_eval["metrics"]["mAP50_95"],
        "precision": val_eval["metrics"]["precision"],
        "recall": val_eval["metrics"]["recall"],
        "F1": val_eval["metrics"]["F1"],
    }
    if test_eval is not None:
        hparam_metrics["test_mAP50"] = test_eval["metrics"]["mAP50"]
        hparam_metrics["test_mAP50_95"] = test_eval["metrics"]["mAP50_95"]

    tb_logger.log_hparams(
        hparams={
            "mode": mode_tag,
            "weights": str(model_cfg.get("weights")),
            "imgsz": imgsz,
            "epochs": final_step,
            "batch": int(train_cfg.get("batch", 8)),
            "optimizer": str(train_cfg.get("optimizer", "AdamW")),
            "lr0": float(train_cfg.get("lr0", 0.002)),
            "seed": seed,
        },
        final_metrics=hparam_metrics,
    )
    tb_logger.close()

    # Сохраняем полные структурированные метаданные запуска
    metadata = {
        "run_id": run_dir.name,
        "mode": mode_tag,
        "train_elapsed_sec": round(train_elapsed, 2),
        "git": git_meta,
        "runtime": runtime_meta,
        "dataset": {
            "data_yaml": str(data_yaml_path),
            "dataset_yaml_sha256": dataset_yaml_sha,
            "manifest_sha256": manifest_sha,
            "num_classes": len(classes),
            "classes": classes,
            "split_sizes": {sp: len(list_image_files(dirs[0])) for sp, dirs in splits.items()},
        },
        "checkpoints": {
            "best": str(best_ckpt.relative_to(run_dir)) if best_ckpt else None,
            "last": str(last_ckpt.relative_to(run_dir)) if last_ckpt else None,
        },
        "val_evaluation": val_eval,
        "test_evaluation": test_eval,
        "inference_benchmark": bench_metrics,
    }
    save_json(metadata, run_dir / "run_metadata.json")
    save_json(tb_logger.epoch_history, run_dir / "metrics_history.json")

    logger.info("Эксперимент завершён. Логи TensorBoard: %s", run_dir)
    logger.info("Команда просмотра: tensorboard --logdir %s", runs_rel.as_posix())
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Воспроизводимый запуск детекции Ultralytics YOLO с логированием в TensorBoard."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/yolo.yaml"), help="YAML-конфигурация эксперимента.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/dataset.yaml"), help="Путь к dataset.yaml.")
    parser.add_argument("--dry-run", action="store_true", help="Быстрый дымовой прогон (1 эпоха, 320px) для проверки всего пайплайна.")
    parser.add_argument("--name", type=str, default=None, help="Переопределить имя эксперимента.")
    parser.add_argument("--weights", type=str, default=None, help="Переопределить базовые веса YOLO (например, yolo11s.pt).")
    parser.add_argument("--epochs", type=int, default=None, help="Переопределить число эпох.")
    parser.add_argument("--imgsz", type=int, default=None, help="Переопределить разрешение входа.")
    parser.add_argument("--batch", type=int, default=None, help="Переопределить размер батча.")
    parser.add_argument("--lr0", type=float, default=None, help="Переопределить начальный learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Устройство: auto | mps | cpu | 0.")
    parser.add_argument("--seed", type=int, default=None, help="Переопределить random seed.")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Каталог сохранения запусков (по умолчанию runs).")
    parser.add_argument("--skip-test", action="store_true", help="Пропустить оценку на test-сплите.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Пропустить замер FPS/latency.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    run_experiment(
        config_path=args.config,
        data_yaml_path=args.data,
        dry_run=args.dry_run,
        overrides={
            "name": args.name,
            "weights": args.weights,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "lr0": args.lr0,
            "device": args.device,
            "seed": args.seed,
            "runs_dir": args.runs_dir,
            "skip_test": args.skip_test,
            "skip_benchmark": args.skip_benchmark,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
