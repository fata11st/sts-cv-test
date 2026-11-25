"""Модуль логирования CV-эксперимента в TensorBoard через SummaryWriter.

Обеспечивает:
- единую документированную схему тегов TensorBoard (Scalars, Images, Text, HParams);
- привязку к коллбэкам Ultralytics YOLO на каждой эпохе;
- рендеринг confusion matrix, PR/F1 кривых и визуального Error Analysis (TP / FP / FN)
  с сортировкой худших кадров (Top-K error frames).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.validate_labels import BBox

logger = logging.getLogger("sts_cv.tb")

# Официальная спецификация состава логов TensorBoard (документируется в README.md)
TENSORBOARD_LOG_SCHEMA: dict[str, list[tuple[str, str]]] = {
    "Scalars": [
        ("loss/train_box, loss/train_cls, loss/train_dfl, loss/train_total", "Функции потерь на обучении по эпохам"),
        ("loss/val_box, loss/val_cls, loss/val_dfl, loss/val_total", "Функции потерь на валидации по эпохам"),
        ("metrics/precision, metrics/recall, metrics/mAP50, metrics/mAP50_95, metrics/F1", "Основные метрики качества детекции на val по эпохам"),
        ("metrics_per_class/AP50_<class>", "Поклассовый AP@0.5 на финальной валидации"),
        ("test/precision, test/recall, test/mAP50, test/mAP50_95, test/F1", "Итоговая оценка лучшего checkpoint на независимом test-сплите"),
        ("lr/pg0, lr/pg1, lr/pg2", "Расписание learning rate по группам параметров оптимизатора"),
        ("system/epoch_time_sec, system/device_mem_mb", "Длительность эпохи и использование памяти ускорителя"),
        ("benchmark/latency_ms_mean, benchmark/latency_ms_p95, benchmark/fps", "Производительность инференса модели (latency и FPS)"),
    ],
    "Images": [
        ("dataset/class_distribution", "График распределения классов датасета"),
        ("eval/confusion_matrix_val, eval/confusion_matrix_val_normalized", "Матрица ошибок на val (абсолютная и нормализованная)"),
        ("eval/confusion_matrix_test_normalized", "Нормализованная матрица ошибок на test"),
        ("eval/pr_f1_curves_val, eval/pr_f1_curves_test", "Precision-Recall и F1-Confidence кривые (подбор рабочего порога conf)"),
        ("predictions/val_error_analysis_<i>", "Коллажи GT (слева) vs Error Analysis TP/FP/FN (справа) для худших кадров val"),
        ("predictions/test_error_analysis_<i>", "Коллажи GT (слева) vs Error Analysis TP/FP/FN (справа) для худших кадров test"),
    ],
    "Text": [
        ("meta/experiment_info", "Таблица версии кода (Git), хэшей датасета, железа и окружения"),
        ("meta/config_resolved", "Полная YAML-конфигурация запуска"),
        ("eval/per_class_metrics_val, eval/per_class_metrics_test", "Подробная Markdown-таблица метрик по всем 11 классам"),
        ("eval/inference_benchmark", "Сводка скорости инференса (ms/img, FPS, размер весов)"),
    ],
    "HParams": [
        ("hparams -> hparam/mAP50, hparam/mAP50_95, hparam/precision, hparam/recall, hparam/test_mAP50", "Сравнение гиперпараметров и итоговых метрик между запусками"),
    ],
}


@dataclass
class DetectionPred:
    """Предсказанный детектором bounding box."""

    class_id: int
    conf: float
    box: BBox


def _load_unicode_font(size: int = 12) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    if font_path.exists():
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError:
            logger.debug("Не удалось загрузить шрифт %s.", font_path)
    return ImageFont.load_default()


def _pil_to_chw(img: Image.Image) -> np.ndarray:
    """Преобразует PIL.Image (RGB) в uint8 массив (C, H, W) для SummaryWriter.add_image."""
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return np.transpose(arr, (2, 0, 1))


def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Конвертирует matplotlib Figure в PIL.Image."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(rgba[:, :, :3].copy(), mode="RGB")
    plt.close(fig)
    return img


def match_predictions_to_gt(
    gt_boxes: list[BBox],
    preds: list[DetectionPred],
    iou_threshold: float = 0.50,
    conf_threshold: float = 0.25,
) -> tuple[list[tuple[DetectionPred, BBox, float]], list[DetectionPred], list[BBox]]:
    """Жадное сопоставление предсказаний с GT по убыванию confidence -> (TP, FP, FN)."""
    filtered = sorted(
        [p for p in preds if p.conf >= conf_threshold],
        key=lambda p: -p.conf,
    )
    matched_gt: set[int] = set()
    tps: list[tuple[DetectionPred, BBox, float]] = []
    fps: list[DetectionPred] = []

    for pred in filtered:
        best_iou = 0.0
        best_gt_idx = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt or gt.class_id != pred.class_id:
                continue
            iou_val = pred.box.iou(gt)
            if iou_val >= iou_threshold and iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = gi
        if best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            tps.append((pred, gt_boxes[best_gt_idx], best_iou))
        else:
            fps.append(pred)

    fns = [gt for gi, gt in enumerate(gt_boxes) if gi not in matched_gt]
    return tps, fps, fns


def compute_confusion_matrix(
    all_gt: list[list[BBox]],
    all_preds: list[list[DetectionPred]],
    num_classes: int,
    iou_threshold: float = 0.50,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """Строит матрицу ошибок (num_classes + 1) x (num_classes + 1) с классом background.

    Строки — предсказанный класс (0..N-1, N=background для FN),
    столбцы — истинный класс (0..N-1, N=background для FP).
    """
    bg = num_classes
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.float64)

    for gts, preds in zip(all_gt, all_preds, strict=True):
        filtered = sorted(
            [p for p in preds if p.conf >= conf_threshold],
            key=lambda p: -p.conf,
        )
        matched_gt: set[int] = set()
        for pred in filtered:
            best_iou = 0.0
            best_gi = -1
            for gi, gt in enumerate(gts):
                if gi in matched_gt:
                    continue
                iou_val = pred.box.iou(gt)
                if iou_val >= iou_threshold and iou_val > best_iou:
                    best_iou = iou_val
                    best_gi = gi
            if best_gi >= 0:
                matched_gt.add(best_gi)
                true_cls = gts[best_gi].class_id
                if 0 <= pred.class_id < num_classes and 0 <= true_cls < num_classes:
                    matrix[pred.class_id, true_cls] += 1.0
            elif 0 <= pred.class_id < num_classes:
                matrix[pred.class_id, bg] += 1.0

        for gi, gt in enumerate(gts):
            if gi not in matched_gt and 0 <= gt.class_id < num_classes:
                matrix[bg, gt.class_id] += 1.0

    return matrix


def render_confusion_matrix_figure(
    matrix: np.ndarray,
    classes: list[str],
    title: str,
    normalize: bool = False,
) -> Image.Image:
    """Отрисовывает читаемую матрицу ошибок (N+1 x N+1 с классом background)."""
    labels = [*classes, "background"]
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (len(labels), len(labels)):
        resized = np.zeros((len(labels), len(labels)), dtype=np.float64)
        r = min(mat.shape[0], len(labels))
        c = min(mat.shape[1], len(labels))
        resized[:r, :c] = mat[:r, :c]
        mat = resized

    if normalize:
        col_sums = mat.sum(axis=0, keepdims=True)
        disp = np.divide(mat, col_sums, out=np.zeros_like(mat), where=col_sums > 0)
    else:
        disp = mat

    fig, ax = plt.subplots(figsize=(9.5, 8.2), dpi=130)
    im = ax.imshow(disp, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=(1.0 if normalize else max(1.0, float(disp.max()))))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Истинный класс (True)", fontsize=10)
    ax.set_ylabel("Предсказанный класс (Predicted)", fontsize=10)
    ax.set_title(title, fontsize=11.5, pad=12)

    thresh = disp.max() * 0.55 if disp.max() > 0 else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = disp[i, j]
            if val <= 0:
                continue
            txt = f"{val:.2f}" if normalize else f"{round(val)}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=7.5,
                color=("white" if val > thresh else "black"),
            )

    fig.tight_layout()
    return _fig_to_pil(fig)


def render_pr_f1_curves(
    all_gt: list[list[BBox]],
    all_preds: list[list[DetectionPred]],
    iou_threshold: float = 0.50,
    split_name: str = "val",
) -> tuple[Image.Image, dict[str, float]]:
    """Строит микро-усреднённые Precision-Recall и F1-Confidence кривые и находит лучший порог conf."""
    confs = np.linspace(0.05, 0.90, 35)
    precs: list[float] = []
    recs: list[float] = []
    f1s: list[float] = []

    for thr in confs:
        tp_sum = fp_sum = fn_sum = 0
        for gts, preds in zip(all_gt, all_preds, strict=True):
            tps, fps, fns = match_predictions_to_gt(gts, preds, iou_threshold=iou_threshold, conf_threshold=float(thr))
            tp_sum += len(tps)
            fp_sum += len(fps)
            fn_sum += len(fns)
        p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precs.append(p)
        recs.append(r)
        f1s.append(f1)

    best_idx = int(np.argmax(f1s)) if f1s else 0
    best_conf = float(confs[best_idx])
    best_f1 = float(f1s[best_idx])

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=130)
    axes[0].plot(recs, precs, marker="o", markersize=3.5, linewidth=2.0, color="#2980b9")
    axes[0].set_xlim(0.0, 1.02)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title(f"Precision-Recall кривая ({split_name}, IoU={iou_threshold:.2f})")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(confs, f1s, label="F1", linewidth=2.2, color="#27ae60")
    axes[1].plot(confs, precs, label="Precision", linestyle="--", linewidth=1.4, color="#2980b9")
    axes[1].plot(confs, recs, label="Recall", linestyle="--", linewidth=1.4, color="#e67e22")
    axes[1].axvline(best_conf, color="#c0392b", linestyle=":", linewidth=1.8, label=f"Best conf={best_conf:.2f} (F1={best_f1:.2f})")
    axes[1].set_xlim(0.05, 0.90)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xlabel("Confidence threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"Подбор порога уверенности ({split_name})")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return _fig_to_pil(fig), {"best_conf": best_conf, "best_f1": best_f1}


def render_gt_vs_error_analysis(
    image_path: Path,
    gt_boxes: list[BBox],
    preds: list[DetectionPred],
    classes: list[str],
    iou_threshold: float = 0.50,
    conf_threshold: float = 0.25,
) -> tuple[Image.Image, dict[str, int]]:
    """Рендерит коллаж: слева Ground Truth, справа разбор ошибок TP (зелёный) / FP (красный) / FN (жёлтый)."""
    tps, fps, fns = match_predictions_to_gt(
        gt_boxes, preds, iou_threshold=iou_threshold, conf_threshold=conf_threshold
    )

    with Image.open(image_path) as raw_im:
        base = ImageOps.exif_transpose(raw_im).convert("RGB")
    target_w = 680
    target_h = max(1, round(base.height * target_w / base.width))
    base_r = base.resize((target_w, target_h), Image.Resampling.LANCZOS)

    header_h = 34
    h_font = _load_unicode_font(12)
    l_font = _load_unicode_font(10)

    left = Image.new("RGB", (target_w, target_h + header_h), (24, 28, 36))
    left.paste(base_r, (0, header_h))
    dl = ImageDraw.Draw(left)
    dl.text((10, 9), f"GT: {image_path.name} ({len(gt_boxes)} obj)", fill=(240, 244, 248), font=h_font)
    for gt in gt_boxes:
        x1, y1, x2, y2 = gt.to_xyxy()
        rx1, ry1, rx2, ry2 = x1 * target_w, y1 * target_h + header_h, x2 * target_w, y2 * target_h + header_h
        dl.rectangle([rx1, ry1, rx2, ry2], outline=(46, 204, 113), width=2)
        tag = classes[gt.class_id]
        dl.text((max(2, rx1 + 2), max(header_h, ry1 - 12)), tag, fill=(46, 204, 113), font=l_font)

    right = Image.new("RGB", (target_w, target_h + header_h), (24, 28, 36))
    right.paste(base_r, (0, header_h))
    dr = ImageDraw.Draw(right)
    dr.text(
        (10, 9),
        f"PRED Error Analysis (conf>={conf_threshold:.2f}): TP={len(tps)} (grn) | FP={len(fps)} (red) | FN={len(fns)} (ylw)",
        fill=(240, 244, 248),
        font=h_font,
    )

    # 1. FN (пропущенные объекты) — жёлтый
    for gt in fns:
        x1, y1, x2, y2 = gt.to_xyxy()
        rx1, ry1, rx2, ry2 = x1 * target_w, y1 * target_h + header_h, x2 * target_w, y2 * target_h + header_h
        dr.rectangle([rx1, ry1, rx2, ry2], outline=(241, 196, 15), width=3)
        dr.text((max(2, rx1 + 2), max(header_h, ry1 - 12)), f"FN:{classes[gt.class_id]}", fill=(241, 196, 15), font=l_font)

    # 2. TP (точные попадания) — зелёный
    for pred, _, _ in tps:
        x1, y1, x2, y2 = pred.box.to_xyxy()
        rx1, ry1, rx2, ry2 = x1 * target_w, y1 * target_h + header_h, x2 * target_w, y2 * target_h + header_h
        dr.rectangle([rx1, ry1, rx2, ry2], outline=(46, 204, 113), width=2)
        dr.text(
            (max(2, rx1 + 2), max(header_h, ry1 - 12)),
            f"TP:{classes[pred.class_id]} {pred.conf:.2f}",
            fill=(46, 204, 113),
            font=l_font,
        )

    # 3. FP (ложные срабатывания) — красный
    for pred in fps:
        x1, y1, x2, y2 = pred.box.to_xyxy()
        rx1, ry1, rx2, ry2 = x1 * target_w, y1 * target_h + header_h, x2 * target_w, y2 * target_h + header_h
        dr.rectangle([rx1, ry1, rx2, ry2], outline=(231, 76, 60), width=3)
        dr.text(
            (max(2, rx1 + 2), max(header_h, ry1 - 12)),
            f"FP:{classes[pred.class_id]} {pred.conf:.2f}",
            fill=(231, 76, 60),
            font=l_font,
        )

    combo = Image.new("RGB", (target_w * 2 + 10, target_h + header_h), (14, 16, 20))
    combo.paste(left, (0, 0))
    combo.paste(right, (target_w + 10, 0))
    return combo, {"tp": len(tps), "fp": len(fps), "fn": len(fns)}


class ExperimentTensorBoardLogger:
    """Обёртка над torch.utils.tensorboard.SummaryWriter для CV-экспериментов YOLO."""

    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer: SummaryWriter = SummaryWriter(log_dir=str(self.log_dir))
        self.epoch_history: list[dict[str, Any]] = []
        self._epoch_start_ts: float = time.perf_counter()

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        self.writer.add_text(tag, text, global_step=step)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if np.isfinite(value):
            self.writer.add_scalar(tag, float(value), global_step=step)

    def log_image(self, tag: str, image: Image.Image, step: int = 0) -> None:
        self.writer.add_image(tag, _pil_to_chw(image), global_step=step, dataformats="CHW")

    def log_metadata(
        self,
        run_info: dict[str, Any],
        resolved_config_yaml: str,
    ) -> None:
        """Логирует параметры запуска, версию Git и хэши датасета во вкладку Text."""
        rows = ["| Параметр | Значение |", "|---|---|"]
        for k, v in run_info.items():
            rows.append(f"| `{k}` | `{v}` |")
        self.log_text("meta/experiment_info", "\n".join(rows), step=0)
        self.log_text("meta/config_resolved", f"```yaml\n{resolved_config_yaml}\n```", step=0)

    def log_epoch_metrics(
        self,
        epoch: int,
        train_losses: dict[str, float],
        val_losses: dict[str, float],
        metrics: dict[str, float],
        lrs: dict[str, float],
        epoch_time_sec: float,
        device_mem_mb: float | None = None,
    ) -> None:
        """Записывает полный набор скаляров одной эпохи."""
        tr_total = sum(v for v in train_losses.values() if np.isfinite(v))
        for name, val in train_losses.items():
            self.log_scalar(f"loss/train_{name}", val, epoch)
        if train_losses:
            self.log_scalar("loss/train_total", tr_total, epoch)

        va_total = sum(v for v in val_losses.values() if np.isfinite(v))
        for name, val in val_losses.items():
            self.log_scalar(f"loss/val_{name}", val, epoch)
        if val_losses:
            self.log_scalar("loss/val_total", va_total, epoch)

        prec = metrics.get("precision", 0.0)
        rec = metrics.get("recall", 0.0)
        map50 = metrics.get("mAP50", 0.0)
        map50_95 = metrics.get("mAP50_95", 0.0)
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        self.log_scalar("metrics/precision", prec, epoch)
        self.log_scalar("metrics/recall", rec, epoch)
        self.log_scalar("metrics/mAP50", map50, epoch)
        self.log_scalar("metrics/mAP50_95", map50_95, epoch)
        self.log_scalar("metrics/F1", f1, epoch)

        for lr_k, lr_v in lrs.items():
            self.log_scalar(f"lr/{lr_k}", lr_v, epoch)

        self.log_scalar("system/epoch_time_sec", epoch_time_sec, epoch)
        if device_mem_mb is not None:
            self.log_scalar("system/device_mem_mb", device_mem_mb, epoch)

        self.epoch_history.append(
            {
                "epoch": epoch,
                "train_losses": {k: round(float(v), 5) for k, v in train_losses.items()},
                "train_loss_total": round(float(tr_total), 5),
                "val_losses": {k: round(float(v), 5) for k, v in val_losses.items()},
                "val_loss_total": round(float(va_total), 5),
                "precision": round(float(prec), 5),
                "recall": round(float(rec), 5),
                "mAP50": round(float(map50), 5),
                "mAP50_95": round(float(map50_95), 5),
                "F1": round(float(f1), 5),
                "lrs": {k: float(v) for k, v in lrs.items()},
                "epoch_time_sec": round(float(epoch_time_sec), 3),
            }
        )
        self.writer.flush()

    def log_per_class_metrics(
        self,
        split_name: str,
        per_class_rows: list[dict[str, Any]],
        step: int,
    ) -> None:
        """Логирует поклассовые AP@0.5 в Scalars и Markdown-таблицу в Text."""
        md = [
            f"### Поклассовые метрики (`{split_name}`)",
            "",
            "| ID | Класс | GT боксов | Precision | Recall | AP@0.5 | AP@0.5:0.95 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for row in per_class_rows:
            cname = row["class_name"]
            if split_name == "val":
                self.log_scalar(f"metrics_per_class/AP50_{cname}", row["ap50"], step)
            md.append(
                f"| {row['class_id']} | `{cname}` | {row['gt_count']} | "
                f"{row['precision']:.3f} | {row['recall']:.3f} | "
                f"{row['ap50']:.3f} | {row['ap50_95']:.3f} |"
            )
        self.log_text(f"eval/per_class_metrics_{split_name}", "\n".join(md), step=step)

    def log_hparams(self, hparams: dict[str, Any], final_metrics: dict[str, float]) -> None:
        """Логирует HParams в TensorBoard для сравнения запусков."""
        clean_hp: dict[str, int | float | str | bool] = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                clean_hp[k] = v
            else:
                clean_hp[k] = str(v)
        clean_m = {f"hparam/{k}": float(v) for k, v in final_metrics.items() if np.isfinite(v)}
        if clean_hp and clean_m:
            self.writer.add_hparams(clean_hp, clean_m, run_name=".")

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


def _extract_device_mem_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        if (
            hasattr(torch, "mps")
            and hasattr(torch.mps, "current_allocated_memory")
            and torch.backends.mps.is_available()
        ):
            return float(torch.mps.current_allocated_memory() / (1024.0 * 1024.0))
    except (ImportError, RuntimeError, AttributeError):
        return None
    return None


def attach_ultralytics_callbacks(model: Any, tb_logger: ExperimentTensorBoardLogger) -> None:
    """Подключает коллбэки Ultralytics Trainer -> ExperimentTensorBoardLogger."""

    seen_epochs: set[int] = set()

    def _on_train_epoch_start(trainer: Any) -> None:
        tb_logger._epoch_start_ts = time.perf_counter()

    def _on_fit_epoch_end(trainer: Any) -> None:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        max_epochs = int(getattr(trainer, "epochs", epoch))
        raw_metrics: dict[str, Any] = getattr(trainer, "metrics", {}) or {}
        val_losses: dict[str, float] = {}
        for k, v in raw_metrics.items():
            if k.startswith("val/"):
                short = k.replace("val/", "").replace("_loss", "")
                val_losses[short] = float(v)

        # Пропускаем дополнительный пост-тренировочный вызов final_eval (без val_losses и с epoch > epochs)
        if epoch in seen_epochs or (not val_losses and epoch > max_epochs):
            return
        seen_epochs.add(epoch)

        elapsed = time.perf_counter() - tb_logger._epoch_start_ts

        # Train losses
        train_losses: dict[str, float] = {}
        label_losses = getattr(trainer, "label_loss_items", None)
        tloss = getattr(trainer, "tloss", None)
        if callable(label_losses) and tloss is not None:
            try:
                items = label_losses(tloss, prefix="train")
                for k, v in items.items():
                    short = k.replace("train/", "").replace("_loss", "")
                    train_losses[short] = float(v)
            except (RuntimeError, TypeError, ValueError, AttributeError):
                logger.debug("Не удалось разобрать train loss на эпохе %d.", epoch)

        det_metrics = {
            "precision": float(raw_metrics.get("metrics/precision(B)", 0.0)),
            "recall": float(raw_metrics.get("metrics/recall(B)", 0.0)),
            "mAP50": float(raw_metrics.get("metrics/mAP50(B)", 0.0)),
            "mAP50_95": float(raw_metrics.get("metrics/mAP50-95(B)", 0.0)),
        }

        lrs: dict[str, float] = {}
        raw_lr = getattr(trainer, "lr", {}) or {}
        for k, v in raw_lr.items():
            short_lr = k.replace("lr/", "")
            lrs[short_lr] = float(v)

        tb_logger.log_epoch_metrics(
            epoch=epoch,
            train_losses=train_losses,
            val_losses=val_losses,
            metrics=det_metrics,
            lrs=lrs,
            epoch_time_sec=elapsed,
            device_mem_mb=_extract_device_mem_mb(),
        )

    model.add_callback("on_train_epoch_start", _on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
