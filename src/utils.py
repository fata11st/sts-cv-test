"""Общие утилиты проекта: воспроизводимость, хэширование, Git-метаданные, I/O."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Настраивает единый формат логирования проекта."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("sts_cv")


def normalize_stem(name: str) -> str:
    """Нормализует Unicode-имя файла к NFC (защита от расхождений NFD/NFC на macOS и Linux)."""
    return unicodedata.normalize("NFC", name)


def list_image_files(images_dir: Path) -> list[Path]:
    """Возвращает детерминированно отсортированный список изображений без скрытых файлов."""
    if not images_dir.exists():
        raise FileNotFoundError(f"Каталог изображений не найден: {images_dir}")
    files = [
        p
        for p in images_dir.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: normalize_stem(p.name))


def list_label_files(labels_dir: Path) -> list[Path]:
    """Возвращает детерминированно отсортированный список файлов YOLO-разметки (.txt)."""
    if not labels_dir.exists():
        return []
    files = [
        p
        for p in labels_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() == ".txt"
    ]
    return sorted(files, key=lambda p: normalize_stem(p.name))


def load_classes(classes_path: Path) -> list[str]:
    """Загружает и проверяет список классов из classes.txt (0-based индексация)."""
    if not classes_path.exists():
        raise FileNotFoundError(f"Файл классов не найден: {classes_path}")
    classes = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not classes:
        raise ValueError(f"Файл классов пуст: {classes_path}")
    if len(set(classes)) != len(classes):
        raise ValueError(f"Обнаружены дублирующиеся имена классов в {classes_path}: {classes}")
    return classes


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Фиксирует все генераторы случайных чисел для воспроизводимости."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = not deterministic
    except ImportError:
        pass


def get_best_device(requested: str = "auto") -> str:
    """Определяет доступное вычислительное устройство (CUDA > MPS > CPU)."""
    req = str(requested).strip().lower()
    if req != "auto":
        return str(requested)
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def sha256_bytes(data: bytes) -> str:
    """Вычисляет SHA-256 от байтовой последовательности."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Потоково вычисляет SHA-256 файла."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def to_posix_relpath(path: Path, base_dir: Path) -> str:
    """Возвращает POSIX-путь относительно base_dir без абсолютных путей."""
    rel = path.resolve().relative_to(base_dir.resolve())
    return rel.as_posix()


def write_manifest(
    paths: Iterable[Path],
    base_dir: Path,
    manifest_path: Path,
) -> dict[str, str]:
    """Формирует воспроизводимый SHA-256 manifest в стандартном формате sha256sum."""
    entries: dict[str, str] = {}
    for p in paths:
        if p.is_file():
            rel = normalize_stem(to_posix_relpath(p, base_dir))
            entries[rel] = sha256_file(p)

    sorted_items = sorted(entries.items(), key=lambda kv: kv[0])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{digest}  {rel_path}" for rel_path, digest in sorted_items]
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return dict(sorted_items)


def verify_manifest(manifest_path: Path, base_dir: Path) -> tuple[bool, list[str]]:
    """Проверяет целостность файлов по SHA-256 manifest."""
    errors: list[str] = []
    if not manifest_path.exists():
        return False, [f"Manifest не найден: {manifest_path}"]

    lines = [
        ln.strip()
        for ln in manifest_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if not lines:
        return False, ["Manifest пуст"]

    for idx, line in enumerate(lines, start=1):
        parts = line.split("  ", maxsplit=1)
        if len(parts) != 2:
            errors.append(f"Строка {idx}: некорректный формат manifest: {line!r}")
            continue
        expected_hash, rel_path = parts
        if Path(rel_path).is_absolute():
            errors.append(f"Строка {idx}: запрещённый абсолютный путь: {rel_path}")
            continue
        target = base_dir / rel_path
        if not target.exists():
            errors.append(f"Отсутствует файл из manifest: {rel_path}")
            continue
        actual_hash = sha256_file(target)
        if actual_hash != expected_hash:
            errors.append(
                f"Несовпадение SHA-256 для {rel_path}: ожидалось {expected_hash[:12]}, "
                f"получено {actual_hash[:12]}"
            )

    return len(errors) == 0, errors


def get_git_metadata(repo_dir: Path | None = None) -> dict[str, Any]:
    """Собирает метаданные версии кода из Git."""
    cwd = str((repo_dir or PROJECT_ROOT).resolve())

    def _run(args: list[str]) -> str | None:
        try:
            res = subprocess.run(
                ["git", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if res.returncode == 0:
                return res.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return None

    commit = _run(["rev-parse", "HEAD"]) or "uncommitted"
    short_commit = _run(["rev-parse", "--short=8", "HEAD"]) or "uncommit"
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    status_out = _run(["status", "--porcelain"])
    is_dirty = bool(status_out) if status_out is not None else None

    return {
        "commit": commit,
        "short_commit": short_commit,
        "branch": branch,
        "dirty": is_dirty,
    }


def get_runtime_metadata(device: str = "auto") -> dict[str, Any]:
    """Собирает сведения об окружении Python, библиотеках и аппаратном ускорении."""
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
    }
    for pkg_name, import_name in [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
        ("tensorboard", "tensorboard"),
        ("opencv", "cv2"),
        ("pillow", "PIL"),
        ("numpy", "numpy"),
    ]:
        try:
            mod = __import__(import_name)
            versions[pkg_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[pkg_name] = None

    hardware: dict[str, Any] = {"selected_device": get_best_device(device)}
    try:
        import torch

        hardware["cuda_available"] = torch.cuda.is_available()
        hardware["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if torch.cuda.is_available():
            hardware["gpu_name"] = torch.cuda.get_device_name(0)
        elif hardware["mps_available"]:
            hardware["gpu_name"] = f"Apple Silicon MPS ({platform.machine()})"
        else:
            hardware["gpu_name"] = "CPU"
    except ImportError:
        hardware["gpu_name"] = "unknown"

    return {"versions": versions, "hardware": hardware}


def load_yaml(path: Path) -> dict[str, Any]:
    """Читает YAML-файл в словарь."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_yaml(data: dict[str, Any], path: Path) -> None:
    """Сохраняет словарь в YAML в UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(data: Any, path: Path) -> None:
    """Сохраняет структуру в отформатированный JSON в UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
