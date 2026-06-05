"""
Structured file logger for the Feeldx slab pipeline.

Thread-safe: safe to call from ThreadPoolExecutor workers.
Always call setup_logger() once per session before processing.
"""

import logging
from pathlib import Path
from datetime import datetime
from threading import Lock

_logger: logging.Logger | None = None
_log_path: Path | None = None
_warn_count: int = 0
_lock = Lock()


def setup_logger(output_dir: Path) -> tuple:
    """Initialize session logger. Returns (logger, log_path)."""
    global _logger, _log_path, _warn_count
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = output_dir / f"feeldx_{ts}.log"
    _warn_count = 0

    name = f"feeldx_{ts}"
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()
    _logger.propagate = False

    fh = logging.FileHandler(_log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    _logger.addHandler(fh)

    return _logger, _log_path


def get_logger() -> logging.Logger:
    """Always returns a valid logger (NullHandler if not initialized)."""
    if _logger is not None:
        return _logger
    null = logging.getLogger("feeldx_null")
    if not null.handlers:
        null.addHandler(logging.NullHandler())
    return null


def get_log_path() -> Path | None:
    return _log_path


def get_warn_count() -> int:
    with _lock:
        return _warn_count


def _inc_warn():
    global _warn_count
    with _lock:
        _warn_count += 1


def log_session_start(pdf_name: str, pages: list, scale: int) -> None:
    pages_str = ",".join(str(p + 1) for p in pages)
    get_logger().info(f"PDF: {pdf_name} | Pages: {pages_str} | Scale: 1:{scale}")


def log_extraction_counts(page_idx: int, filled: int, reconstructed: int, after_filter: int) -> None:
    get_logger().info(
        f"PAGE {page_idx + 1} | "
        f"Filled polygons: {filled} | Reconstructed: {reconstructed} | After filter: {after_filter}"
    )


def log_slab(page_idx: int, slab) -> None:
    poly = getattr(slab, "real_polygon", None) or slab.polygon
    if poly and not poly.is_empty:
        b = poly.bounds
        bbox_str = f"{b[2] - b[0]:.0f}x{b[3] - b[1]:.0f}mm"
    else:
        bbox_str = "N/A"
    ffl_str = f"{slab.ffl_m:.3f}m" if slab.ffl_m is not None else "NO_FFL"
    area_str = f"{slab.area_m2:.2f}m2"
    get_logger().info(
        f"PAGE {page_idx + 1} | SLAB {slab.label} | "
        f"Bbox: {bbox_str} | Area: {area_str} | FFL: {ffl_str} | OK"
    )


def log_warn(page_idx: int, message: str) -> None:
    _inc_warn()
    get_logger().warning(f"PAGE {page_idx + 1} | WARN: {message}")


def log_summary(total_slabs: int, page_count: int, unique_ffls: int) -> None:
    warn = get_warn_count()
    get_logger().info(
        f"SUMMARY | Pages: {page_count} | Total slabs: {total_slabs} | "
        f"Unique FFL: {unique_ffls} | Warnings: {warn}"
    )
