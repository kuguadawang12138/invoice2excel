# -*- coding: utf-8 -*-
"""
PDF 发票识别任务排队。

规则:
  - 最大同时执行 max_workers = 1（单进程只跑 1 个 PDF）
  - 最大排队等待 max_queue = 9
  - 正在执行 + 排队 >= 10 时，新请求立即返回 BusyError（系统繁忙）

环境变量:
  INVOICE_MAX_WORKERS=1
  INVOICE_MAX_QUEUE=9
"""
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict

logger = logging.getLogger("invoice_api")

_DEFAULT_MAX_WORKERS = int(os.environ.get("INVOICE_MAX_WORKERS", "1"))
_DEFAULT_MAX_QUEUE = int(os.environ.get("INVOICE_MAX_QUEUE", "9"))

_lock = threading.Lock()
_executor = None
_max_workers = _DEFAULT_MAX_WORKERS
_max_queue = _DEFAULT_MAX_QUEUE
_pending = 0
_running = 0
_stats_lock = threading.Lock()


class BusyError(Exception):
    """队列已满，拒绝入队。"""

    def __init__(self, message: str = "系统繁忙，请稍后再试"):
        self.message = message
        super().__init__(message)


CODE_BUSY = 42900


def _ensure_executor() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=_max_workers,
                thread_name_prefix="invoice-ocr",
            )
            logger.info(
                "识别队列已启动 max_workers=%d max_queue=%d capacity=%d",
                _max_workers,
                _max_queue,
                _max_workers + _max_queue,
            )
        return _executor


def get_queue_status() -> Dict[str, Any]:
    """当前队列状态（GET /api/status 随时可查）。"""
    with _stats_lock:
        capacity = _max_workers + _max_queue
        active = _pending + _running
        return {
            "max_workers": _max_workers,
            "max_queue": _max_queue,
            "capacity": capacity,
            "running": _running,
            "pending": _pending,
            "active": active,
            "available": max(0, capacity - active),
            "busy": active >= capacity,
        }


def _acquire_slot() -> None:
    global _pending
    with _stats_lock:
        capacity = _max_workers + _max_queue
        if _pending + _running >= capacity:
            raise BusyError(
                "系统繁忙，请稍后再试（执行 %d，排队 %d，上限 %d）"
                % (_running, _pending, capacity)
            )
        _pending += 1


def _run_job(fn: Callable[[], Any]) -> Any:
    global _pending, _running
    with _stats_lock:
        _pending = max(0, _pending - 1)
        _running += 1
    try:
        return fn()
    finally:
        with _stats_lock:
            _running = max(0, _running - 1)


def submit_job(fn: Callable[[], Any]) -> Future:
    """
    提交识别任务。队列满时立即抛 BusyError，不会无限堆积。
    返回 Future，调用方 await fut.result() 获取结果。
    """
    global _pending
    ex = _ensure_executor()
    _acquire_slot()
    try:
        return ex.submit(_run_job, fn)
    except Exception:
        with _stats_lock:
            _pending = max(0, _pending - 1)
        raise


def shutdown(wait: bool = True) -> None:
    global _executor
    with _lock:
        if _executor is not None:
            _executor.shutdown(wait=wait)
            _executor = None
