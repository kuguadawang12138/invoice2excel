# -*- coding: utf-8 -*-
"""
PDF 电子发票识别 HTTP 服务。

启动:
  python invoice_api_server.py

排队规则: 1 个在执行 + 最多 9 个排队 = 共 10 人；超出返回「系统繁忙」。
GET /health、GET /api/status 随时可查，不受 POST 识别阻塞。

成功响应为 JSON（含 excel_base64），Apifox/前端可直接查看，不会乱码。
"""
import asyncio
import base64
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from invoice_pdf_ocr import run_pdf_job
from invoice_queue import CODE_BUSY, BusyError, get_queue_status, shutdown, submit_job

if hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except (ValueError, AttributeError):
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("INVOICE_DATA_DIR", os.path.join(BASE_DIR, "data"))
API_PORT = int(os.environ.get("INVOICE_API_PORT", "8001"))
# 跨域：默认允许 localhost / 127.0.0.1 任意端口（前端本地开发）
# 生产环境可设 INVOICE_CORS_ORIGINS=http://a.com,http://b.com 覆盖
CORS_ORIGINS_ENV = os.environ.get("INVOICE_CORS_ORIGINS", "").strip()
CORS_LOCALHOST_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"

logger = logging.getLogger("invoice_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_handler)
logger.propagate = False


def make_job_id() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def _json_ok(content: dict, status: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=content,
        media_type="application/json; charset=utf-8",
    )


def _error_json(code: int, message: str, status: int = 400, extra: dict = None) -> JSONResponse:
    body = {"code": code, "message": message}
    if extra:
        body.update(extra)
    return _json_ok(body, status=status)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(DATA_DIR, exist_ok=True)
    st = get_queue_status()
    logger.info(
        "服务就绪 port=%s data_dir=%s capacity=%d",
        API_PORT,
        DATA_DIR,
        st["capacity"],
    )
    yield
    shutdown(wait=True)


app = FastAPI(title="PDF 电子发票识别", version="1.2.0", lifespan=lifespan)

if CORS_ORIGINS_ENV:
    _cors_origins = [o.strip() for o in CORS_ORIGINS_ENV.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_origin_regex=CORS_LOCALHOST_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
async def health():
    """健康检查，随时可用。"""
    return {"status": "ok"}


@app.get("/api/status")
async def api_status():
    """队列状态，随时可用（识别进行中也可查）。"""
    return get_queue_status()


@app.post("/api/invoice/pdf")
async def parse_invoice_pdf(file: UploadFile = File(...)):
    """
    上传 PDF，识别后返回 JSON（含 Excel 的 base64）。
    队列满时立即返回 429，不入队。
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return _error_json(40001, "请上传 PDF 文件")

    try:
        content = await file.read()
        if not content:
            return _error_json(40002, "PDF 文件为空")
    except Exception as e:
        return _error_json(40003, "读取 PDF 失败: %s" % e)

    job_id = make_job_id()
    job_dir = os.path.join(DATA_DIR, job_id)
    pdf_path = os.path.join(job_dir, "%s.pdf" % job_id)
    excel_name = "%s_invoices.xlsx" % job_id

    try:
        os.makedirs(job_dir, exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(content)
    except Exception as e:
        return _error_json(40004, "保存 PDF 失败: %s" % e)

    t_recv = time.time()

    try:
        fut = submit_job(lambda: run_pdf_job(pdf_path, job_dir))
    except BusyError as e:
        logger.info(
            "job=%s status=busy wait_ms=0 infer_ms=0 queue=%s",
            job_id,
            get_queue_status(),
        )
        return _error_json(
            CODE_BUSY,
            "系统繁忙，请稍后再试",
            status=429,
            extra={"queue": get_queue_status(), "detail": e.message},
        )

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, fut.result)
    except Exception as e:
        logger.info(
            "job=%s status=fail pages=0 invoices=0 wait_ms=%.0f infer_ms=0 err=%s",
            job_id,
            (time.time() - t_recv) * 1000,
            e,
        )
        return _error_json(50001, "识别失败: %s" % e, status=500)

    wait_ms = round((time.time() - t_recv) * 1000 - result["elapsed_ms"], 1)
    excel_path = result["excel_path"]

    with open(excel_path, "rb") as f:
        excel_b64 = base64.b64encode(f.read()).decode("ascii")

    logger.info(
        "job=%s status=ok pages=%d invoices=%d wait_ms=%.0f infer_ms=%.0f excel=%s queue=%s",
        job_id,
        result["page_count"],
        result["invoice_count"],
        wait_ms,
        result["elapsed_ms"],
        excel_path,
        get_queue_status(),
    )

    return _json_ok(
        {
            "code": 0,
            "message": "ok",
            "job_id": job_id,
            "count": result["invoice_count"],
            "filename": excel_name,
            "excel_base64": excel_b64,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
