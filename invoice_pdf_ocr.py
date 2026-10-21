# -*- coding: utf-8 -*-
"""
PDF 电子发票识别模块。

流程：
  1. pdf_to_images       — 将 PDF 每一页转为图片
  2. filter_invoice_pages — 筛选含「价税合计（大写）」的发票汇总页
  3. extract_invoice_fields — 对汇总页做 PaddleOCR，按固定规则提取字段

字段提取规则（无兜底，严格按关键词 + 坐标定位）：
  - 发票号码、开票日期：关键词块内正则提取
  - 销售方名称：先找「销售方信息」，再取距离最近的「名称」块
  - 税率：找「税率」表头，取下方同列第一个数字
  - 合计金额/税额：找「合计」或同行「合」「计」，取该行右侧两个金额（先金额后税额）
  - 价税合计（大写）文本、价税合计（小写）金额：大写行定位 → 同行小写
"""
import io
import json
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

# 强制标准输出 UTF-8，避免 Windows 控制台中文乱码
if hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except (ValueError, AttributeError):
        pass

import fitz  # PyMuPDF

from invoice_ocr import (
    XIAOJI_Y_TOLERANCE,
    Y_TOLERANCE,
    build_blocks,
    get_ocr,
    parse_money,
)

# 发票号码、日期正则
INVOICE_NO_RE = re.compile(r"发票号码[：:]\s*(\d+)")
INVOICE_DATE_RE = re.compile(r"开票日期[：:]\s*(.+)")
SELLER_NAME_RE = re.compile(r"名称[：:]\s*(.+)")

# 价税合计（大写）中文金额特征
UPPER_AMOUNT_RE = re.compile(r"[壹贰叁肆伍陆柒捌玖拾佰仟万亿圆整]")

# Excel 导出列：（内部字段名, 表头显示名）；pdf页码 仅写 JSON/日志，不进 Excel
EXCEL_COLUMNS: List[Tuple[str, str]] = [
    ("发票号码", "发票号码"),
    ("开票日期", "开票日期"),
    ("销售方名称", "销售方名称"),
    ("税率", "税率"),
    ("合计金额", "合计金额"),
    ("合计税额", "合计税额"),
    ("价税合计大写", "价税合计（大写）"),
    ("价税合计小写", "价税合计（小写）"),
]

# 金额列（导出为两位小数）
MONEY_FIELDS = frozenset({"合计金额", "合计税额", "价税合计小写"})


def pdf_to_images(
    pdf_path: str,
    output_dir: Optional[str] = None,
    dpi: int = 200,
) -> List[str]:
    """
    第一步：将 PDF 每一页转为 PNG 图片。

    参数:
        pdf_path: PDF 文件路径
        output_dir: 图片输出目录；不传则使用系统临时目录
        dpi: 渲染分辨率，默认 200

    返回:
        按页码顺序排列的图片路径列表
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError("PDF 文件不存在: %s" % pdf_path)

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="invoice_pdf_")
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_paths: List[str] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img_path = os.path.join(output_dir, "page_%04d.png" % (page_index + 1))
            pix.save(img_path)
            image_paths.append(img_path)
    finally:
        doc.close()

    return image_paths


def ocr_blocks(img_path: str) -> List[Dict[str, Any]]:
    """对单张图片执行 OCR，返回文本块列表。"""
    ocr = get_ocr()
    results = ocr.predict(img_path)
    for res in results:
        return build_blocks(res)
    return []


def page_has_total_uppercase(blocks: List[Dict[str, Any]]) -> bool:
    """判断该页是否含「价税合计（大写）」—— 即发票汇总页。"""
    return any("价税合计" in b["text"] for b in blocks)


def find_summary_pages_from_pdf(pdf_path: str) -> List[int]:
    """
    利用 PDF 文本层快速定位含「价税合计」的页码（1 起）。
    无文本层或扫描件时返回空列表，由 OCR 逐页筛选兜底。
    """
    doc = fitz.open(pdf_path)
    page_nums: List[int] = []
    try:
        for page_index in range(len(doc)):
            text = doc.load_page(page_index).get_text()
            if "价税合计" in text:
                page_nums.append(page_index + 1)
    finally:
        doc.close()
    return page_nums


def filter_invoice_pages(
    image_paths: List[str],
    pdf_path: Optional[str] = None,
) -> List[Tuple[str, int]]:
    """
    第二步：从全部页面中筛出含「价税合计（大写）」的发票汇总页。
    优先用 PDF 文本层定位；无文本层时再 OCR 逐页判断。

    返回:
        [(图片路径, 页码)] 列表，页码从 1 开始
    """
    if pdf_path:
        text_pages = find_summary_pages_from_pdf(pdf_path)
        if text_pages:
            return [
                (image_paths[page_num - 1], page_num)
                for page_num in text_pages
                if 0 < page_num <= len(image_paths)
            ]

    invoice_pages: List[Tuple[str, int]] = []
    for page_num, img_path in enumerate(image_paths, start=1):
        blocks = ocr_blocks(img_path)
        if page_has_total_uppercase(blocks):
            invoice_pages.append((img_path, page_num))
    return invoice_pages


def _block_center_x(block: Dict[str, Any]) -> float:
    return (block["x_min"] + block["x_max"]) / 2.0


def _distance(b1: Dict[str, Any], b2: Dict[str, Any]) -> float:
    dx = _block_center_x(b1) - _block_center_x(b2)
    dy = b1["y_center"] - b2["y_center"]
    return dx * dx + dy * dy


def _same_line_right_amount(
    blocks: List[Dict[str, Any]],
    ref: Dict[str, Any],
    y_tol: int = Y_TOLERANCE,
) -> Optional[float]:
    """在 ref 所在水平线右侧，找最靠左的金额。"""
    best: Optional[Tuple[float, float]] = None
    for b in blocks:
        if b is ref:
            continue
        if abs(b["y_center"] - ref["y_center"]) > y_tol:
            continue
        if b["x_min"] < ref["x_min"]:
            continue
        amt = parse_money(b["text"])
        if amt is not None and (best is None or b["x_min"] < best[1]):
            best = (amt, b["x_min"])
    return best[0] if best else None


def extract_invoice_number(blocks: List[Dict[str, Any]]) -> Optional[str]:
    """提取发票号码。"""
    for b in blocks:
        if "发票号码" not in b["text"]:
            continue
        m = INVOICE_NO_RE.search(b["text"])
        if m:
            return m.group(1)
    return None


def extract_invoice_date(blocks: List[Dict[str, Any]]) -> Optional[str]:
    """提取开票日期。"""
    for b in blocks:
        if "开票日期" not in b["text"]:
            continue
        m = INVOICE_DATE_RE.search(b["text"])
        if m:
            return m.group(1).strip()
    return None


def extract_seller_name(blocks: List[Dict[str, Any]]) -> Optional[str]:
    """
    提取销售方名称。
    先找「销售方信息」，再取与其距离最近的「名称」块。
    """
    seller_info_blocks = [b for b in blocks if "销售方信息" in b["text"]]
    name_blocks = [
        b for b in blocks
        if "名称" in b["text"] and ("：" in b["text"] or ":" in b["text"])
    ]
    if not seller_info_blocks or not name_blocks:
        return None

    ref = seller_info_blocks[0]
    nearest = min(name_blocks, key=lambda b: _distance(ref, b))
    m = SELLER_NAME_RE.search(nearest["text"])
    if m:
        return m.group(1).strip()
    return None


def extract_tax_rate(blocks: List[Dict[str, Any]]) -> Optional[str]:
    """
    提取税率。
    找含「税率」的表头块，取下方同列（x 中心落在表头范围内）第一个数字。
    """
    header = None
    for b in blocks:
        if "税率" in b["text"]:
            header = b
            break
    if header is None:
        return None

    col_x_min = header["x_min"] - 10
    col_x_max = header["x_max"] + 10
    candidates: List[Tuple[float, str]] = []

    for b in blocks:
        if b["y_center"] <= header["y_center"] + 5:
            continue
        cx = _block_center_x(b)
        if cx < col_x_min or cx > col_x_max:
            continue
        text = b["text"].strip()
        if re.match(r"^\d+(\.\d+)?%?$", text):
            candidates.append((b["y_center"], text))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _is_xiaoji_row(blocks: List[Dict[str, Any]], row_y: float) -> bool:
    """判断该行是否为「小计」行（含整块「小计」、「小合」或同行「小」+「计」）。"""
    for b in blocks:
        if abs(b["y_center"] - row_y) > XIAOJI_Y_TOLERANCE:
            continue
        text = b["text"].strip()
        if "小计" in text:
            return True
        if text.startswith("小") and ("计" in text or "合" in text):
            return True

    xiao_blocks = [
        b
        for b in blocks
        if b["text"].strip().startswith("小")
        and abs(b["y_center"] - row_y) <= XIAOJI_Y_TOLERANCE
    ]
    ji_blocks = [
        b
        for b in blocks
        if "计" in b["text"].strip()
        and abs(b["y_center"] - row_y) <= XIAOJI_Y_TOLERANCE
    ]
    for xiao in xiao_blocks:
        for ji in ji_blocks:
            if ji["x_min"] > xiao["x_min"] and ji["x_min"] - xiao["x_max"] < 50:
                return True
    return False


def _row_money_amounts(
    blocks: List[Dict[str, Any]],
    row_y: float,
    min_x: float = 400,
) -> List[Tuple[float, float]]:
    """取某一行右侧的金额块列表，返回 [(x_min, amount), ...]。"""
    amounts: List[Tuple[float, float]] = []
    for b in blocks:
        if abs(b["y_center"] - row_y) > Y_TOLERANCE:
            continue
        if b["x_min"] < min_x:
            continue
        amt = parse_money(b["text"])
        if amt is not None:
            amounts.append((b["x_min"], amt))
    amounts.sort(key=lambda item: item[0])
    return amounts


def _find_heji_row_y(
    blocks: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """
    定位「合计」行 y 坐标。
    支持：整块「合计」；OCR 拆成「合」「计」；或「合」「计」被误识时按金额行定位。
    排除「价税合计」「小计」；多候选时取「价税合计」上方最近一行。
    """
    daxie_blocks = [b for b in blocks if "价税合计" in b["text"]]
    daxie_y = daxie_blocks[0]["y_center"] if daxie_blocks else None

    candidates: List[Tuple[float, Dict[str, Any], int]] = []

    for b in blocks:
        text = b["text"].strip()
        if "小计" in text or "价税合计" in text:
            continue
        if text.startswith("小"):
            continue
        if text == "合计" or (text.startswith("合计") and "价税" not in text):
            if not _is_xiaoji_row(blocks, b["y_center"]):
                candidates.append((b["y_center"], b, 2))
            continue
        if "合计" in text:
            if not _is_xiaoji_row(blocks, b["y_center"]):
                candidates.append((b["y_center"], b, 2))

    he_blocks = [
        b
        for b in blocks
        if b["text"].strip() in ("合", "合合")
        or (b["text"].strip().endswith("合") and not b["text"].strip().startswith("小"))
    ]
    ji_blocks = [
        b
        for b in blocks
        if b["text"].strip() in ("计", "计计")
        or b["text"].strip().endswith("计")
    ]

    for he in he_blocks:
        if he["text"].strip().startswith("小"):
            continue
        for ji in ji_blocks:
            if ji["text"].strip().startswith("小"):
                continue
            if abs(he["y_center"] - ji["y_center"]) > Y_TOLERANCE:
                continue
            if ji["x_min"] < he["x_min"]:
                continue
            y = (he["y_center"] + ji["y_center"]) / 2.0
            if _is_xiaoji_row(blocks, y):
                continue
            candidates.append((y, he, 2))

    row_ys = sorted({round(b["y_center"]) for b in blocks})
    for row_y in row_ys:
        if daxie_y is not None and row_y >= daxie_y - 5:
            continue
        if _is_xiaoji_row(blocks, float(row_y)):
            continue
        amounts = _row_money_amounts(blocks, float(row_y))
        if len(amounts) >= 2:
            ref = {
                "y_center": float(row_y),
                "x_min": 0,
                "x_max": 0,
                "text": "",
            }
            candidates.append((float(row_y), ref, 1))

    if not candidates:
        return None, None

    if daxie_y is not None:
        above = [(y, ref, pri) for y, ref, pri in candidates if y < daxie_y - 5]
        if above:
            above.sort(key=lambda item: (item[2], item[0]), reverse=True)
            best = above[0]
            return best[0], best[1]

    candidates.sort(key=lambda item: (item[2], item[0]), reverse=True)
    return candidates[0][0], candidates[0][1]


def extract_subtotal_amounts(
    blocks: List[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float]]:
    """
    提取「合计」行的金额与税额。
    找「合计」或「合+计」所在行，取该行右侧两个金额，依次为金额、税额。
    """
    row_y, ref = _find_heji_row_y(blocks)
    if row_y is None or ref is None:
        return None, None

    right_edge = ref["x_max"]
    if ref["text"].strip() == "合":
        for b in blocks:
            if b["text"].strip() == "计" and abs(b["y_center"] - row_y) <= Y_TOLERANCE:
                right_edge = max(right_edge, b["x_max"])

    amounts: List[Tuple[float, float]] = []
    for b in blocks:
        if abs(b["y_center"] - row_y) > Y_TOLERANCE:
            continue
        if b["x_min"] <= right_edge + 5:
            continue
        amt = parse_money(b["text"])
        if amt is not None:
            amounts.append((b["x_min"], amt))

    amounts.sort(key=lambda item: item[0])
    if len(amounts) < 2:
        return None, None
    return amounts[0][1], amounts[1][1]


def extract_total_uppercase_text(blocks: List[Dict[str, Any]]) -> Optional[str]:
    """提取价税合计（大写）旁的中文大写金额文本。"""
    daxie_blocks = [b for b in blocks if "价税合计" in b["text"]]
    for ref in daxie_blocks:
        best: Optional[Tuple[float, str]] = None
        for b in blocks:
            if b is ref:
                continue
            if abs(b["y_center"] - ref["y_center"]) > Y_TOLERANCE:
                continue
            if b["x_min"] <= ref["x_max"]:
                continue
            if "小写" in b["text"]:
                continue
            if parse_money(b["text"]) is not None:
                continue
            if not UPPER_AMOUNT_RE.search(b["text"]):
                continue
            if best is None or b["x_min"] < best[0]:
                best = (b["x_min"], b["text"].strip())
        if best:
            return best[1]
    return None


def extract_total_lower_amount(blocks: List[Dict[str, Any]]) -> Optional[float]:
    """
    提取价税合计（小写）金额。
    先找「价税合计（大写）」行，再在同一水平线上找「小写」及其金额。
    无兜底逻辑。
    """
    daxie_blocks = [b for b in blocks if "价税合计" in b["text"]]
    for ref in daxie_blocks:
        for b in blocks:
            if "小写" not in b["text"]:
                continue
            if abs(b["y_center"] - ref["y_center"]) > Y_TOLERANCE:
                continue
            amt = parse_money(b["text"])
            if amt is not None:
                return amt
            amt = _same_line_right_amount(blocks, b)
            if amt is not None:
                return amt
    return None


def extract_invoice_fields(
    img_path: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
    prev_page_image: Optional[str] = None,
) -> Dict[str, Any]:
    """
    第三步：对单张发票汇总页提取全部字段。

    参数:
        img_path: 汇总页图片路径
        blocks: 已有 OCR 块（可选，避免重复 OCR）
        prev_page_image: 上一页图片路径；汇总页无物料/无税率时到上一页找税率

    返回字典包含各字段及 page_image 路径。
    """
    if blocks is None:
        blocks = ocr_blocks(img_path)

    amount, tax = extract_subtotal_amounts(blocks)
    total_lower = extract_total_lower_amount(blocks)

    tax_rate = extract_tax_rate(blocks)
    if tax_rate is None and prev_page_image and os.path.isfile(prev_page_image):
        tax_rate = extract_tax_rate(ocr_blocks(prev_page_image))

    return {
        "page_image": img_path,
        "发票号码": extract_invoice_number(blocks),
        "开票日期": extract_invoice_date(blocks),
        "销售方名称": extract_seller_name(blocks),
        "税率": tax_rate,
        "合计金额": amount,
        "合计税额": tax,
        "价税合计大写": extract_total_uppercase_text(blocks),
        "价税合计小写": total_lower,
    }


def process_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
    dpi: int = 200,
    keep_images: bool = True,
) -> List[Dict[str, Any]]:
    """
    完整流程：PDF → 分页图片 → 筛选汇总页 → 字段提取。

    参数:
        pdf_path: PDF 文件路径
        output_dir: 分页图片输出目录；不传则用临时目录
        dpi: 渲染 DPI
        keep_images: 是否在结果中保留 page_image 字段

    返回:
        每张发票汇总页的字段字典列表
    """
    if output_dir is None:
        output_dir = os.path.join(
            tempfile.gettempdir(),
            "invoice_pdf_%s" % os.path.splitext(os.path.basename(pdf_path))[0],
        )

    image_paths = pdf_to_images(pdf_path, output_dir=output_dir, dpi=dpi)
    invoice_pages = filter_invoice_pages(image_paths, pdf_path=pdf_path)

    results: List[Dict[str, Any]] = []
    for img_path, page_num in invoice_pages:
        prev_img = image_paths[page_num - 2] if page_num > 1 else None
        fields = extract_invoice_fields(img_path, prev_page_image=prev_img)
        fields["pdf页码"] = page_num
        if not keep_images:
            fields.pop("page_image", None)
        results.append(fields)

    return results


def run_pdf_job(
    pdf_path: str,
    job_dir: str,
    dpi: int = 200,
) -> Dict[str, Any]:
    """
    在指定任务目录内完成 PDF 识别，并写出 JSON + Excel。

    目录结构（job_dir 一般为 data/{时间戳}/）：
      {时间戳}.pdf、pages/ 分页图、{时间戳}_invoices.json、{时间戳}_invoices.xlsx

    返回:
        { job_id, page_count, invoice_count, excel_path, json_path, invoices, elapsed_ms }
    """
    import time

    t0 = time.time()
    job_id = os.path.basename(os.path.normpath(job_dir))
    pages_dir = os.path.join(job_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)

    invoices = process_pdf(
        pdf_path,
        output_dir=pages_dir,
        dpi=dpi,
        keep_images=False,
    )

    json_path = os.path.join(job_dir, "%s_invoices.json" % job_id)
    excel_path = os.path.join(job_dir, "%s_invoices.xlsx" % job_id)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(invoices, f, ensure_ascii=False, indent=2)
    export_invoices_to_excel(invoices, excel_path)

    import logging
    _log = logging.getLogger("invoice_api")
    for inv in invoices:
        _log.info(
            "job=%s saved invoice pdf_page=%s no=%s seller=%s total=%s",
            job_id,
            inv.get("pdf页码"),
            inv.get("发票号码"),
            inv.get("销售方名称"),
            inv.get("价税合计小写"),
        )
    _log.info(
        "job=%s saved json=%s excel=%s invoices=%d",
        job_id,
        json_path,
        excel_path,
        len(invoices),
    )

    page_count = len(
        [f for f in os.listdir(pages_dir) if f.lower().endswith(".png")]
    ) if os.path.isdir(pages_dir) else 0

    return {
        "job_id": job_id,
        "page_count": page_count,
        "invoice_count": len(invoices),
        "excel_path": excel_path,
        "json_path": json_path,
        "invoices": invoices,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }


def export_invoices_to_excel(
    invoices: List[Dict[str, Any]],
    excel_path: str,
) -> str:
    """
    将识别结果导出为 Excel（.xlsx），每张发票汇总页一行。

    参数:
        invoices: process_pdf 或 extract_invoice_fields 的结果列表
        excel_path: 输出 xlsx 路径

    返回:
        实际写入的文件路径
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    wb = Workbook()
    ws = wb.active
    ws.title = "发票识别结果"

    headers = [title for _, title in EXCEL_COLUMNS]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

    for inv in invoices:
        row: List[Any] = []
        for field, _title in EXCEL_COLUMNS:
            val = inv.get(field)
            if val is None:
                row.append("")
            elif field in MONEY_FIELDS and isinstance(val, (int, float)):
                row.append(round(float(val), 2))
            else:
                row.append(val)
        ws.append(row)

    # 金额列右对齐、保留两位小数显示
    money_col_indexes = [
        idx + 1
        for idx, (field, _) in enumerate(EXCEL_COLUMNS)
        if field in MONEY_FIELDS
    ]
    for row_idx in range(2, ws.max_row + 1):
        for col_idx in money_col_indexes:
            cell = ws.cell(row=row_idx, column=col_idx)
            if cell.value != "" and cell.value is not None:
                cell.number_format = "#,##0.00"
                cell.alignment = Alignment(horizontal="right")

    # 自动列宽（按表头与内容估算）
    for col_idx, (_field, title) in enumerate(EXCEL_COLUMNS, start=1):
        max_len = len(title)
        for row_idx in range(2, ws.max_row + 1):
            val = ws.cell(row=row_idx, column=col_idx).value
            if val is not None:
                max_len = max(max_len, len(str(val)))
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(
            max_len + 2, 50
        )

    os.makedirs(os.path.dirname(os.path.abspath(excel_path)) or ".", exist_ok=True)
    wb.save(excel_path)
    return excel_path


def _format_result(fields: Dict[str, Any]) -> str:
    lines = [
        "PDF页码: %s" % fields.get("pdf页码", "-"),
        "发票号码: %s" % fields.get("发票号码", "-"),
        "开票日期: %s" % fields.get("开票日期", "-"),
        "销售方名称: %s" % fields.get("销售方名称", "-"),
        "税率: %s" % fields.get("税率", "-"),
        "合计金额: %s" % (
            "%.2f" % fields["合计金额"] if fields.get("合计金额") is not None else "-"
        ),
        "合计税额: %s" % (
            "%.2f" % fields["合计税额"] if fields.get("合计税额") is not None else "-"
        ),
        "价税合计（大写）: %s" % fields.get("价税合计大写", "-"),
        "价税合计（小写）: %s" % (
            "%.2f" % fields["价税合计小写"]
            if fields.get("价税合计小写") is not None
            else "-"
        ),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python invoice_pdf_ocr.py <pdf路径> [图片输出目录]")
        print("  python invoice_pdf_ocr.py --image <单张发票图片>")
        sys.exit(1)

    if sys.argv[1] == "--image":
        img = sys.argv[2] if len(sys.argv) > 2 else r"D:\paddle\photos\fp1.jpg"
        result = extract_invoice_fields(img)
        print("=" * 50)
        print("单张图片识别结果")
        print("=" * 50)
        for key, val in result.items():
            if key == "page_image":
                continue
            if isinstance(val, float):
                print("%s: %.2f" % (key, val))
            else:
                print("%s: %s" % (key, val))
        sys.exit(0)

    pdf = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    invoices = process_pdf(pdf, output_dir=out_dir)

    print("=" * 50)
    print("PDF: %s" % pdf)
    print("识别到 %d 张发票汇总页" % len(invoices))
    print("=" * 50)

    for i, inv in enumerate(invoices, start=1):
        print("\n--- 发票 %d ---" % i)
        print(_format_result(inv))

    json_path = os.path.splitext(pdf)[0] + "_invoices.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(invoices, f, ensure_ascii=False, indent=2)
    print("\n结果已保存: %s" % json_path)

    excel_path = os.path.splitext(pdf)[0] + "_invoices.xlsx"
    export_invoices_to_excel(invoices, excel_path)
    print("Excel 已保存: %s" % excel_path)
