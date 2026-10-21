# -*- coding: utf-8 -*-
"""
发票价税合计（小写）金额识别。

提取思路（按"先大写、再同行小写"定位）：
  1. 先找含"价税合计"（大写）的文本块，拿到它的 y 坐标，作为水平基准线；
  2. 在这条水平线上（y 中心相近）找含"小写"的块；
     - 若"小写"块内直接带金额（如 "(小写)¥4510.00"），正则提取；
     - 否则沿这条线往右找金额块。
  3. 兜底：若未识别出"大写"或"小写"，退化为直接找含"小写"的块。

这样基于关键词 + 同行坐标定位，不会误取上方的"合计"(¥3991.16)。
"""
import sys
import io
import re

# 强制标准输出 UTF-8，避免 Windows 控制台中文乱码
if hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except (ValueError, AttributeError):
        pass

from paddleocr import PaddleOCR

# 金额正则：可带 ¥/￥ 前缀，整数部分可含千分位逗号，必须两位小数结尾
MONEY_RE = re.compile(r"[¥￥]?\s*([0-9][0-9,]*\.\d{2})(?!\d)")

# 判定"同一水平线"的 y 中心容差（像素）
Y_TOLERANCE = 15

# 判定「小计」行的 y 容差（更严，避免相邻合计行被误判）
XIAOJI_Y_TOLERANCE = 8

# 全局复用 OCR 实例（避免重复加载模型）
_OCR = None


def get_ocr():
    """惰性初始化并复用 PaddleOCR 实例。"""
    global _OCR
    if _OCR is None:
        _OCR = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,  # 规避 paddle 3.3.1 CPU oneDNN 推理 bug
            lang="ch",
        )
    return _OCR


def build_blocks(res):
    """把 OCR 结果整理成统一的文本块列表：text + 外接框坐标 + y 中心。"""
    texts = res["rec_texts"]
    polys = res.get("rec_polys", res.get("dt_polys"))
    blocks = []
    for t, poly in zip(texts, polys):
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        blocks.append(
            {
                "text": t,
                "x_min": min(xs),
                "x_max": max(xs),
                "y_center": (min(ys) + max(ys)) / 2,
            }
        )
    return blocks


def parse_money(text):
    """从文本中提取金额，返回 float；无金额返回 None。"""
    m = MONEY_RE.search(text)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def same_line_right_amount(blocks, ref, y_tol=Y_TOLERANCE):
    """在 ref 所在水平线上、其右侧，找最靠左的金额块。"""
    best = None  # (金额, x_min)
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


def extract_total(blocks):
    """按"先大写、再同行小写"的思路提取金额，返回 (金额, 命中方式)。"""
    # 1. 定位"价税合计"（大写）块，作为水平基准线
    daxie_blocks = [b for b in blocks if "价税合计" in b["text"]]
    for ref in daxie_blocks:
        # 2. 在同一水平线上找含"小写"的块
        for b in blocks:
            if "小写" not in b["text"]:
                continue
            if abs(b["y_center"] - ref["y_center"]) > Y_TOLERANCE:
                continue
            # 2a. "小写"块内直接带金额
            amt = parse_money(b["text"])
            if amt is not None:
                return amt, "大写定位 → 同行小写块内金额"
            # 2b. "小写"块右侧找金额
            amt = same_line_right_amount(blocks, b)
            if amt is not None:
                return amt, "大写定位 → 小写右侧金额"
        # 3. 兜底：大写行右侧直接找金额（万一"小写"没单独识别出来）
        amt = same_line_right_amount(blocks, ref)
        if amt is not None:
            return amt, "大写行右侧金额(兜底)"

    # 4. 最后兜底：直接找含"小写"的块（万一"大写"没识别出来）
    for b in [b for b in blocks if "小写" in b["text"]]:
        amt = parse_money(b["text"])
        if amt is not None:
            return amt, "小写块内金额(兜底)"
        amt = same_line_right_amount(blocks, b)
        if amt is not None:
            return amt, "小写右侧金额(兜底)"

    return None, "未找到"


def recognize_invoice_total(img_path):
    """识别单张发票的价税合计（小写）金额，返回 (金额, 命中方式)。"""
    ocr = get_ocr()
    results = ocr.predict(img_path)
    for res in results:
        blocks = build_blocks(res)
        return extract_total(blocks)
    return None, "无识别结果"


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else r"D:\paddle\photos\fp1.jpg"
    amount, how = recognize_invoice_total(img)
    print("=" * 40)
    print("图片：", img)
    if amount is not None:
        print("价税合计（小写）金额：%.2f" % amount)
        print("命中方式：", how)
    else:
        print("未能识别到金额，原因：", how)
    print("=" * 40)
