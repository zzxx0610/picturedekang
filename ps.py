
# st_app.py
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
import io
import zipfile
import os
import re

# OCR
try:
    import pytesseract
except ImportError:
    pytesseract = None

# --- 配置和常量 ---
RECT1_COLOR = (255, 0, 0)   # 红色
RECT2_COLOR = (0, 0, 255)   # 蓝色
PREVIEW_WIDTH = 600

# --- 工具函数 ---
def create_hq_preview(img: Image.Image, target_width: int) -> Image.Image:
    """创建高质量预览图像"""
    w, h = img.size
    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)
    preview_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return preview_img

def clamp_rect(rect_def, base_size):
    """将输入矩形限制在 base_size 内，返回 (left, top, right, bottom)"""
    bw, bh = base_size
    x, y, w, h = rect_def['x'], rect_def['y'], rect_def['width'], rect_def['height']
    x = max(0, min(x, bw - 1))
    y = max(0, min(y, bh - 1))
    w = max(1, min(w, bw - x))
    h = max(1, min(h, bh - y))
    return (x, y, x + w, y + h)

def paste_position(rect_def):
    """返回粘贴左上角位置 (x, y)"""
    return (rect_def['x'], rect_def['y'])

def ocr_digits_from_patch(patch_img: Image.Image):
    """
    对补丁做简单预处理并用 pytesseract 提取数字串。
    返回 (best_number_str, candidates_list)
    """
    if pytesseract is None:
        return "", []

    g = patch_img.convert('L')
    from PIL import ImageOps as _IOps
    g = _IOps.autocontrast(g, cutoff=2)
    target_min_w = 800
    scale = max(1, target_min_w / g.width) if g.width < target_min_w else 1
    if scale > 1:
        g = g.resize((int(g.width * scale), int(g.height * scale)), Image.Resampling.LANCZOS)

    config = '--psm 6 -c tessedit_char_whitelist=0123456789'

    try:
        data = pytesseract.image_to_data(g, lang='eng', output_type=pytesseract.Output.DICT, config=config)
        words = [w for w in data['text'] if w and w.strip()]
    except Exception:
        words = []

    candidates = []
    for w in words:
        candidates.extend(re.findall(r'\d+', w))

    if not candidates:
        try:
            s = pytesseract.image_to_string(g, lang='eng', config=config)
            candidates.extend(re.findall(r'\d+', s))
        except Exception:
            pass

    if candidates:
        best = max(candidates, key=len)
        return best, candidates
    return "", []

def sanitize_filename(name: str) -> str:
    """清理文件名非法字符"""
    safe = re.sub(r'[^0-9A-Za-z_\-一-龥]', '_', name)
    return safe.strip('_') or "未命名"

def unique_name(base: str, ext: str, used: set):
    """确保文件名唯一"""
    candidate = f"{base}{ext}"
    if candidate not in used:
        used.add(candidate)
        return candidate
    idx = 1
    while True:
        candidate = f"{base}_{idx}{ext}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        idx += 1

def process_single_image_pair(base_img_pil, source_img_pil, rect1_def, rect2_def, base_size):
    """处理单对图像：返回合成图，以及用于 OCR 的两个补丁"""
    bw, bh = base_size

    rect1_coords = clamp_rect(rect1_def, base_size)
    rect2_coords = clamp_rect(rect2_def, base_size)

    patch1 = source_img_pil.crop(rect1_coords)
    patch2 = source_img_pil.crop(rect2_coords)

    final_image = base_img_pil.copy()
    final_image.paste(patch1, paste_position(rect1_def))
    final_image.paste(patch2, paste_position(rect2_def))
    return final_image, patch1, patch2

# --- Streamlit UI 布局 ---
st.set_page_config(layout="wide")
st.title("🚀 批量图像合成工具 (v3.2)")
st.info("以首次上传的底图尺寸为模板；源图尺寸必须与底图一致，否则报错。其余功能保持：OCR 命名、两份文件（运单号/运单号-装1）。")

# --- 侧边栏上传和控制 ---
with st.sidebar:
    st.header("1. 上传图像")
    base_file = st.file_uploader("上传基础图像（底图）", type=['png', 'jpg', 'jpeg'])
    source_files = st.file_uploader(
        "上传多个源图（尺寸需与底图一致）",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

# --- 主界面 ---
if base_file:
    # 加载底图（EXIF 方向修正）
    base_img_pil_orig = ImageOps.exif_transpose(Image.open(base_file).convert("RGBA"))
    base_w, base_h = base_img_pil_orig.size
    st.markdown(f"底图尺寸：{base_w} × {base_h} 像素")

    # 默认矩形参数（根据你提供的值），并在 UI 初始化时按底图尺寸夹紧默认值
    def clamp_default_xywhr(x, y, w, h):
        x0 = min(max(0, x), base_w - 1)
        y0 = min(max(0, y), base_h - 1)
        w0 = max(1, min(w, base_w - x0))
        h0 = max(1, min(h, base_h - y0))
        return x0, y0, w0, h0

    def_r1x, def_r1y, def_r1w, def_r1h = clamp_default_xywhr(360, 210, 2150, 350)
    def_r2x, def_r2y, def_r2w, def_r2h = clamp_default_xywhr(2280, 870, 150, 180)

    with st.sidebar:
        if source_files:
            st.header("2. 定义裁剪区域（应用于所有源图）")
        else:
            st.header("2. 定义裁剪区域")

        with st.expander("红色区域 (补丁1)", expanded=True):
            r1_x = st.number_input("X坐标", min_value=0, max_value=base_w, value=int(def_r1x), key='r1x')
            r1_y = st.number_input("Y坐标", min_value=0, max_value=base_h, value=int(def_r1y), key='r1y')
            r1_w = st.number_input("宽度", min_value=1, max_value=max(1, base_w - int(r1_x)), value=int(min(def_r1w, max(1, base_w - int(r1_x)))), key='r1w')
            r1_h = st.number_input("高度", min_value=1, max_value=max(1, base_h - int(r1_y)), value=int(min(def_r1h, max(1, base_h - int(r1_y)))), key='r1h')
            rect1_definition = {"x": int(r1_x), "y": int(r1_y), "width": int(r1_w), "height": int(r1_h)}

        with st.expander("蓝色区域 (补丁2)", expanded=True):
            r2_x = st.number_input("X坐标 ", min_value=0, max_value=base_w, value=int(def_r2x), key='r2x')
            r2_y = st.number_input("Y坐标 ", min_value=0, max_value=base_h, value=int(def_r2y), key='r2y')
            r2_w = st.number_input("宽度 ", min_value=1, max_value=max(1, base_w - int(r2_x)), value=int(min(def_r2w, max(1, base_w - int(r2_x)))), key='r2w')
            r2_h = st.number_input("高度 ", min_value=1, max_value=max(1, base_h - int(r2_y)), value=int(min(def_r2h, max(1, base_h - int(r2_y)))), key='r2h')
            rect2_definition = {"x": int(r2_x), "y": int(r2_y), "width": int(r2_w), "height": int(r2_h)}

    # 源图尺寸校验（若已上传）
    bad_sources = []
    if source_files:
        for f in source_files:
            try:
                img = ImageOps.exif_transpose(Image.open(f))
                if img.size != (base_w, base_h):
                    bad_sources.append((f.name, img.size))
            except Exception:
                bad_sources.append((f.name, "无法读取尺寸"))

    # 画矩形预览（在底图上）
    base_with_rects = base_img_pil_orig.copy()
    draw = ImageDraw.Draw(base_with_rects)
    draw.rectangle(
        (rect1_definition['x'], rect1_definition['y'],
         rect1_definition['x'] + rect1_definition['width'],
         rect1_definition['y'] + rect1_definition['height']),
        outline=RECT1_COLOR, width= max(1, int(max(base_w, base_h) * 0.004))  # 动态线宽
    )
    draw.rectangle(
        (rect2_definition['x'], rect2_definition['y'],
         rect2_definition['x'] + rect2_definition['width'],
         rect2_definition['y'] + rect2_definition['height']),
        outline=RECT2_COLOR, width= max(1, int(max(base_w, base_h) * 0.004))
    )

    hq_base_preview = create_hq_preview(base_with_rects, PREVIEW_WIDTH)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("基础图像与区域预览")
        st.image(hq_base_preview, caption=f"底图尺寸：{base_w}×{base_h}px（预览已缩放）")

    if source_files:
        # 展示首张源图预览（仅作展示，不做尺寸变更）
        try:
            first_source_img_pil_orig = ImageOps.exif_transpose(Image.open(source_files[0]).convert("RGBA"))
            hq_source_preview = create_hq_preview(first_source_img_pil_orig, PREVIEW_WIDTH)
            with col2:
                st.subheader(f"源图预览 (已上传 {len(source_files)} 张)")
                cap = f"首张源图尺寸：{first_source_img_pil_orig.size[0]}×{first_source_img_pil_orig.size[1]}px"
                if bad_sources and source_files[0].name in [n for n, _ in bad_sources]:
                    cap += "（尺寸不一致）"
                st.image(hq_source_preview, caption=cap)
        except Exception:
            with col2:
                st.subheader(f"源图预览 (已上传 {len(source_files)} 张)")
                st.info("首张源图无法预览")

    st.divider()

    # 尺寸不一致时，提示并阻止处理
    if source_files and bad_sources:
        st.error("以下源图尺寸与底图不一致，请更换或重新导出后再上传：")
        for name, sz in bad_sources:
            st.write(f"- {name}：{sz}")

    process_button = st.button(
        f"🚀 开始批量处理并OCR命名 ({len(source_files) if source_files else 0} 张)",
        use_container_width=True,
        disabled=not source_files or bool(bad_sources)
    )

    if source_files and process_button:
        # OCR 可用性检查
        if pytesseract is None:
            st.error("未安装 pytesseract。请先安装：\n"
                     "- macOS: brew install tesseract\n"
                     "- Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                     "- Windows: 安装 Tesseract 并将目录加入 PATH\n"
                     "然后 pip install pytesseract，再重新运行本应用。")
            st.stop()

        # [修改] 创建两个ZIP文件的内存缓冲区
        zip_buffer_main = io.BytesIO()
        zip_buffer_zhuang1 = io.BytesIO()

        progress_bar = st.progress(0, text="开始处理...")
        last_processed_image = None
        ocr_logs = []

        # [修改] 为两个ZIP包分别设置命名冲突检查
        used_names_main = set()
        used_names_zhuang1 = set()
# 修改后的代码
# [修改] 使用 with 同时管理两个ZIP文件

        with zipfile.ZipFile(zip_buffer_main, 'w', zipfile.ZIP_DEFLATED) as zip_main,\
            zipfile.ZipFile(zip_buffer_zhuang1, 'w', zipfile.ZIP_DEFLATED) as zip_zhuang1:
            for i, source_file_uploaded in enumerate(source_files):
                progress_text = f"处理中: {source_file_uploaded.name} ({i+1}/{len(source_files)})"
                progress_bar.progress((i + 1) / len(source_files), text=progress_text)

                current_source_img = ImageOps.exif_transpose(Image.open(source_file_uploaded).convert("RGBA"))

                # 再次严谨校验尺寸（防御）
                if current_source_img.size != (base_w, base_h):
                    st.error(f"尺寸不一致：{source_file_uploaded.name} 的尺寸为 {current_source_img.size}，底图为 {(base_w, base_h)}。已跳过该文件。")
                    continue

                # 合成 + 取补丁做 OCR
                final_image, patch1, patch2 = process_single_image_pair(
                    base_img_pil_orig,
                    current_source_img,
                    rect1_definition,
                    rect2_definition,
                    base_size=(base_w, base_h)
                )

                # 只对 R2（蓝色矩形）做 OCR
                num2, cand2 = ocr_digits_from_patch(patch2)

                status = ""
                warn = ""

                def pick_11_digits(primary, candidates):
                    """优先返回 primary 的 11 位数字，否则在候选里找"""
                    if primary and len(primary) == 11 and primary.isdigit():
                        return primary
                    for c in candidates:
                        if len(c) == 11 and c.isdigit():
                            return c
                    return ""
                cand_set2 = list(set(cand2 or []))  # 不要把 num2 混进去
                r2_pick = pick_11_digits(num2, cand_set2)


                # 原始运单号（不受文件名冲突影响）
                if r2_pick:
                    final_number = r2_pick
                    status = "OK_R2"
                else:
                    original_filename, _ = os.path.splitext(source_file_uploaded.name)
                    final_number = original_filename
                    status = "FAIL"
                    warn = "R2未识别到有效运单号，回退为源图文件名"

                # 用原始运单号生成安全文件名
                picked_safe = sanitize_filename(final_number)

                # 文件保存时检查冲突，但不影响 final_number
                img_buffer = io.BytesIO()
                final_image.save(img_buffer, format="PNG")
                img_bytes = img_buffer.getvalue()

                # 修改后的代码
                # [修改] 分别处理和写入两个ZIP包
                name1_base = picked_safe # <--- 在这里定义了 name1_base
                name2_base = f"{picked_safe}-装1"
                out1 = unique_name(name1_base, ".png", used_names_main)
                out2 = unique_name(name2_base, ".png", used_names_zhuang1)

                zip_main.writestr(out1, img_bytes)
                zip_zhuang1.writestr(out2, img_bytes)

                last_processed_image = final_image

                # 记录日志（使用原始运单号 final_number）
                ocr_logs.append({
                    "序号": i + 1,
                    "源图文件": source_file_uploaded.name,
                    "R2识别": num2,
                    "最终运单号": final_number,
                    "输出文件1": out1,
                    "输出文件2": out2,
                    "状态": status,
                    "备注": warn
                })

        # 保存到 session_state
        st.session_state.zip_main = zip_buffer_main.getvalue()
        st.session_state.zip_zhuang1 = zip_buffer_zhuang1.getvalue()
        st.session_state.ocr_logs = ocr_logs
        st.session_state.last_processed_image = last_processed_image
        st.session_state.processing_complete = True

        progress_bar.empty()
        st.success("🎉 批量处理完成！已根据 OCR 运单号完成命名并打包。")

    # 始终显示结果部分，如果已处理
    if 'processing_complete' in st.session_state and st.session_state.processing_complete:
        # 结果预览 + OCR 报告
        if 'last_processed_image' in st.session_state:
            st.subheader("最后处理结果预览")
            st.image(st.session_state.last_processed_image, caption="这是批量处理中最后一张图像的结果", use_container_width=True)

        st.subheader("OCR 识别报告")
        try:
            import pandas as pd
            st.dataframe(pd.DataFrame(st.session_state.ocr_logs), use_container_width=True)
        except Exception:
            for row in st.session_state.ocr_logs:
                st.write(row)

        # 汇总提醒（仅提示需要关注的情况）
        issues = [r for r in st.session_state.ocr_logs if r["状态"] == "FAIL"]
        if issues:
            st.warning("以下源图 OCR 识别存在需要关注的情况：")
            for r in issues:
                st.write(f"- 第{r['序号']}张（{r['源图文件']}）：{r['备注']}")

        # 修改后的代码
        # [修改] 提供两个独立的下载按钮
        st.subheader("下载处理结果")
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            st.download_button(
                label="📥 下载主文件 (.zip)",
                data=st.session_state.zip_main,
                file_name="批量合成_主文件.zip",
                mime="application/zip",
                use_container_width=True
            )

        with dl_col2:
            st.download_button(
                label="📥 下载'-装1'文件 (.zip)",
                data=st.session_state.zip_zhuang1,
                file_name="批量合成_装1.zip",
                mime="application/zip",
                use_container_width=True
            )
else:
    st.warning("请先在侧边栏上传基础图像（底图）。再上传源图时，请确保尺寸与底图完全一致。")
