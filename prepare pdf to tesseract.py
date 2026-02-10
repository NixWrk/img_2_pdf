# prepare_pdf_for_tesseract.py
# pip install pymupdf pillow

import os
from io import BytesIO
import fitz  # PyMuPDF
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# Настройки "легкого" уменьшения
TARGET_LONG_SIDE_PX = 3500   # максимум по длинной стороне
RENDER_DPI = 300             # во что рендерим страницу перед ресайзом
JPEG_QUALITY = 85            # баланс размер/качество

def pick_file_and_folder():
    root = tk.Tk()
    root.withdraw()
    pdf_path = filedialog.askopenfilename(
        title="Выберите PDF из Office Lens",
        filetypes=[("PDF files", "*.pdf")]
    )
    if not pdf_path:
        raise SystemExit("Файл не выбран.")
    out_dir = filedialog.askdirectory(title="Выберите папку для сохранения")
    if not out_dir:
        raise SystemExit("Папка не выбрана.")
    return pdf_path, out_dir

def pil_to_jpeg_bytes(img: Image.Image, quality: int) -> bytes:
    bio = BytesIO()
    # сохраняем в JPEG без EXIF/ориентаций и прочих сюрпризов
    img.save(bio, format="JPEG", quality=quality, optimize=True, progressive=True)
    return bio.getvalue()

def main():
    pdf_path, out_dir = pick_file_and_folder()
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(out_dir, f"{base} prepared to tesseract.pdf")

    src = fitz.open(pdf_path)
    dst = fitz.open()

    for i, page in enumerate(src, start=1):
        # Рендер страницы в растр (цвет сохраняется)
        pix = page.get_pixmap(dpi=RENDER_DPI, alpha=False)

        # Pixmap -> PIL Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # Лёгкий даунскейл по длинной стороне
        w, h = img.size
        long_side = max(w, h)
        if long_side > TARGET_LONG_SIDE_PX:
            scale = TARGET_LONG_SIDE_PX / long_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        jpg_bytes = pil_to_jpeg_bytes(img, JPEG_QUALITY)

        # Создаём страницу того же размера, что и оригинал, и кладём картинку на весь лист
        rect = page.rect
        new_page = dst.new_page(width=rect.width, height=rect.height)
        new_page.insert_image(rect, stream=jpg_bytes)

        print(f"Page {i}/{src.page_count} OK")

    # Сохраняем
    dst.save(out_path, deflate=True)
    dst.close()
    src.close()

    print("\nГотово:", out_path)

if __name__ == "__main__":
    main()
