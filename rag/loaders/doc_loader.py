import os
import tempfile
from typing import List, Optional

try:
    from docling.document_converter import DocumentConverter
    from docling.document_converter import PdfFormatOption  
    from docling.datamodel.base_models import InputFormat  
    from docling.datamodel.pipeline_options import PdfPipelineOptions  
except Exception:
    DocumentConverter = None  
    PdfFormatOption = None  
    InputFormat = None  
    PdfPipelineOptions = None  
import fitz  

from rag.types import Chunk

def _fitz_fallback(pdf_path: str, image_out_dir: Optional[str] = None) -> List[Chunk]:
    doc = fitz.open(pdf_path)
    chunks: List[Chunk] = []
    img_out_dir = image_out_dir or os.path.join(tempfile.gettempdir(), "docling_images")
    os.makedirs(img_out_dir, exist_ok=True)
    for i, page in enumerate(doc, start=1):
        text = (page.get_text("text") or "").strip()
        if text:
            chunks.append(Chunk(text=text, page=i, source=os.path.basename(pdf_path), heading=None, modality="text"))
        try:
            image_list = page.get_images(full=True) or []
            for j, imginfo in enumerate(image_list):
                xref = imginfo[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_p{i}_{j}.png"
                img_path = os.path.join(img_out_dir, img_filename)
                pix.save(img_path)
                chunks.append(Chunk(text="", page=i, source=os.path.basename(pdf_path), heading=None, modality="image", image_path=img_path))
                pix = None  
        except Exception:
            continue
    return chunks

def load_pdf_with_docling(pdf_path: str, image_out_dir: Optional[str] = None) -> List[Chunk]:

    if DocumentConverter is None:
        return _fitz_fallback(pdf_path, image_out_dir=image_out_dir)
    try:
        if PdfFormatOption is not None and PdfPipelineOptions is not None and InputFormat is not None:
            opts = PdfPipelineOptions()

            opts.generate_page_images = False
            opts.generate_picture_images = True
            try:

                opts.images_scale = 2.0
                opts.do_ocr = True
            except Exception:
                pass
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
                }
            )
        else:
            converter = DocumentConverter()
        result = converter.convert(pdf_path)
    except Exception:
        return _fitz_fallback(pdf_path, image_out_dir=image_out_dir)

    chunks: List[Chunk] = []

    img_out_dir = image_out_dir or os.path.join(tempfile.gettempdir(), "docling_images")
    os.makedirs(img_out_dir, exist_ok=True)

    pages_obj = getattr(result.document, "pages", None)
    if isinstance(pages_obj, dict):
        page_iter = [(int(k), v) for k, v in pages_obj.items()]
        page_iter.sort(key=lambda x: x[0])
    else:
        page_iter = list(enumerate(pages_obj or [], start=1))

    for page_idx, page in page_iter:
        page_num = page_idx

        page_text_parts: List[str] = []

        for block in getattr(page, "blocks", []) or []:
            lines = []
            for line in getattr(block, "lines", []) or []:
                spans = [span.text for span in getattr(line, "spans", []) if getattr(span, "text", None)]
                if spans:
                    lines.append("".join(spans))
            if lines:
                page_text_parts.append(" ".join(lines))
            else:

                blk_text = getattr(block, "text", None)
                if isinstance(blk_text, str) and blk_text.strip():
                    page_text_parts.append(blk_text.strip())

        if not page_text_parts:
            for tb in getattr(page, "text_blocks", []) or []:
                t = getattr(tb, "text", None)
                if isinstance(t, str) and t.strip():
                    page_text_parts.append(t.strip())

        full_text = "\n".join([p.strip() for p in page_text_parts if isinstance(p, str) and p.strip()])
        if full_text:
            chunks.append(Chunk(text=full_text, page=page_num, source=os.path.basename(pdf_path), heading=None, modality="text"))

    try:
        pictures = getattr(result.document, "pictures", []) or []
        for i, pic in enumerate(pictures):
            img_obj = getattr(pic, "image", None)
            if img_obj is None:
                continue
            pil_img = getattr(img_obj, "pil_image", None)
            if pil_img is None and hasattr(img_obj, "to_pillow"):
                try:
                    pil_img = img_obj.to_pillow()
                except Exception:
                    pil_img = None
            if pil_img is None:
                continue
            img_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_figure_{i}.png"
            img_path = os.path.join(img_out_dir, img_filename)
            try:
                pil_img.save(img_path)

                pno = getattr(pic, "page_no", None)
                if pno is None:
                    pno = getattr(pic, "page", None)
                if pno is None:
                    pno = 1
                chunks.append(Chunk(text="", page=int(pno), source=os.path.basename(pdf_path), heading=None, modality="image", image_path=img_path))
            except Exception:
                continue
    except Exception:
        pass

    if not any(c.modality == "text" and c.text for c in chunks):
        try:

            md_text = None
            for attr in ("export_to_markdown", "to_markdown", "export_markdown"):
                fn = getattr(result.document, attr, None)
                if callable(fn):
                    md_text = fn()
                    break

            if md_text is None:
                exporter = getattr(result.document, "markdown", None)
                if callable(exporter):
                    md_text = exporter()
            if isinstance(md_text, str) and md_text.strip():
                chunks.append(Chunk(text=md_text.strip(), page=1, source=os.path.basename(pdf_path), heading=None, modality="text"))
        except Exception:

            return _fitz_fallback(pdf_path, image_out_dir=image_out_dir)

    return chunks
