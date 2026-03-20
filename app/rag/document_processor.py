import io, re, uuid
from pathlib import Path

import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.models.schemas import AllowedFileType, DocumentChunk

settings = get_settings()


def _ocr_page(page: pypdf.PageObject) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        writer = pypdf.PdfWriter()
        writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        buf.seek(0)
        images = convert_from_bytes(buf.read(), dpi=250)
        return "\n".join(pytesseract.image_to_string(img, lang="eng") for img in images).strip()
    except Exception:
        return ""


class DocumentProcessor:
    _SPLITTER = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    @staticmethod
    def validate_extension(filename: str) -> AllowedFileType:
        ext = Path(filename).suffix.lstrip(".").lower()
        try:
            return AllowedFileType(ext)
        except ValueError:
            raise ValueError(f"Unsupported type '.{ext}'. Allowed: {[e.value for e in AllowedFileType]}")

    def process_txt(self, file_bytes: bytes, filename: str) -> tuple[str, list[DocumentChunk]]:
        text = self._decode_txt(file_bytes, filename)
        doc_id = str(uuid.uuid4())
        return doc_id, self._chunk(text, doc_id, filename)

    def iter_pdf_pages(self, file_bytes: bytes, filename: str):
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        doc_id = str(uuid.uuid4())
        offset = 0
        for page_num, page in enumerate(reader.pages, start=1):
            used_ocr = False
            text = self._native_extract(page)
            if not text:
                used_ocr = True
                text = _ocr_page(page)
            chunks = []
            if text:
                prefixed = f"[Page {page_num}/{total_pages}]\n{text}"
                chunks = self._chunk(prefixed, doc_id, filename, page_num=page_num, offset=offset)
                offset += len(chunks)
            yield {"page_num": page_num, "total_pages": total_pages,
                   "doc_id": doc_id, "chunks": chunks,
                   "used_ocr": used_ocr, "text_found": bool(text)}

    @staticmethod
    def _native_extract(page: pypdf.PageObject) -> str:
        for mode in ("layout", None):
            try:
                kwargs = {"extraction_mode": mode} if mode else {}
                raw = page.extract_text(**kwargs) or ""
                if raw.strip():
                    raw = re.sub(r"\n{3,}", "\n\n", raw)
                    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
                    return raw.strip()
            except Exception:
                continue
        return ""

    @staticmethod
    def _decode_txt(file_bytes: bytes, filename: str) -> str:
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return file_bytes.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        raise ValueError(f"Cannot decode '{filename}'.")

    def _chunk(self, text, doc_id, filename, page_num=None, offset=0):
        result = []
        for i, content in enumerate(self._SPLITTER.split_text(text)):
            meta = {"document_id": doc_id, "filename": filename, "chunk_index": offset + i}
            if page_num is not None:
                meta["page_number"] = page_num
            result.append(DocumentChunk(
                chunk_id=f"{doc_id}_{offset+i}", document_id=doc_id,
                content=content, metadata=meta,
            ))
        return result
