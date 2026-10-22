"""Build a tiny text-only PDF that pypdf can extract, without extra libraries."""

from __future__ import annotations


def simple_pdf(pages: list[str]) -> bytes:
    """Return a one-font PDF whose pages contain the given strings.

    Characters outside Latin-1 are replaced. The helper is for tests and for
    the sample file; it is not a general PDF writer.
    """
    if not pages:
        raise ValueError("at least one page is required")

    font_id = 3
    next_id = 4
    page_ids: list[int] = []
    content_ids: list[int] = []
    for _ in pages:
        page_ids.append(next_id)
        content_ids.append(next_id + 1)
        next_id += 2

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode("ascii"),
        font_id: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for text, page_id, content_id in zip(pages, page_ids, content_ids, strict=True):
        escaped = (
            text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("\n", " ")
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1", errors="replace")
        content = stream + b"\n"
        objects[content_id] = (
            f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"endstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("ascii")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    max_id = next_id - 1
    for object_id in range(1, max_id + 1):
        offsets.append(len(out))
        out.extend(f"{object_id} 0 obj\n".encode("ascii"))
        out.extend(objects[object_id])
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {max_id + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for object_id in range(1, max_id + 1):
        out.extend(f"{offsets[object_id]:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(out)
