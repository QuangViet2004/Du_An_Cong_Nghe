import re
from typing import List, Optional

from rag.types import Chunk

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.!?。！？])\s+")

def split_into_sentences(block: str) -> List[str]:
    parts = _SENTENCE_SPLIT_RE.split(block.strip())
    return [p.strip() for p in parts if p and p.strip()]

def semantic_chunk_blocks(
    chunks: List[Chunk],
    target_chars: int = 900,
    overlap_chars: int = 150,
) -> List[Chunk]:
    out: List[Chunk] = []
    for c in chunks:
        if c.modality != "text" or not c.text:
            out.append(c)
            continue
        sentences = split_into_sentences(c.text)
        buf = ""
        i = 0
        while i < len(sentences):
            s = sentences[i]
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= target_chars:
                buf += " " + s
            else:
                out.append(Chunk(text=buf, page=c.page, source=c.source, heading=c.heading, modality="text"))
                if overlap_chars > 0 and len(buf) > overlap_chars:
                    tail = buf[-overlap_chars:]
                    tail = tail.lstrip()
                    buf = tail + " " + s
                else:
                    buf = s
            i += 1
        if buf:
            out.append(Chunk(text=buf, page=c.page, source=c.source, heading=c.heading, modality="text"))
    return out

def section_chunk_blocks(
    chunks: List[Chunk],
    target_chars: int = 1200,
    tag_keywords: Optional[List[str]] = None,
) -> List[Chunk]:
    out: List[Chunk] = []
    kw = [k.lower() for k in (tag_keywords or [])]
    for c in chunks:
        if c.modality != "text" or not c.text:
            out.append(c)
            continue
        lines = [ln.strip() for ln in c.text.splitlines() if ln and ln.strip()]
        cur_title = None
        buf = []
        for ln in lines:
            is_title = False
            if len(ln) <= 140:
                if ln.endswith(":"):
                    is_title = True
                elif ln.isupper() and len(ln.split()) <= 12:
                    is_title = True
                elif re.match(r"^(\d+(?:\.|\))\s+.+)$", ln):
                    is_title = True
            if is_title:
                if buf:
                    txt = " ".join(buf)
                    if txt:
                        tags = None
                        if kw:
                            low = txt.lower()
                            tags = [k for k in kw if k in low]
                        out.append(Chunk(text=txt[:target_chars], page=c.page, source=c.source, heading=cur_title, modality="text", tags=tags))
                    buf = []
                cur_title = ln.rstrip(":")
            else:
                buf.append(ln)
        if buf:
            txt = " ".join(buf)
            if txt:
                tags = None
                if kw:
                    low = txt.lower()
                    tags = [k for k in kw if k in low]
                out.append(Chunk(text=txt[:target_chars], page=c.page, source=c.source, heading=cur_title, modality="text", tags=tags))
    return out

def semantic_chunk_blocks_docling(
    chunks: List[Chunk],
    encode_sentences,  
    target_chars: int = 900,
    overlap_chars: int = 150,
    sim_drop_threshold: float = 0.35,
) -> List[Chunk]:
    out: List[Chunk] = []
    for c in chunks:
        if c.modality != "text" or not c.text:
            out.append(c)
            continue
        sentences = split_into_sentences(c.text)
        if not sentences:
            continue
        embs = encode_sentences(sentences)
        buf_sent = [sentences[0]]
        i = 1
        while i < len(sentences):
            prev_emb = embs[i-1]
            cur_emb = embs[i]
            sim = float(prev_emb @ cur_emb)
            next_s = sentences[i]
            cur_text = " ".join(buf_sent)
            if (len(cur_text) + 1 + len(next_s) <= target_chars) and (sim >= sim_drop_threshold):
                buf_sent.append(next_s)
            else:
                text = " ".join(buf_sent)
                out.append(Chunk(text=text, page=c.page, source=c.source, heading=c.heading, modality="text"))
                if overlap_chars > 0 and len(text) > overlap_chars:
                    tail = text[-overlap_chars:].lstrip()
                    buf = tail
                    buf_sent = [buf, next_s] if buf else [next_s]
                else:
                    buf_sent = [next_s]
            i += 1
        if buf_sent:
            out.append(Chunk(text=" ".join(buf_sent), page=c.page, source=c.source, heading=c.heading, modality="text"))
    return out
