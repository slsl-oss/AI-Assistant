import re
from typing import List
from langchain_core.documents import Document
from utils.logger_handler import logger


def estimate_tokens(text: str) -> int:
    """中英文混合 token 估算：中文 ~1.5 token/字，英文 ~0.25 token/字"""
    zh = sum(1 for c in text if '一' <= c <= '鿿')
    return int(zh * 1.5 + (len(text) - zh) * 0.25)


def split_markdown(documents: List[Document], chunk_tokens: int = 500,
                   overlap_sections: int = 1) -> List[Document]:
    """Markdown 智能分块：多级标题层层降级 + token 控制"""
    all_chunks = []
    for doc in documents:
        text = doc.page_content
        meta = dict(doc.metadata)
        headings = _find_headings(text)
        if not headings:
            chunks = _split_by_paragraphs(text, chunk_tokens, overlap_sections, meta)
        else:
            chunks = _split_by_headings(text, headings, chunk_tokens, overlap_sections, meta)
        all_chunks.extend(chunks)
    logger.info(f"[MarkdownChunker] {len(documents)} doc → {len(all_chunks)} chunks")
    return all_chunks


def _find_headings(text: str) -> list:
    """找到所有 # 标题的位置和级别 [(level, start, end, title), ...]"""
    headings = []
    for m in re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE):
        headings.append({
            "level": len(m.group(1)), "start": m.start(),
            "title": m.group(2), "line": m.group(0)
        })
    for i, h in enumerate(headings):
        h["end"] = headings[i + 1]["start"] if i + 1 < len(headings) else len(text)
    return headings


def _split_by_headings(text: str, headings: list, chunk_tokens: int,
                       overlap_sections: int, meta: dict) -> List[Document]:
    """按标题层级递归分块"""
    chunks = []
    top_level = min(h["level"] for h in headings)
    top = [h for h in headings if h["level"] == top_level]

    for i, h in enumerate(top):
        end = top[i + 1]["start"] if i + 1 < len(top) else len(text)
        section = text[h["start"]:end]

        if estimate_tokens(section) <= chunk_tokens:
            chunks.append(Document(page_content=section.strip(),
                metadata={**meta, "heading": h["title"], "level": str(top_level)}))
        else:
            sub = [s for s in headings
                   if s["start"] >= h["start"] and s["start"] < end and s["level"] > top_level]
            if sub:
                chunks.extend(_split_by_headings(section, sub, chunk_tokens, overlap_sections,
                    {**meta, "heading": h["title"], "level": str(top_level)}))
            else:
                chunks.extend(_split_by_paragraphs(section, chunk_tokens, overlap_sections,
                    {**meta, "heading": h["title"], "level": str(top_level)}))
    return _overlap(chunks, overlap_sections)


def _split_by_paragraphs(text: str, chunk_tokens: int,
                         overlap_sections: int, meta: dict) -> List[Document]:
    """按段落拆分 + token 控制"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        return []
    chunks, current, cur_tokens = [], [], 0
    for p in paragraphs:
        pt = estimate_tokens(p)
        if cur_tokens + pt > chunk_tokens and current:
            chunks.append(Document(page_content='\n\n'.join(current), metadata=dict(meta)))
            current = current[-overlap_sections:] if overlap_sections > 0 else []
            cur_tokens = sum(estimate_tokens(x) for x in current)
        current.append(p)
        cur_tokens += pt
    if current:
        chunks.append(Document(page_content='\n\n'.join(current), metadata=dict(meta)))
    return chunks


def _overlap(chunks: List[Document], n: int) -> List[Document]:
    if n <= 0 or len(chunks) < 2:
        return chunks
    for i in range(len(chunks) - 1):
        prev = chunks[i].page_content.split('\n\n')
        chunks[i + 1].page_content = '\n\n'.join(prev[-n:]) + '\n\n' + chunks[i + 1].page_content
    return chunks
