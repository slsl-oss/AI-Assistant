"""Three read-only tool contracts. No HTTP, database or model is loaded here.

Backends are explicitly injected and bound to the authenticated user's scope by
the host, never by model arguments. The bundled backend is synthetic/offline.
"""
import ipaddress
from typing import Protocol
from urllib.parse import urlsplit


TOOL_SCHEMAS = [
    {"name": "search_web", "description": "搜索公开资料的摘要；摘要不等于已读取原文。",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"], "additionalProperties": False}},
    {"name": "search_knowledge_base", "description": "检索当前用户有权读取的内部知识片段。",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                    "required": ["query"], "additionalProperties": False}},
    {"name": "fetch_page", "description": "读取本轮搜索已发现的公开网页正文，以核实摘要与冲突。",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}},
                    "required": ["url"], "additionalProperties": False}},
]


class ToolRejected(ValueError):
    pass


def public_url(url: str) -> bool:
    """Syntactic boundary only; a future HTTP backend MUST also validate DNS/redirects."""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").rstrip(".").lower()
        if (parts.scheme not in {"http", "https"} or not host or
                parts.username is not None or parts.password is not None or
                parts.port not in {None, 80, 443} or parts.fragment or
                any(c.isspace() or ord(c) < 32 for c in url) or "\\" in url):
            return False
        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            return ("." in host and not host.endswith((".localhost", ".local", ".internal"))
                    and not host.replace(".", "").isdigit()
                    and not host.startswith("0x"))
    except (ValueError, TypeError):
        return False


class ResearchBackend(Protocol):
    """Cancellation-cooperative async adapters; return raw evidence, not an LLM summary.

    Each record contains title, text, and optional url/published_at. A production
    adapter must enforce tenant ACLs, retrieval timestamps, DNS/redirect checks,
    response-size limits and network timeouts itself. Do not wrap blocking work
    in an async function and assume asyncio can forcibly stop it.
    """
    async def search_web(self, query: str) -> list[dict]: ...
    async def search_knowledge_base(self, query: str) -> list[dict]: ...
    async def fetch_page(self, url: str) -> list[dict]: ...


class ResearchTools:
    def __init__(self, backend: ResearchBackend):
        self.backend = backend

    def validate(self, name: str, arguments: dict, discovered_urls: set[str]) -> None:
        if not isinstance(name, str) or name not in {s["name"] for s in TOOL_SCHEMAS}:
            raise ToolRejected("tool_not_allowed")
        key = "url" if name == "fetch_page" else "query"
        if type(arguments) is not dict or set(arguments) != {key}:
            raise ToolRejected("invalid_arguments")
        value = arguments[key]
        if not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ToolRejected("invalid_arguments")
        if key == "url" and (not public_url(value) or value not in discovered_urls):
            raise ToolRejected("url_not_discovered_or_not_public")

    async def execute(self, name: str, arguments: dict, discovered_urls: set[str]) -> list[dict]:
        self.validate(name, arguments, discovered_urls)
        # Never getattr/eval an arbitrary model-supplied name.
        handlers = {"search_web": self.backend.search_web,
                    "search_knowledge_base": self.backend.search_knowledge_base,
                    "fetch_page": self.backend.fetch_page}
        records = await handlers[name](**arguments)
        if not isinstance(records, list) or len(records) > 5:
            raise ToolRejected("invalid_tool_result")
        kind = {"search_web": "snippet", "search_knowledge_base": "knowledge",
                "fetch_page": "page"}[name]
        clean = []
        for record in records:
            if not isinstance(record, dict):
                raise ToolRejected("invalid_tool_result")
            title, content = record.get("title"), record.get("text")
            url, date = record.get("url", ""), record.get("published_at", "")
            if (not isinstance(title, str) or not isinstance(content, str) or
                    not isinstance(url, str) or not isinstance(date, str) or
                    len(title) > 300 or len(url) > 2000 or len(date) > 40):
                raise ToolRejected("invalid_tool_result")
            if not content.strip():
                continue
            if kind != "knowledge" and not public_url(url):
                raise ToolRejected("invalid_tool_result_url")
            if kind == "page" and url != arguments["url"]:
                raise ToolRejected("unexpected_page_url")
            clean.append({"kind": kind, "title": title, "text": content[:4000],
                          "url": url, "published_at": date,
                          "truncated": len(content) > 4000})
        return clean
