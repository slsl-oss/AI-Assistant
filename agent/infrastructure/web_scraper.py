"""
网页抓取器：BS4 解析 + WorkerPool 并发控制 + 全局频率限制
"""
import asyncio
import time
import re
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup
from utils.logger_handler import logger


class GlobalRateLimiter:
    """全局频率限制器，控制每秒请求数"""

    def __init__(self, requests_per_second: float = 10.0):
        self.min_interval = 1.0 / requests_per_second
        self._last_request = 0.0

    async def wait(self):
        """等待直到可以发起下一次请求"""
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self._last_request = time.monotonic()


# 全局实例
global_limiter = GlobalRateLimiter(requests_per_second=10.0)


class WorkerPool:
    """用信号量控制同时抓取的数量"""

    def __init__(self, max_workers: int = 15):
        self.semaphore = asyncio.Semaphore(max_workers)

    async def throttle(self):
        """获取抓取许可，超限则排队等待"""
        async with self.semaphore:
            await global_limiter.wait()
            yield


class WebScraper:
    """网页抓取器：BS4 解析，提取正文"""

    def __init__(self, max_workers: int = 15, timeout: int = 15):
        self.pool = WorkerPool(max_workers=max_workers)
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def scrape_many(self, urls: list[str]) -> list[dict]:
        """并行抓取多个 URL，返回结构化数据"""
        if not urls:
            return []

        session = await self._get_session()
        tasks = [self._scrape_one(url, session) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"[WebScraper] 抓取异常: {r}")
            elif r is not None:
                docs.append(r)

        logger.info(f"[WebScraper] {len(docs)}/{len(urls)} 抓取成功")
        return docs

    async def _scrape_one(self, url: str, session: aiohttp.ClientSession) -> Optional[dict]:
        """抓取单个 URL（受 WorkerPool 并发控制）"""
        async for _ in self.pool.throttle():
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.warning(f"[WebScraper] HTTP {resp.status}: {url}")
                        return None

                    html = await resp.text()
                    text, title = self._extract_text(html, url)

                    if not text or len(text) < 100:
                        logger.warning(f"[WebScraper] 内容过短: {url}")
                        return None

                    return {
                        "url": url,
                        "title": title or url,
                        "raw_content": text,
                    }
            except asyncio.TimeoutError:
                logger.warning(f"[WebScraper] 超时: {url}")
                return None
            except Exception as e:
                logger.warning(f"[WebScraper] 失败: {url} → {e}")
                return None

    def _extract_text(self, html: str, url: str = "") -> tuple[str, str]:
        """BS4 解析 HTML，提取正文和标题"""
        soup = BeautifulSoup(html, "lxml")

        # 提取标题
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # 移除无用标签
        for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                                   "noscript", "iframe", "form", "button"]):
            tag.decompose()

        # 优先从 article/main 提取
        body = soup.find("article") or soup.find("main") or soup.find("body")
        if not body:
            return "", title

        text = body.get_text(separator="\n", strip=True)
        # 合并多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)

        return text, title