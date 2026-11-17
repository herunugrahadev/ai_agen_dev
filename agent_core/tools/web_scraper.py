"""Custom tool: web_scraper for fetching and parsing web pages."""

from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from agent_core.config import get_settings
from agent_core.tools.base import BaseTool


class WebScraperTool(BaseTool):
    """Fetches a web page over HTTP(S) and extracts structured text, title, and headings."""

    name = "web_scraper"
    description = (
        "Scrape a public HTTP/HTTPS webpage and extract its title, meta description, "
        "headings, and clean text content."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full HTTP or HTTPS URL to scrape.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum number of characters of body text to return.",
                "default": 3000,
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        timeout_seconds: float | None = None,
        trust_env: bool = True,
    ) -> None:
        self.timeout_seconds = timeout_seconds or get_settings().scraper_timeout_seconds
        self.trust_env = trust_env

    async def execute(
        self,
        *,
        task_id: str,
        url: str,
        max_chars: int = 3000,
        **_: Any,
    ) -> dict[str, Any]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid URL '{url}': only http/https URLs are supported.")

        headers = {
            "User-Agent": "OpenMausBot/0.1 (+https://github.com/herunugrahadev/ai_agen_dev)"
        }
        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            trust_env=self.trust_env,
        ) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Strip non-content elements
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        meta_desc_tag = soup.find("meta", attrs={"name": "description"})
        meta_description = (
            meta_desc_tag.get("content", "").strip()
            if meta_desc_tag and meta_desc_tag.get("content")
            else ""
        )
        headings = [
            h.get_text(" ", strip=True)
            for h in soup.find_all(["h1", "h2", "h3"])
            if h.get_text(strip=True)
        ][:15]

        raw_text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)[:max_chars]

        return {
            "task_id": task_id,
            "url": str(response.url),
            "status_code": response.status_code,
            "title": title,
            "meta_description": meta_description,
            "headings": headings,
            "content_excerpt": clean_text,
            "content_length": len(clean_text),
        }
