"""统一的数据模型与源抓取逻辑。

支持两种源：
  - RSS：用 feedparser 解析
  - PubMed：调用 E-utilities esearch + efetch（免 key）
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import httpx
from dateutil import parser as dateparser

log = logging.getLogger(__name__)

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HTTP_TIMEOUT = 30.0
USER_AGENT = "ai-med-daily/0.1 (GitHub Actions)"


@dataclass
class Article:
    title: str
    url: str
    source: str
    category: str
    abstract: str = ""
    published_at: datetime | None = None
    doi: str = ""
    # 以下字段由下游模块填充
    score: int | None = None
    summary_zh: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def hash_key(self) -> str:
        """用 DOI 去重，没有 DOI 就用 URL。"""
        seed = self.doi.strip().lower() if self.doi else self.url.strip().lower()
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    def to_row(self) -> tuple[str, str, str, str]:
        dt = self.published_at.date().isoformat() if self.published_at else datetime.now(timezone.utc).date().isoformat()
        return (self.hash_key, self.url, self.title, dt)


def _within_window(dt: datetime | None, hours: int) -> bool:
    """判断文章是否在最近 hours 小时内。无时间信息时默认保留。"""
    if dt is None:
        return True
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt) <= timedelta(hours=hours)


def fetch_rss(source: dict[str, Any], window_hours: int = 48) -> list[Article]:
    """拉取 RSS 源。window_hours 控制只取多少小时以内的条目。"""
    url = source["url"]
    name = source["name"]
    category = source["category"]
    max_items = source.get("max_items", 20)

    log.info("[rss] fetching %s", name)
    parsed = feedparser.parse(url, agent=USER_AGENT)
    if parsed.bozo and not parsed.entries:
        raise RuntimeError(f"feedparser failed for {name}: {parsed.bozo_exception}")

    articles: list[Article] = []
    for entry in parsed.entries[:max_items]:
        published = None
        for field_name in ("published", "updated", "created"):
            raw = entry.get(field_name)
            if raw:
                try:
                    published = dateparser.parse(raw)
                    break
                except (ValueError, TypeError):
                    pass

        if not _within_window(published, window_hours):
            continue

        abstract = entry.get("summary", "") or entry.get("description", "")
        abstract = _strip_html(abstract)[:1500]

        articles.append(
            Article(
                title=_strip_html(entry.get("title", "")).strip(),
                url=entry.get("link", "").strip(),
                source=name,
                category=category,
                abstract=abstract,
                published_at=published,
                doi=_extract_doi(entry.get("id", "") + " " + entry.get("link", "")),
            )
        )
    log.info("[rss] %s: %d articles within window", name, len(articles))
    return articles


def fetch_pubmed(source: dict[str, Any], window_days: int = 2) -> list[Article]:
    """用 E-utilities 查 PubMed。先 esearch 拿 PMID，再 efetch 拿详情。"""
    name = source["name"]
    category = source["category"]
    query = source["query"]
    max_items = source.get("max_items", 20)

    log.info("[pubmed] fetching %s", name)
    with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}) as client:
        # esearch: 按日期倒序取最近 window_days 天
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_items,
            "sort": "date",
            "datetype": "pdat",
            "reldate": window_days,
            "retmode": "json",
        }
        r = client.get(f"{PUBMED_BASE}/esearch.fcgi", params=search_params)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            log.info("[pubmed] %s: 0 results", name)
            return []

        # 避免触发限速（3 req/s）
        time.sleep(0.4)

        # efetch: 拿摘要
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",
        }
        r = client.get(f"{PUBMED_BASE}/efetch.fcgi", params=fetch_params)
        r.raise_for_status()
        articles = _parse_pubmed_xml(r.text, source=name, category=category)
    log.info("[pubmed] %s: %d articles", name, len(articles))
    return articles


def _parse_pubmed_xml(xml_text: str, source: str, category: str) -> list[Article]:
    root = ET.fromstring(xml_text)
    out: list[Article] = []
    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""
        title_el = art.find(".//ArticleTitle")
        title = _inner_text(title_el)

        abstract_parts = [
            _inner_text(a) for a in art.findall(".//Abstract/AbstractText")
        ]
        abstract = " ".join(p for p in abstract_parts if p).strip()[:2000]

        doi = ""
        for aid in art.findall(".//ArticleId"):
            if aid.attrib.get("IdType") == "doi" and aid.text:
                doi = aid.text.strip()
                break

        # 发表日期
        published = None
        pub_date = art.find(".//PubDate")
        if pub_date is not None:
            y = pub_date.findtext("Year")
            m = pub_date.findtext("Month") or "1"
            d = pub_date.findtext("Day") or "1"
            if y:
                try:
                    published = dateparser.parse(f"{y}-{m}-{d}").replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
        if not url or not title:
            continue
        out.append(
            Article(
                title=title,
                url=url,
                source=source,
                category=category,
                abstract=abstract,
                published_at=published,
                doi=doi,
            )
        )
    return out


def _inner_text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


_HTML_TAG = re.compile(r"<[^>]+>")
_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)


def _strip_html(text: str) -> str:
    return _HTML_TAG.sub("", text or "").replace("&nbsp;", " ").strip()


def _extract_doi(text: str) -> str:
    if not text:
        return ""
    m = _DOI_PATTERN.search(text)
    return m.group(0) if m else ""


def fetch_all(sources: list[dict[str, Any]], window_hours: int = 48) -> tuple[list[Article], list[str]]:
    """遍历所有源，返回 (articles, failed_source_names)。单源失败不影响整体。"""
    articles: list[Article] = []
    failed: list[str] = []
    for src in sources:
        try:
            if src["type"] == "rss":
                articles.extend(fetch_rss(src, window_hours=window_hours))
            elif src["type"] == "pubmed":
                articles.extend(fetch_pubmed(src, window_days=max(1, window_hours // 24)))
            else:
                log.warning("unknown source type: %s", src["type"])
        except Exception as e:
            log.exception("source failed: %s -- %s", src["name"], e)
            failed.append(src["name"])
    return articles, failed
