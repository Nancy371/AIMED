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
from urllib.parse import parse_qs, urlparse
import os

import feedparser
import httpx
from dateutil import parser as dateparser
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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
    score: int | None = None  # final_score = relevance * 0.6 + practice_impact * 0.4
    relevance: int | None = None     # 学术相关性 0-10
    practice_impact: int | None = None  # 改变现有实践的可能性 0-10
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


def fetch_youtube_playlist(source: dict[str, Any], window_hours: int = 168) -> list[Article]:
    """
    从 YouTube 播放列表抓取视频，用 YouTube Data API v3 获取字幕作为 abstract。
    window_hours 默认 168（7 天），因为播客更新频率低。
    需要环境变量 YOUTUBE_API_KEY。
    """
    url = source["url"]
    name = source["name"]
    category = source["category"]
    max_items = source.get("max_items", 5)
    skip_scoring = source.get("skip_scoring", False)

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        log.warning("[youtube] YOUTUBE_API_KEY not set, skipping %s", name)
        return []

    log.info("[youtube] fetching %s", name)

    # 从播放列表 URL 提取 playlist ID
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    playlist_id = query.get("list", [None])[0]
    if not playlist_id:
        raise ValueError(f"Invalid YouTube playlist URL: {url}")

    # 用 YouTube Data API v3 的 RSS feed 获取播放列表视频（免费，不消耗 quota）
    rss_url = f"https://www.youtube.com/feeds/videos.xml?playlist_id={playlist_id}"
    parsed_feed = feedparser.parse(rss_url, agent=USER_AGENT)
    if parsed_feed.bozo and parsed_feed.bozo_exception:
        raise RuntimeError(f"feedparser failed for {name}: {parsed_feed.bozo_exception}")

    entries = parsed_feed.entries[:max_items]
    articles = []

    # 初始化 YouTube API 客户端
    youtube = build('youtube', 'v3', developerKey=api_key)

    for entry in entries:
        # 解析发布时间
        pub_dt = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

        # 检查时间窗口
        if not _within_window(pub_dt, window_hours):
            continue

        # 提取 video ID
        video_url = entry.get("link", "")
        video_id = video_url.split("v=")[-1].split("&")[0] if "v=" in video_url else ""

        if not video_id:
            log.warning("[youtube] skipping entry without video ID: %s", entry.get("title"))
            continue

        # 获取字幕
        transcript_text = ""
        try:
            # 1. 列出可用的字幕
            captions_response = youtube.captions().list(
                part='snippet',
                videoId=video_id
            ).execute()

            caption_id = None
            # 优先选择英文自动生成字幕，其次是英文手动字幕
            for item in captions_response.get('items', []):
                snippet = item['snippet']
                if snippet['language'] in ['en', 'zh', 'zh-Hans']:
                    caption_id = item['id']
                    if snippet.get('trackKind') == 'asr':  # 自动生成优先
                        break

            # 2. 下载字幕
            if caption_id:
                caption_content = youtube.captions().download(
                    id=caption_id,
                    tfmt='srt'  # SubRip 格式
                ).execute()

                # 解析 SRT 格式，提取纯文本
                transcript_text = _parse_srt(caption_content)
            else:
                log.warning("[youtube] no captions available for %s (%s)", entry.get("title"), video_id)
                transcript_text = _strip_html(entry.get("summary", ""))

        except HttpError as e:
            if e.resp.status == 403:
                log.warning("[youtube] captions disabled for %s (%s)", entry.get("title"), video_id)
            else:
                log.warning("[youtube] API error for %s: %s", video_id, e)
            transcript_text = _strip_html(entry.get("summary", ""))
        except Exception as e:
            log.warning("[youtube] caption fetch failed for %s: %s", video_id, e)
            transcript_text = _strip_html(entry.get("summary", ""))

        article = Article(
            title=entry.get("title", "Untitled"),
            url=video_url,
            source=name,
            category=category,
            abstract=transcript_text[:8000],  # 限制长度，避免 token 爆炸
            published_at=pub_dt,
        )
        article.skip_scoring = skip_scoring
        articles.append(article)

    log.info("[youtube] %s: %d videos within window", name, len(articles))
    return articles


def _parse_srt(srt_content: str) -> str:
    """从 SRT 字幕格式提取纯文本。"""
    lines = srt_content.split('\n')
    text_lines = []
    for line in lines:
        line = line.strip()
        # 跳过序号行、时间戳行、空行
        if not line or line.isdigit() or '-->' in line:
            continue
        text_lines.append(line)
    return ' '.join(text_lines)


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
            elif src["type"] == "youtube_playlist":
                articles.extend(fetch_youtube_playlist(src, window_hours=window_hours))
            else:
                log.warning("unknown source type: %s", src["type"])
        except Exception as e:
            log.exception("source failed: %s -- %s", src["name"], e)
            failed.append(src["name"])
    return articles, failed
