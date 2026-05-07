"""飞书自定义机器人 webhook 推送。

使用交互式卡片（msg_type=interactive）格式，按 category 分组展示。
如果群机器人开启了「签名校验」，设置环境变量 FEISHU_SIGN_SECRET 即可自动加签。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from .fetch import Article

log = logging.getLogger(__name__)

CATEGORY_NAMES = {
    "drug_discovery": "🧬 药物发现与研发",
    "clinical_llm": "🏥 临床决策与大模型",
}

HTTP_TIMEOUT = 15.0


def push_digest(
    webhook_url: str,
    articles: list[Article],
    sign_secret: str | None = None,
    failed_sources: list[str] | None = None,
) -> None:
    """按 category 分组生成卡片并推送。"""
    if not articles and not failed_sources:
        log.info("no articles to push, skipping")
        return

    card = _build_card(articles, failed_sources or [])
    payload: dict[str, Any] = {"msg_type": "interactive", "card": card}

    if sign_secret:
        ts = str(int(time.time()))
        payload["timestamp"] = ts
        payload["sign"] = _gen_sign(ts, sign_secret)

    with httpx.Client(timeout=HTTP_TIMEOUT) as client:
        r = client.post(webhook_url, json=payload)
        r.raise_for_status()
        data = r.json()
        if data.get("code", 0) != 0 and data.get("StatusCode", 0) != 0:
            # 飞书返回 code != 0 表示业务错误
            raise RuntimeError(f"feishu rejected: {data}")
        log.info("feishu push ok: %s", data)


def _gen_sign(timestamp: str, secret: str) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    h = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return base64.b64encode(h).decode("utf-8")


def _build_card(articles: list[Article], failed_sources: list[str]) -> dict[str, Any]:
    today = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    elements: list[dict[str, Any]] = []

    grouped: dict[str, list[Article]] = {}
    for a in articles:
        grouped.setdefault(a.category, []).append(a)

    for cat, items in grouped.items():
        items.sort(key=lambda x: x.score or 0, reverse=True)
        elements.append(
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**{CATEGORY_NAMES.get(cat, cat)}** · {len(items)} 条",
                },
            }
        )
        for a in items:
            elements.append({"tag": "hr"})
            elements.append(_article_element(a))

    if failed_sources:
        elements.append({"tag": "hr"})
        elements.append(
            {
                "tag": "note",
                "elements": [
                    {
                        "tag": "plain_text",
                        "content": f"⚠️ 以下源抓取失败：{', '.join(failed_sources)}",
                    }
                ],
            }
        )

    if not articles:
        elements.insert(
            0,
            {
                "tag": "div",
                "text": {"tag": "lark_md", "content": "_今日无符合阈值的新内容_"},
            },
        )

    return {
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": f"🤖 AI 医疗情报 · {today}"},
        },
        "elements": elements,
    }


def _article_element(a: Article) -> dict[str, Any]:
    title_esc = _escape_md(a.title)
    url_esc = a.url.replace(")", "%29")
    tag_str = " ".join(f"`{t}`" for t in a.tags) if a.tags else ""
    score = a.score if a.score is not None else "-"

    lines = [
        f"**[{title_esc}]({url_esc})**",
        f"⭐ {score}/10 · {_escape_md(a.source)}  {tag_str}",
    ]
    if a.summary_zh:
        lines.append(_escape_md(a.summary_zh))
    return {"tag": "div", "text": {"tag": "lark_md", "content": "\n".join(lines)}}


def _escape_md(text: str) -> str:
    return text.replace("[", "【").replace("]", "】")


def load_config() -> tuple[str, str | None]:
    """从环境变量读 webhook URL 和可选 sign secret。"""
    url = os.getenv("FEISHU_WEBHOOK_URL")
    if not url:
        raise RuntimeError("FEISHU_WEBHOOK_URL env var is required")
    secret = os.getenv("FEISHU_SIGN_SECRET") or None
    return url, secret
