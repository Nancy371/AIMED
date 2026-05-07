"""对文章做中文摘要 + 相关性打分。

LLM 后端通过环境变量选择（见 src/llm.py）：
  - LLM_PROVIDER=anthropic（默认） → 用 Claude Haiku，启用 prompt caching
  - LLM_PROVIDER=openai            → OpenAI/DeepSeek/Kimi/Qwen/GLM 等

批量调用以降低成本，单次处理 batch_size 篇。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .fetch import Article
from .llm import LLMClient, get_client

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一位 AI 医疗领域的资深分析师，专注于药物发现与研发、临床决策与大模型这两个子领域。

你的任务：对输入的英文论文/博客条目做两件事
1. 相关性评分（0-10 分整数）
   - 10：顶刊突破性成果 / 知名团队重要进展
   - 7-9：高质量研究或有实质方法创新
   - 4-6：相关但偏综述、重复性工作或初步研究
   - 0-3：只是边缘相关或纯工程细节
2. 生成 80 字以内的中文摘要，突出：研究对象、方法要点、主要结论

输出要求：严格返回 JSON 数组，不要任何其他文字、不要 markdown 代码块。
格式：[{"index": 0, "score": 8, "summary_zh": "...", "tags": ["药物发现"]}, ...]

tags 从这些选：药物发现、分子生成、蛋白结构、临床LLM、医学问答、诊断辅助、监管政策、影像、基础模型、评测基准、其他"""


def score_articles(articles: list[Article], batch_size: int = 10) -> list[Article]:
    """对 articles 列表打分 + 摘要，直接修改对象并返回。"""
    if not articles:
        return []

    client = get_client()
    scored: list[Article] = []

    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        log.info("scoring batch %d-%d / %d", i, i + len(batch), len(articles))
        try:
            _score_batch(client, batch)
            scored.extend(batch)
        except Exception:
            log.exception("batch scoring failed, skipping %d articles", len(batch))
    return scored


def _score_batch(client: LLMClient, batch: list[Article]) -> None:
    user_payload = [
        {
            "index": idx,
            "title": a.title,
            "source": a.source,
            "abstract": a.abstract[:1200],
        }
        for idx, a in enumerate(batch)
    ]

    user_msg = (
        f"请对以下 {len(batch)} 条目打分并生成中文摘要：\n\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
    )
    text = client.complete(system=SYSTEM_PROMPT, user=user_msg, max_tokens=2000)

    results = _parse_json_array(text)
    if not results:
        log.warning("could not parse scoring response: %s", text[:500])
        return

    by_index = {r.get("index"): r for r in results if isinstance(r, dict)}
    for idx, article in enumerate(batch):
        r = by_index.get(idx)
        if not r:
            continue
        try:
            article.score = int(r.get("score", 0))
        except (ValueError, TypeError):
            article.score = 0
        article.summary_zh = str(r.get("summary_zh", "")).strip()
        tags = r.get("tags", [])
        if isinstance(tags, list):
            article.tags = [str(t) for t in tags]


_JSON_ARRAY_PATTERN = re.compile(r"\[\s*\{.*\}\s*\]", re.DOTALL)


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    """从模型输出中提取 JSON 数组。容忍前后的多余文字或 markdown 围栏。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass

    m = _JSON_ARRAY_PATTERN.search(text)
    if m:
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            pass
    return []


def filter_by_score(articles: list[Article], threshold: int) -> list[Article]:
    return [a for a in articles if (a.score or 0) >= threshold]
