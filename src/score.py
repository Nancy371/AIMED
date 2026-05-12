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

你的任务：对输入的英文论文/博客条目按两个维度独立打分，并生成结构化中文摘要。

**维度 1：学术相关性 relevance（0-10 分）**
- 10：顶刊突破性成果 / 知名团队（DeepMind、Isomorphic、Google Health 等）重要进展
- 7-9：高质量研究或有实质方法创新
- 4-6：相关但偏综述、重复性工作或初步研究
- 0-3：只是边缘相关或纯工程细节

**维度 2：实践影响力 practice_impact（0-10 分）—— 该信息改变现有实践的可能性**
- 9-10：立即可改变实践 — FDA/NMPA/EMA 批准、新临床指南、即将商业化的工具、可直接落地的开源 SOTA
- 6-8：中期可能改变 — 大型 RCT 阳性结果、被多家机构验证的方法、有明确临床转化路径
- 3-5：概念验证 — 单中心研究、新方法但需更多验证、性能提升但缺真实场景测试
- 0-2：纯探索/理论 — 综述、benchmark、技术报告无下游应用、纯方法学改进

**摘要格式要求（200 字以内中文）：**

1. **核心观点**（1-2 句话总结主要发现或论点）

2. **PICO 框架**（如适用于临床研究，否则跳过）：
   - P（患者/人群）：研究对象特征
   - I（干预）：实验组接受的处理
   - C（对照）：对照组设置
   - O（结局）：主要终点指标与结果

3. **商业化探索**（2-3 个问题，引导思考产业化路径）：
   - 例如："该技术的监管路径是什么？"
   - "哪些现有产品线可以整合这一方法？"
   - "规模化部署的主要技术瓶颈在哪？"

**输出要求：** 严格返回 JSON 数组，无任何其他文字、无 markdown 代码块。
格式：[{"index": 0, "relevance": 8, "practice_impact": 6, "summary_zh": "...", "tags": ["药物发现"]}, ...]

tags 从这些选：药物发现、分子生成、蛋白结构、临床LLM、医学问答、诊断辅助、监管政策、影像、基础模型、评测基准、其他"""


# final_score 权重：相关性 60% + 实践影响力 40%
RELEVANCE_WEIGHT = 0.6
PRACTICE_WEIGHT = 0.4


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

        # 解析两个维度的分数
        try:
            relevance = int(r.get("relevance", 0))
        except (ValueError, TypeError):
            relevance = 0
        try:
            practice_impact = int(r.get("practice_impact", 0))
        except (ValueError, TypeError):
            practice_impact = 0

        # 计算加权 final_score（四舍五入到整数）
        final_score = relevance * RELEVANCE_WEIGHT + practice_impact * PRACTICE_WEIGHT

        article.relevance = relevance
        article.practice_impact = practice_impact
        article.score = round(final_score)
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
