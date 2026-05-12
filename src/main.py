"""主编排：fetch → dedupe → score → push。

用法：
    python -m src.main                     # 完整流程
    python -m src.main --dry-run           # 不推送飞书，只打印结果
    python -m src.main --skip-score        # 跳过打分（测试 fetch/dedupe）
    python -m src.main --limit 5           # 只处理前 N 篇（调试用）
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .dedupe import SeenStore
from .feishu import load_config as load_feishu, push_digest
from .fetch import fetch_all
from .score import filter_by_score, remix_podcasts, score_articles

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "sources.yaml"
DB_PATH = ROOT / "data" / "seen.db"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="AI 医疗每日情报 agent")
    parser.add_argument("--dry-run", action="store_true", help="不推送飞书、不更新 seen.db")
    parser.add_argument("--skip-score", action="store_true", help="跳过 LLM 打分（测试用）")
    parser.add_argument("--limit", type=int, help="只处理前 N 篇新文章")
    parser.add_argument("--window-hours", type=int, default=48, help="拉取最近 N 小时的条目")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("main")
    log.info("starting ai-med-daily (dry_run=%s)", args.dry_run)

    config = load_config()
    sources = config["sources"]
    threshold = config.get("score_threshold", 6)
    batch_size = config.get("batch_size", 10)

    # 1. Fetch
    log.info("=== stage 1: fetch (%d sources) ===", len(sources))
    articles, failed = fetch_all(sources, window_hours=args.window_hours)
    log.info("fetched %d articles (%d sources failed)", len(articles), len(failed))

    # 2. Dedupe
    log.info("=== stage 2: dedupe ===")
    store = SeenStore(DB_PATH)
    try:
        fresh = store.filter_new(articles)

        if args.limit:
            fresh = fresh[: args.limit]
            log.info("limited to %d articles for debugging", len(fresh))

        # 3. Score
        if args.skip_score:
            log.info("=== stage 3: score SKIPPED ===")
            kept = fresh
        else:
            log.info("=== stage 3: score & summarize ===")
            # 分离需要打分的和跳过打分的文章
            to_score = [a for a in fresh if not getattr(a, 'skip_scoring', False)]
            no_score = [a for a in fresh if getattr(a, 'skip_scoring', False)]

            # 对需要打分的文章进行评分和过滤
            scored = score_articles(to_score, batch_size=batch_size) if to_score else []
            kept_scored = filter_by_score(scored, threshold)

            # 对跳过打分的文章（播客等）生成混音简报
            no_score = remix_podcasts(no_score) if no_score else []

            # 合并：打分通过的 + 跳过打分的（全部保留）
            kept = kept_scored + no_score

            log.info(
                "scored %d, kept %d above threshold %d; %d articles skipped scoring (all kept)",
                len(to_score), len(kept_scored), threshold, len(no_score),
            )

        # 4. Push
        log.info("=== stage 4: push ===")
        if args.dry_run:
            _print_preview(kept, failed)
        else:
            if kept or failed:
                webhook_url, sign_secret = load_feishu()
                push_digest(webhook_url, kept, sign_secret=sign_secret, failed_sources=failed)
            else:
                log.info("nothing to push (no kept articles and no failures)")

        # 5. Mark seen (only on non-dry-run successful push)
        if not args.dry_run:
            store.mark_seen(fresh)
            log.info("dedupe db now has %d total entries", store.total())
    finally:
        store.close()

    log.info("done")
    return 0


def _print_preview(articles, failed_sources):
    print("\n" + "=" * 60)
    print(f"DRY RUN preview: {len(articles)} articles")
    print("=" * 60)
    for a in articles:
        print(f"\n[{a.score}/10] {a.title}")
        print(f"  source: {a.source}  tags: {a.tags}")
        print(f"  url: {a.url}")
        if a.summary_zh:
            print(f"  摘要: {a.summary_zh}")
    if failed_sources:
        print(f"\n⚠️ failed sources: {failed_sources}")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
