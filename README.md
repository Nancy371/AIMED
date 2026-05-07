# AI 医疗每日情报 Agent

每天自动抓取 AI 医疗领域（药物发现 + 临床大模型）的最新论文和团队博客，用 LLM 生成中文摘要并打分，推送到飞书群。

## 架构

```
   GitHub Actions (每天北京时间 08:07)
            ↓
   fetch (RSS + PubMed)
            ↓
   dedupe (SQLite)
            ↓
   score & summarize (LLM)
            ↓
   push (飞书交互卡片)
```

- **[src/fetch.py](src/fetch.py)** — RSS（feedparser）+ PubMed E-utilities
- **[src/dedupe.py](src/dedupe.py)** — SQLite 去重，状态 commit 回仓库
- **[src/llm.py](src/llm.py)** — 可替换的 LLM provider（Claude / OpenAI 兼容）
- **[src/score.py](src/score.py)** — 批量打分 + 中文摘要
- **[src/feishu.py](src/feishu.py)** — 飞书 webhook 推送
- **[config/sources.yaml](config/sources.yaml)** — 源清单（易于扩展）

## 🚀 一键搭建（分享给别人用）

把下面这段**整段**发给朋友，他/她粘贴给 Claude Code、Cursor 或任何 AI 编程助手，即可自动搭建：

````
我想搭建一个每天自动抓取 AI 医疗论文、推送到飞书的 agent。
参考项目：https://github.com/Nancy371/AIMED

请帮我完成：
1. 在 GitHub 上 fork 仓库 Nancy371/AIMED 到我的账号
2. git clone 我的 fork 到本地
3. pip install -r requirements.txt
4. 本地用 --dry-run 跑一次验证 fetch 链路
5. 指导我在 GitHub 仓库 Settings → Secrets 里添加：
   - ANTHROPIC_API_KEY（从 console.anthropic.com 获取）
   - FEISHU_WEBHOOK_URL（飞书群 → 设置 → 群机器人 → 添加自定义机器人）
6. 在 Actions 标签页手动触发 "Daily AI Med Digest" workflow
7. 检查飞书群是否收到情报卡片

不熟悉的概念请在每一步解释给我听。
````

**或者用命令行手动搭：**

```bash
# 1. 先在 https://github.com/Nancy371/AIMED 点 Fork
# 2. 克隆你的 fork（替换成你的 GitHub 用户名）
git clone https://github.com/<YOUR_GITHUB_USERNAME>/AIMED.git ai-med-daily
cd ai-med-daily

# 3. 装依赖
pip install -r requirements.txt

# 4. 本地测一下 fetch 链路（不调用 LLM、不推送）
python -m src.main --skip-score --dry-run

# 5. 到你 fork 的仓库 Settings → Secrets 加入：
#    ANTHROPIC_API_KEY, FEISHU_WEBHOOK_URL
# 6. Actions 标签页手动触发一次 "Daily AI Med Digest"
```

> 💡 想换成 DeepSeek / Kimi / Qwen 等国产模型：见下面「切换 LLM 厂商」章节。

## 快速开始

### 1. 准备环境

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置密钥（本地测试）

创建 `.env`（不要 commit），或直接 export：

```bash
# LLM — 默认用 Claude Haiku（最低成本路径）
export LLM_PROVIDER=anthropic
export LLM_API_KEY=sk-ant-xxx

# 飞书自定义机器人 webhook（群聊→设置→群机器人→添加自定义机器人）
export FEISHU_WEBHOOK_URL=https://open.feishu.cn/open-apis/bot/v2/hook/xxx
# 可选：如开启签名校验，填机器人密钥
export FEISHU_SIGN_SECRET=xxx
```

### 3. 本地调试

```bash
# 只 fetch + 打分、不推送
python -m src.main --dry-run

# 限制 3 篇，验证飞书消息格式
python -m src.main --limit 3

# 跳过 LLM（纯测试 fetch / 去重）
python -m src.main --skip-score --dry-run
```

### 4. 部署到 GitHub Actions

在仓库 **Settings → Secrets and variables → Actions** 添加：

| 类型 | 名称 | 说明 |
|---|---|---|
| Secret | `ANTHROPIC_API_KEY` 或 `LLM_API_KEY` | LLM API 密钥 |
| Secret | `FEISHU_WEBHOOK_URL` | 飞书机器人 webhook URL |
| Secret | `FEISHU_SIGN_SECRET` | （可选）飞书签名校验密钥 |
| Variable | `LLM_PROVIDER` | （可选，默认 `anthropic`）换其他厂商时需要 |
| Variable | `LLM_MODEL` | （可选）覆盖默认模型 |
| Variable | `LLM_BASE_URL` | （可选）OpenAI 兼容厂商的 endpoint |

推送代码后，先在 Actions 页面手动触发一次（**Daily AI Med Digest → Run workflow**）验证。cron 会在每天 UTC 00:07（北京 08:07）自动跑。

## 切换 LLM 厂商

代码支持两类 provider，通过环境变量切换：

### Claude（默认）

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5       # 默认
LLM_API_KEY=sk-ant-xxx
```

### OpenAI 兼容（覆盖 OpenAI / DeepSeek / Kimi / Qwen / GLM 等）

国内推荐用 **DeepSeek**，便宜且兼容性好：

```bash
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-xxx
```

其他厂商配置参考：

| 厂商 | `LLM_BASE_URL` | `LLM_MODEL` 推荐值 |
|---|---|---|
| OpenAI | （不设） | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |
| Kimi | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| Qwen (DashScope) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-turbo` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4/` | `glm-4-flash` |
| SiliconFlow | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-7B-Instruct` |

## 扩展源

编辑 [config/sources.yaml](config/sources.yaml) 添加 RSS 或 PubMed 检索式，立刻生效。两种类型：

```yaml
- name: My RSS
  type: rss
  url: https://example.com/feed.xml
  category: drug_discovery  # 或 clinical_llm
  max_items: 10

- name: My PubMed Query
  type: pubmed
  query: ("foundation model"[Title/Abstract]) AND ("radiology"[MeSH])
  category: clinical_llm
  max_items: 15
```

## 成本估算

默认配置下（每天 50-100 篇新文章、Claude Haiku 批量打分）：
- **LLM**：~$0.05/天，约 $1.5/月
- **GitHub Actions**：公开仓库免费；私有仓库每月 2000 分钟免费额度远够用
- 换 DeepSeek 可进一步降低到月 $0.2 以下

## 未来扩展

- [ ] FDA / EMA / NMPA 监管政策抓取（无 RSS，需网页抓取 + diff）
- [ ] 头部公司博客（Isomorphic Labs、Recursion、Nabla 等多数无 RSS）
- [ ] 可选的每周汇总 / 月报
