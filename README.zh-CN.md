# pocket-recall

**给 AI 助手用的无损、本地长期记忆。** 每一段对话都原样存进你电脑上的一个 SQLite 文件。助手需要用到过去的信息时，pocket-recall 会挑出相关的部分，塞进一个很小的 token 预算（2k–8k），这样即使上下文窗口很小，也能用上几个月的历史。

- **不摘要、不丢弃。** 写入时不调用任何大模型：不需要 API key，没有按条计费，也不会因为抽取而丢掉细节。
- **本地、轻量。** 只有一个 SQLite 文件，加一个 0.22GB 的多语言嵌入模型，在 CPU 上运行。在约 24 万 token 的历史上，一次检索约 0.3 秒。
- **有测量。** 证据召回同时在 LongMemEval-S 和 LoCoMo 上测，LoCoMo 全程没有参与调参。端到端答题用官方提示词评判，逐题结果全部公开，每个数字都能复算。没有效果的实验也一并写出来。
- **中英文都支持。** 中文切成两字词做关键词检索；时间说法中英文都能识别，例如“上周”“三月份”“last week”“two weeks ago”。

[English](README.md) · [评测报告（英文）](docs/EVALUATION.md)

## 在 Claude Code 里安装

```
/plugin install pocket-recall --marketplace Cranedon/pocket-recall
```

插件包含三部分：

- 一个带 `recall` 工具的 MCP 服务；
- 一个 Stop hook，每轮对话结束后自动保存（只存你的输入和 Claude 写出来的回复，不存工具输出）；
- 一个 skill，告诉 Claude 什么时候该去翻记忆。

需要先装 [uv](https://docs.astral.sh/uv/)。第一次启动会建立独立的运行环境（约 170MB），并下载嵌入模型（约 240MB），之后完全离线运行。记忆存在 `~/.pocket-recall/memory.db`，升级或卸载插件都不会删除它。

想把安装之前的会话也导进来，直接对 Claude 说“把我以前的 Claude Code 会话导入记忆”，它会替你运行 `pocket-recall import claude-code`。

## 在其他 MCP 客户端里使用

Claude Desktop、Cursor 等客户端可以直接从 GitHub 运行：

```json
{
  "mcpServers": {
    "pocket-recall": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/Cranedon/pocket-recall", "pocket-recall", "mcp"]
    }
  }
}
```

这些客户端没有 Claude Code 的 hook，助手会按服务端的说明，用 `save_conversation` 工具自己保存对话。

## 原理

- **用户优先的打包方式。** 关于你的事实几乎都在你自己说的话里，而助手的回复往往很长。所以只有最相关的 3 轮会连同回复一起放进来，其余预算只放你的原话。在 LongMemEval 上，8k 预算下，只改这一点就把证据完全召回从 76.2% 提到 91.3%。但这只对“用户和助手”的对话有效；在两个真人对话的 LoCoMo 上，它没有带来变化。
- **混合排序。** BM25 擅长名字和数字，向量检索擅长换了说法的问题，用 RRF 融合两者。LongMemEval 上从 91.3% 提到 96.4%，没调过参的 LoCoMo 上从 73.3% 提到 84.1%。
- **时间感知。** “上周我做了什么”会换算成一个日期范围，范围内的对话加分，但范围外内容相关的对话不会被挤掉。只问时间的问题（“昨天我都聊了什么”）会严格按这个时段排序。
- **长消息截取，不硬切。** 贴进来的 3,000 token 简历会吃掉小预算，所以长消息只保留开头和与问题最相关的几段，原文仍完整存在数据库里。

## 结果

| 证据完全召回 | LongMemEval-S（开发集）2k | 8k | LoCoMo（留出集）2k | 8k |
|---|---:|---:|---:|---:|
| 原始 BM25（基线） | 36.4% | 76.2% | 61.5% | 74.8% |
| pocket-recall（默认） | 57.4% | **96.8%** | 69.2% | **84.7%** |

LongMemEval-S 端到端答题：回答和评判都用 Claude Opus 5.5，官方提示词，8k 预算，同一道题的两种方法来自同一个 API 端点，共 448 道配对题。BM25 基线 77.9%，pocket-recall **92.4%**；69 题更好，4 题更差，p = 2×10⁻¹⁶。其中多会话从 54.5% 提到 83.5%，时间推理从 75.6% 提到 92.1%。

复现命令、各类别的完整表格、没有效果的实验（写入时事实抽取、重排序、助手回复检索、bge-m3）以及已知的坑（不同中转的评判宽严不一），都在[评测报告](docs/EVALUATION.md)里。

## 隐私与安全

- 所有数据都留在本机。唯一的网络请求是第一次使用时从 Hugging Face 下载嵌入模型。
- **写入时自动脱敏：** 常见格式的 API key、JWT、私钥、URL 里的密码，以及 `api_key`、`password` 这类变量的值，会被替换成 `[REDACTED:类型]`。旧的记忆库可以用 `pocket-recall redact DB` 清理一遍；设置 `POCKET_RECALL_REDACT=0` 可以关闭。
- 可以用 `forget_session` 删除某段对话，用 `export_markdown()` 导出全部存储内容。

## 本地网页

`pocket-recall web --dir ~/.pocket-recall` 会在 http://localhost:8765 打开一个测试台，中英文可切换：可以浏览每段对话、搜索原文，并查看给定问题、预算和“当前时间”时，`recall` 到底会返回什么。

## 开发

```bash
git clone https://github.com/Cranedon/pocket-recall && cd pocket-recall
uv sync
uv run pytest
```

## 许可

MIT。`ref/` 里是 LongMemEval 的回答和评判提示词（MIT，© 2024 Di Wu）。LongMemEval 和 LoCoMo（CC BY-NC 4.0）由评测脚本下载，本仓库不分发这两个数据集。
