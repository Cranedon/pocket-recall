"""Tiny Chinese/mixed-language retrieval check: is the right message ranked in the top 3?

20 personal facts, each asked about in different words, among 20 unrelated messages.
Far too small to be a benchmark; it only shows whether a model handles Chinese at all.
Run: uv run python -m pocket_recall.eval.zh_check
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from ..core import rank
from ..embed import Embedder, Reranker

PAIRS = [
    ("我对花生过敏，吃了会起疹子。", "做饭的时候我要避开哪些食材？"),
    ("我家的猫叫味噌，今年六月满五岁。", "我的宠物多大了？"),
    ("我在跟伊尔莎老师学大提琴，每周二上课。", "谁在教我乐器？"),
    ("我用的是一台 MacBook Pro M3，16G 内存。", "我的电脑是什么配置？"),
    ("上个月我把旧的咖啡机送人了，换了一台意式咖啡机。", "我最近换了哪些厨房电器？"),
    ("我每天早上都会跑五公里，周末跑十公里。", "我平时的运动习惯是什么？"),
    ("我女儿明年九月上小学一年级。", "我的孩子什么时候开始上学？"),
    ("我在一家做新能源汽车电池的公司当工程师。", "我是做什么工作的？"),
    ("我去年考下了 PMP 项目管理证书。", "我有哪些职业资格认证？"),
    ("我戴了助听器，最近电池总是不够用。", "我日常用哪些医疗设备？"),
    ("我和老婆打算国庆去云南大理玩一周。", "我们假期要去哪里旅行？"),
    ("我最喜欢的米是日本的短粒米，煮出来很糯。", "我喜欢吃什么样的大米？"),
    ("我上周把车的轮胎换成了米其林的。", "我的车最近做过什么保养？"),
    ("我在学西班牙语，用的是 Duolingo，已经连续打卡 200 天。", "我在学什么外语？"),
    ("我每个月给山区小学捐 300 块钱。", "我参与过哪些公益活动？"),
    ("我有轻度的高血压，医生让我少吃盐。", "我的健康有什么需要注意的？"),
    ("我的生日是 1991 年 3 月 8 日。", "我今年多大？"),
    ("我住在杭州西湖区，离公司骑车十分钟。", "我住在哪个城市？"),
    ("周末我一般在家拼乐高，最近在拼千年隼。", "我有什么爱好？"),
    ("我把 Netflix 退订了，现在只用 B 站看纪录片。", "我现在用哪些视频平台？"),
]
DISTRACTORS = [
    "帮我写一封请假邮件，语气正式一点。", "Python 里怎么读取 JSON 文件？", "解释一下什么是通货膨胀。",
    "推荐几本关于二战的历史书。", "番茄炒蛋先放番茄还是先放蛋？", "怎么给 Excel 表格加下拉菜单？",
    "把这段话翻译成英文：今天天气很好。", "量子计算和普通计算有什么区别？", "如何提高 PPT 的设计感？",
    "写一首关于秋天的七言绝句。", "Git 怎么撤销上一次提交？", "介绍一下敦煌莫高窟的历史。",
    "面试时怎么回答“你最大的缺点”？", "什么是复利？举个例子。", "怎么判断西瓜熟没熟？",
    "帮我列一个搬家清单。", "Transformer 模型的注意力机制是什么？", "怎么清洗洗衣机的滚筒？",
    "推荐几部高分的科幻电影。", "如何写一份好的项目周报？",
]
MODELS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-small-zh-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "jinaai/jina-embeddings-v2-base-zh",
]


def hits_at_3(scorer: str, embedder: Embedder | None, reranker=None) -> int:
    texts = [fact for fact, _ in PAIRS] + DISTRACTORS
    return sum(i in rank(texts, question, scorer, embedder, reranker)[:3] for i, (_, question) in enumerate(PAIRS))


def main(models: list[str]) -> None:
    cache = Path(tempfile.mkdtemp()) / "zh_check.sqlite"
    print(f"{'model':<62}{'dense':>7}{'hybrid':>8}   (top-3 hits out of {len(PAIRS)})")
    print(f"{'(BM25 only, CJK bigrams)':<62}{'':>7}{hits_at_3('bm25', None):>8}")
    rerank_model = os.environ.get("POCKET_RECALL_RERANK_MODEL")
    reranker = Reranker(rerank_model) if rerank_model else None
    for model in models:
        emb = Embedder(cache, model)
        line = f"{model:<62}{hits_at_3('dense', emb):>7}{hits_at_3('hybrid', emb):>8}"
        if reranker:
            line += f"{hits_at_3('hybrid', emb, reranker):>10}  (+rerank)"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:] or MODELS)
