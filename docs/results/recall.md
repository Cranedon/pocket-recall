Evidence recall: share of questions whose evidence turns all fit in the context. Embedding model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; abstention questions excluded.


### LongMemEval-S (development set): 470 questions

| method | budget | knowledge-update (72) | multi-session (121) | single-session-assistant (56) | single-session-preference (30) | single-session-user (64) | temporal-reasoning (127) | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 2,000 | 25.0% | 11.6% | 94.6% | 26.7% | 85.9% | 18.1% | **36.4%** |
| bm25 | 8,000 | 90.3% | 52.9% | 100.0% | 63.3% | 96.9% | 72.4% | **76.2%** |
| userfirst-bm25-f3 | 2,000 | 63.9% | 25.6% | 92.9% | 43.3% | 93.8% | 49.6% | **56.4%** |
| userfirst-bm25-f3 | 8,000 | 100.0% | 81.8% | 96.4% | 86.7% | 100.0% | 89.8% | **91.3%** |
| userfirst-hybrid-f3 | 2,000 | 68.1% | 28.9% | 92.9% | 56.7% | 93.8% | 46.5% | **57.9%** |
| userfirst-hybrid-f3 | 8,000 | 100.0% | 92.6% | 96.4% | 100.0% | 100.0% | 95.3% | **96.4%** |
| userfirst-time-hybrid-f3 | 2,000 | 66.7% | 28.9% | 92.9% | 50.0% | 93.8% | 47.2% | **57.4%** |
| userfirst-time-hybrid-f3 | 8,000 | 100.0% | 92.6% | 96.4% | 96.7% | 100.0% | 97.6% | **96.8%** |

### LoCoMo (held out): 1531 questions (9 dropped: evidence ids match no turn)

| method | budget | multi-hop (279) | open-domain (92) | single-hop (840) | temporal (320) | all |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 2,000 | 21.1% | 27.2% | 74.6% | 72.2% | **61.5%** |
| bm25 | 8,000 | 41.6% | 41.3% | 86.1% | 83.8% | **74.8%** |
| userfirst-bm25-f3 | 2,000 | 17.9% | 27.2% | 73.0% | 70.9% | **59.8%** |
| userfirst-bm25-f3 | 8,000 | 39.4% | 39.1% | 84.4% | 83.4% | **73.3%** |
| userfirst-hybrid-f3 | 2,000 | 27.2% | 37.0% | 81.7% | 79.4% | **68.6%** |
| userfirst-hybrid-f3 | 8,000 | 55.9% | 53.3% | 94.4% | 90.3% | **84.1%** |
| userfirst-time-hybrid-f3 | 2,000 | 27.6% | 40.2% | 81.9% | 80.3% | **69.2%** |
| userfirst-time-hybrid-f3 | 8,000 | 57.3% | 53.3% | 94.8% | 91.2% | **84.7%** |
