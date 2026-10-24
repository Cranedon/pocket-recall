Evidence recall: share of questions whose evidence turns all fit in the context. Embedding model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; abstention questions excluded.


### LoCoMo dev half: 801 questions (7 dropped: evidence ids match no turn)

| method | budget | multi-hop (147) | open-domain (50) | single-hop (449) | temporal (155) | all |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 2,000 | 19.7% | 30.0% | 75.9% | 65.2% | **60.7%** |
| bm25 | 8,000 | 37.4% | 48.0% | 86.2% | 77.4% | **73.2%** |
| userfirst-bm25-f3 | 2,000 | 15.6% | 30.0% | 74.4% | 63.2% | **58.7%** |
| userfirst-bm25-f3 | 8,000 | 34.7% | 44.0% | 85.3% | 77.4% | **71.9%** |
| userfirst-hybrid-f3 | 2,000 | 21.1% | 40.0% | 83.7% | 71.6% | **67.2%** |
| userfirst-hybrid-f3 | 8,000 | 53.1% | 60.0% | 94.9% | 87.1% | **83.5%** |
| userfirst-time-hybrid-f3 | 2,000 | 21.1% | 44.0% | 84.2% | 72.3% | **67.8%** |
| userfirst-time-hybrid-f3 | 8,000 | 54.4% | 60.0% | 95.3% | 88.4% | **84.3%** |
| userfirst-time-lane-hybrid-f3 | 2,000 | 21.1% | 44.0% | 84.2% | 72.3% | **67.8%** |
| userfirst-time-lane-hybrid-f3 | 8,000 | 54.4% | 60.0% | 95.3% | 88.4% | **84.3%** |

### LoCoMo test half (held out): 730 questions (2 dropped: evidence ids match no turn)

| method | budget | multi-hop (132) | open-domain (42) | single-hop (391) | temporal (165) | all |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 2,000 | 22.7% | 23.8% | 73.1% | 78.8% | **62.5%** |
| bm25 | 8,000 | 46.2% | 33.3% | 85.9% | 89.7% | **76.6%** |
| userfirst-bm25-f3 | 2,000 | 20.5% | 23.8% | 71.4% | 78.2% | **61.0%** |
| userfirst-bm25-f3 | 8,000 | 44.7% | 33.3% | 83.4% | 89.1% | **74.8%** |
| userfirst-hybrid-f3 | 2,000 | 34.1% | 33.3% | 79.3% | 86.7% | **70.1%** |
| userfirst-hybrid-f3 | 8,000 | 59.1% | 45.2% | 93.9% | 93.3% | **84.7%** |
| userfirst-time-hybrid-f3 | 2,000 | 34.8% | 35.7% | 79.3% | 87.9% | **70.7%** |
| userfirst-time-hybrid-f3 | 8,000 | 60.6% | 45.2% | 94.1% | 93.9% | **85.2%** |
| userfirst-time-lane-hybrid-f3 | 2,000 | 34.8% | 35.7% | 79.3% | 87.9% | **70.7%** |
| userfirst-time-lane-hybrid-f3 | 8,000 | 60.6% | 45.2% | 94.1% | 93.9% | **85.2%** |
