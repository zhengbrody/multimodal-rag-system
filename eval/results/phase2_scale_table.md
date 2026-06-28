# Flickr30k caption->image retrieval (phase2_scale)

Queries: 1000 captions · depth=10

| Retriever | Modality | Recall@1 | Recall@5 (95% CI) | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| CLIP + Qdrant (31K gallery) | text->image (vector DB, 31K gallery) | 0.188 | 0.382 ±0.030 | 0.473 | 0.273 | 0.321 |
