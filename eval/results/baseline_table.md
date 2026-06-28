# Flickr30k caption->image retrieval (baseline)

Queries: 100 captions · depth=10

| Retriever | Modality | Recall@1 | Recall@5 (95% CI) | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 (numpy, 1K) | text->image (cross-modal) | 0.630 | 0.830 ±0.074 | 0.900 | 0.728 | 0.770 |
| CLIP + Qdrant (1K gallery) | text->image (vector DB, 1K gallery) | 0.670 | 0.840 ±0.072 | 0.920 | 0.745 | 0.786 |
