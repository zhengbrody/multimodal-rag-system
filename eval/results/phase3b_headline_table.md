# Flickr30k caption->image retrieval (phase3b_headline)

Queries: 5000 captions · depth=10

| Retriever | Modality | Recall@1 | Recall@5 (95% CI) | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| OpenCLIP ViT-H/14 (numpy, 1K) | text->image (cross-modal, ViT-H/14) | 0.777 | 0.942 ±0.006 | 0.967 | 0.847 | 0.877 |
