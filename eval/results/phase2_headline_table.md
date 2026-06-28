# Flickr30k caption->image retrieval (phase2_headline)

Queries: 5000 captions · depth=10

| Retriever | Modality | Recall@1 | Recall@5 (95% CI) | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 (numpy, 1K) | text->image (cross-modal) | 0.593 | 0.844 ±0.010 | 0.902 | 0.699 | 0.748 |
