# Flickr30k caption->image retrieval (phase1)

Queries: 1000 captions · depth=10

| Retriever | Modality | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 (numpy, 1K) | text->image (cross-modal) | 0.567 | 0.822 | 0.886 | 0.677 | 0.728 |
| CLIP + Qdrant (1K gallery) | text->image (vector DB, 1K gallery) | 0.568 | 0.813 | 0.883 | 0.673 | 0.724 |
| CLIP + Qdrant (31K gallery) | text->image (vector DB, 31K gallery) | 0.188 | 0.382 | 0.473 | 0.273 | 0.321 |
| Dense MiniLM | text->text (dense proxy, local) | 0.564 | 0.778 | 0.847 | 0.655 | 0.701 |
| Mock (keyword) | text->text (lexical proxy) | 0.269 | 0.451 | 0.525 | 0.348 | 0.390 |
