# Backbone sweep — Recall@5 lift comes from the dense backbone

Flickr30k caption→image, standard 5,000-caption protocol, 1K gallery. Only the
backbone size varies (LAION-2B weights throughout); **no reranking**.

| Dense backbone | Recall@1 | Recall@5 (95% CI) | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| OpenCLIP ViT-B/32 (laion2b) | 0.664 | 0.881 ±0.009 | 0.934 | 0.758 | 0.801 |
| OpenCLIP ViT-L/14 (laion2b) | 0.755 | 0.929 ±0.007 | 0.959 | 0.830 | 0.862 |
| OpenCLIP ViT-H/14 (laion2b) | 0.777 | 0.942 ±0.006 | 0.967 | 0.847 | 0.877 |
