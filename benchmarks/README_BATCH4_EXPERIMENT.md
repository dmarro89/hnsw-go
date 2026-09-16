# Search-layer four-distance experiment

This branch tests the blocked four-candidate Euclidean distance kernel in the real production arena construction path. Candidate IDs are marked visited in the existing neighbor order; groups of four distances are computed together, then consumed sequentially so heap threshold checks are re-evaluated after every candidate. Custom distance functions retain the scalar row-major path.

Promotion requires paired 100k x 128d single-thread efConstruction 64/200 benefit plus existing correctness, Recall@10 and cross-language gates. The extra blocked arena copy is intentionally measured as part of the experiment's allocation/memory cost.
