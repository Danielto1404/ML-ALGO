### Most popular matrix factorization algorithms for collaborative filtering:

- [SVD](https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75)
```python
svd = StochasticGradientDescent(iterations=1e5, factors=64, learning_rate=1e-4, alpha=1e-5)
svd.fit(user_to_item)

svd.similar_items(item_id=0, top_k=20)
svd.recommend(user_index=239, amount=5)
```

- [ALS](http://yifanhu.net/PUB/cf.pdf)
```python
als = ALS(iterations=20, factors=64, confidence=40)
als.fit(user_to_item)

als.similar_items(item_id=0, top_k=20)
als.recommend(user_index=239, amount=5)
```

- [BPR](https://arxiv.org/pdf/1205.2618.pdf)
```python
bpr = BPR(iterations=200, factors=64, learning_rate=1e-2, alpha=1e-5)
bpr.fit(user_to_item)

bpr.similar_items(item_id=0, top_k=20)
bpr.recommend(user_index=239, amount=5)
```

- [WARP](http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf)
```python
warp = WARP(iterations=50, factors=64, learning_rate=1e-3, alpha=1e-2, max_warp_sampled=100)
warp.fit(user_to_item)

warp.similar_items(item_id=0, top_k=20)
warp.recommend(user_index=239, amount=5)
```
