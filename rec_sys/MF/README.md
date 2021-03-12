### Most popular matrix factorization algorithms for collaborative filtering:

- [SVD](https://hackernoon.com/introduction-to-recommender-system-part-1-collaborative-filtering-singular-value-decomposition-44c9659c5e75)
	'''python
	svd = StochasticGradientDescent(iterations=1e5, factors=64)
	
	svd.fit(user_to_item)
	'''

- [ALS](http://yifanhu.net/PUB/cf.pdf)
	- [source code](ALS.py)

- [BPR](https://arxiv.org/pdf/1205.2618.pdf)
	- [source code](BPR.py)

- [WARP](http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf)
	- [source code](WARP.py)
