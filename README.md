# ML-ALGO

Implementation of most popular ML algorithms

- ### [Neural networks](networks)
  Implementation of [*Multilayer perceptron*](https://en.wikipedia.org/wiki/Multilayer_perceptron). Used naive [*Stochastic Gradient Descent*](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
  to minimize [*Loss function*](https://en.wikipedia.org/wiki/Loss_function).

  #### Also, you can check your progress via progress bar (implemented via [*tqdm*](https://github.com/tqdm/tqdm))
  ![Screenshot](tests/screenshots/loss.png)

  Don't support **GPU and parallel** evaluations.

  #### List of available **SGD** optimizers:
    - Naive SGD
    - [Momentum](https://en.wikipedia.org/wiki/Momentum_(technical_analysis))
    - [Adam](https://arxiv.org/abs/1412.6980)
    - RMSProp
    - AdaDelta
    - AdaGrad
  
  #### Todo in the future:
    - Batch normalization layer
    - NAG optimizer
    - Nadam optimizer
    - MiniBatch SGD
    - CNN or RNN

  Other optimizations overviewed [here](https://towardsdatascience.com/deep-learning-optimizers-436171c9e23f)

  **Main classes**:
    - [Network class](networks/net.py)
    - [Loss functions](networks/core/functions/losses.py)
    - [Activation functions](networks/core/functions/activations.py)


- ### [Recommendation systems](recsys)
  Implementation of **Matrix factorizations** algorithms.

    - [Matrix factorizations](recsys/mf)
        - [SVD](recsys/mf/sgd.py)
        - [ALS](recsys/mf/als.py)
        - [BPR](recsys/mf/bpr.py)
        - [WARP](recsys/mf/warp.py)
