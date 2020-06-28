#### ref:
- https://zhuanlan.zhihu.com/p/32626442

#### SGD
- torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
  - args:
    - lr: learning rate
    - momentum: momentum factor
    - dampening: dampening for momentum
    - weight_decay: L2 penalty
    - nesterov: enables Nesterov momentum
  - ref:
    - https://blog.csdn.net/angel_hben/article/details/104620694
    - https://pytorch.org/docs/1.2.0/_modules/torch/optim/sgd.html#SGD
  - formula:
    - nesterov=False
      - v_{t+1} = momentum ∗ v_t + (g + weight_decay * p_t)  * (1 - dampening)
      - p_{t+1} = p_t − lr ∗ v_{t+1}
      - v_0 = g + weight_decay * p_0
        - p: parameters
        - g: gradient 
        - v: velocity, 惯性冲量
    - nesterov=True (相当于使用最新一步的惯性冲量，而不是上一步计算的惯性冲量)
      - v_{t+1} = momentum ∗ v_t + (g + weight_decay * p_t)  * (1 - dampening)
      - v_nesterov = (g + weight_decay * p_t) + momentum * v_{t+1}
      - p_{t+1} = p_t − lr ∗ v_nesterov
      - v_0 = g + weight_decay * p_0
        - p: parameters
        - g: gradient 
        - v: velocity, 惯性冲量
    

### per-parameter update
#### Adagrad
- formula:
  - cache = cache + g^2  # monotonically increasing 
  - p_{t+1} = p{t-1} - learning_rate * g / sqrt{cache + \epsilon}
  
#### RMSprop
- formula:
  - cache = \lambda * cache + (1 - \lambda) * g^2
  - p_{t+1} = p{t-1} - learning_rate * g / sqrt{cache + \epsilon}
  
#### Adadetla
  
#### Adam
- formula:
  - cache = \lambda * cache + (1 - \lambda) * g^2
  - v_{t+1} = \beta ∗ v_t + (1 - \beta) * g 
  - p_{t+1} = p{t-1} - learning_rate * v_{t+1} / sqrt{cache + \epsilon}