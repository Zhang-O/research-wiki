#### ref:
- https://zhuanlan.zhihu.com/p/32626442
- exponential moving average: 
  - https://blog.csdn.net/jason_cuijiahui/article/details/87652503
  - https://blog.csdn.net/bestrivern/article/details/86023616

#### [SGD](http://www.columbia.edu/~nq6/publications/momentum.pdf)
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
#### [Adagrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
- formula:
  - v_t = v_{t-1} + g^2  # monotonically increasing 
  - p_{t+1} = p_t - learning_rate * g / {sqrt{v_t} + \epsilon}
  
#### [RMSprop](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- formula:
  - v_t = \lambda * v_{t-1} + (1 - \lambda) * g^2
  - p_{t+1} = p_t - learning_rate * g / (sqrt{v_t}  + \epsilon)
  
#### Adadetla
  
#### [Adam](https://arxiv.org/abs/1412.6980)
- formula:
  - m_0 = 0, v_0= 0
  - v_t = \lambda * v_{t-1} + (1 - \lambda) * g^2
  - m_t = \beta ∗ m_{t-1} + (1 - \beta) * g 
  - p_{t+1} = p_t - learning_rate * m_t / (sqrt{v_t} + \epsilon)
  
  - if add bias correction:
    - \hat{m_t} = m_t / (1 - \beta ^ t)
    - \hat{v_t} = v_t / (1 - \lambda ^ t)
    - p_{t+1} = p_t - learning_rate * \hat{m_t} / (sqrt{\hat{v_t}} +  + \epsilon)
  

#### NAdam
- add NAG to Adam
- formula:
    - m_0 = 0, v_0= 0
  - v_t = \lambda * v_{t-1} + (1 - \lambda) * g^2
  - m_t = \beta ∗ m_{t-1} + (1 - \beta) * g 
  - m_t^{\prime} = \beta ∗ m_t + (1 - \beta) * g 
  - p_{t+1} = p_t - learning_rate * m_t^{\prime} / (sqrt{v_t} + \epsilon)
  
  - if add bias correction:
    - \hat{m_t} = m_t^{\prime} / (1 - \beta ^ {t+1})
    - \hat{v_t} = v_t / (1 - \lambda ^ t)
    - p_{t+1} = p_t - learning_rate * \hat{m_t} / (sqrt{\hat{v_t}} +  + \epsilon)