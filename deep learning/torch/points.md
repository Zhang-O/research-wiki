<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
### Torch
- torch.Tensor
  - torch.Tensor is an alias for the default tensor type (torch.FloatTensor).
  - torch.Tensor is a class, torch.tensor is just a function.
  - torch.Tensor(torch.FloatTensor) is the default first weight (and bias if bias=True) init method (the default another weight init method is kaiming_uniform_, details can see the function reset_parameters) when you construct a conv model(eg:torch.Tensor(2, 3) creat a tensor of size 2 * 3 with random float value range from [-infinite, infinite] (obviously, these could not be used as init value to trained a model which will  make the model loss diverge), the actual random sampling method is not found yet). Generally you need to set weight again using your choosed method or resumimg from a pretrained model.
  - Current implementation of torch.Tensor introduces memory overhead, thus it might lead to unexpectedly high memory usage in the applications with many tiny tensors. If this is your case, consider using one large structure. Instead, using torch.from_numpy
  
 
- torch.nn.Parameter()
  - https://www.jianshu.com/p/d8b77cc02410
  

- init
  - xavier_uniform_: sampled from U(−a, a), a=gain \times \sqrt{\frac{6}{fan_in + fan_out}}
  - xavier_normal_: sampled from N(0, std^2), a=gain \times \sqrt{\frac{2}{fan_in + fan_out}}
  - kaiming_uniform_: sampled from U(−a, a), a=gain \times \sqrt{\frac{3}{fan_mode}}
  - kaiming_normal_: sampled from N(0, std^2), a=gain \times \sqrt{\frac{1}{fan_mode}}
  
  
- dtype
  - float32 is the default type of tensor
  - numpy default dtype is flaot64
​	
 
​	