### Torch
- torch.Tensor
  - torch.Tensor is an alias for the default tensor type (torch.FloatTensor).
  
  - torch.Tensor is a class, torch.tensor is just a function.
  - torch.Tensor(torch.FloatTensor) is the default first weight (and bias if bias=True) init method (the default another weight init method is kaiming_uniform_, details can see the function reset_parameters) when you construct a conv model(eg:torch.Tensor(2, 3) creat a tensor of size 2 * 3 with random float value range from [-infinite, infinite] (obviously, these could not be used as init value to trained a model which will  make the model loss diverge), the actual random sampling method is not found yet). Generally you need to set weight again using your choosed method or resumimg from a pretrained model.
  - Current implementation of torch.Tensor introduces memory overhead, thus it might lead to unexpectedly high memory usage in the applications with many tiny tensors. If this is your case, consider using one large structure. Instead, using torch.from_numpy
  
- grad
  - only Tensors of floating point dtype can require gradients

 
- torch.nn.Parameter()
  - https://www.jianshu.com/p/d8b77cc02410
  

- init
  - xavier_uniform_: sampled from U(−a, a), a=gain \times \sqrt{\frac{6}{fan_in + fan_out}}
  
  - xavier_normal_: sampled from N(0, std^2), a=gain \times \sqrt{\frac{2}{fan_in + fan_out}}
  - kaiming_uniform_: sampled from U(−a, a), a=gain \times \sqrt{\frac{3}{fan_mode}}
  - kaiming_normal_: sampled from N(0, std^2), a=gain \times \sqrt{\frac{1}{fan_mode}}
  
  
- dtype
  - float32 is the default type of tensor
 
  - numpy default dtype is float64
  - tensor type transform: eg: tensor.float() or tensor.type(torch.FloatTensor)
    - tensor.byte()  -> torch.ByteTensor (dtype=torch.uint8)
    
    - tensor.char()  -> torch.CharTensor (dtype=torch.int8)
    - tensor.short()  -> torch.ShortTensor (dtype=torch.int16 / torch.short)
    - tensor.int()  -> torch.IntTensor (dtype=torch.int32 / torch.int)
    - tensor.long()  -> torch.LongTensor (dtype=torch.int64 / torch.long)
    - tensor.half()  -> torch.HalfTensor (dtype=torch.float16 / torch.half)
    - tensor.bfloat16()  -> torch.BFloat16Tensor (dtype=torch.bfloat16)
    - tensor.float()  -> torch.FloatTensor (dtype=torch.float32 / torch.float)
    - tensor.double()  -> torch.DoubleTensor (dtype=torch.float64 / torch.double)
    - tensor.bool()  -> torch.BoolTensor (dtype=torch.bool)
    
- numpy && tensor transformation, memory sharing
  - https://blog.csdn.net/shey666/article/details/85337212
  
- model.eval() && torch.no_grad()
  - with torch.no_grad： disables tracking of gradients in autograd.
  
  - model.eval()： changes the forward() behaviour of the module it is called upon. eg, it disables dropout and has batch norm use the entire population statistics
  
  
- ToTensor()
  - Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
​	

- tensor.to("cuda") or tensor.cuda()
  - 当gpu 负载较高时，to(DEVICE) 耗时非常大，小图片都能达到80ms，负载不高时，可能只需要0.1ms 左右，相差数百倍.压测的时候这个地方困惑了我不少时间
 
​	