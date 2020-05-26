### Torch
- torch.Tensor
  - torch.Tensor is an alias for the default tensor type (torch.FloatTensor).
  - torch.Tensor is a class, torch.tensor is just a function.
  - torch.Tensor(torch.FloatTensor) is the default weight init method when you construct a conv model(eg:torch.Tensor(2, 3) creat a tensor of size 2 * 3 with random float value range from [-infinite, infinite] (obviously, these could not be used as init value to trained a model which will  make the model loss diverge), the actual random sampling method is not found yet). Generally you need to   set weight again using your choosed method or resumimg from a pretrained model.
  - Current implementation of torch.Tensor introduces memory overhead, thus it might lead to unexpectedly high memory usage in the applications with many tiny tensors. If this is your case, consider using one large structure. Instead, using torch.from_numpy