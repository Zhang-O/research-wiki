- onnx
  - Segmentation fault (core dumped)
    - 在import torch之前import onnx，二者的前后顺序要注意
  
  - upsampling-bilinear:
    - https://www.jianshu.com/p/dcd450aa2e41
    
  - onnx / pytorch produce different results
    - [通过在测试代码中运行pytorch模型推断之前添加 model.eval()解决问题](https://stackoverflow.com/questions/57423150/outputs-are-different-between-onnx-and-pytorch)