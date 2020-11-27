import numpy as np
import torch
from Quant_funcs import *

a = torch.tensor([[0.1,0.5,0.9],[-0.2,0.36,-0.6],[-1.2,-0.75,0.98]])
quant_list = quant_val_list(1.0,8)
quantized_array = quantize_array(a,quant_list)
print(quantized_array)