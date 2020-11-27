import numpy as np
import torch

def quant_val_list(max_val, quant_bit_width):
    neg_list = [ ( - max_val * i1 / ( 2**(quant_bit_width - 1) - 1 ) ) for i1 in range(2**(quant_bit_width - 1) - 1,0,-1) ]
    pos_list = [ ( max_val * i2 / ( 2**(quant_bit_width - 1) - 1 ) ) for i2 in range(2**(quant_bit_width - 1) ) ]
    return neg_list + pos_list

def quant_thresh(quant_vals):
    return abs(quant_vals[0] - quant_vals[1]) / 2

def quantize_array(array, quant_vals):
    quant_threshold = quant_thresh(quant_vals)
    max_set = np.where( array.numpy() >= max(quant_vals), max(quant_vals), 0 )
    min_set = np.where( array.numpy() <= min(quant_vals), min(quant_vals), 0 )
    quantized_arr = np.array([ np.where( (quant_vals[i] - array.numpy()) < quant_threshold - 10**-8, np.where( (quant_vals[i] - array.numpy()) >= -quant_threshold, quant_vals[i], 0 ), 0) for i in range(len(quant_vals)) ]).sum(axis = 0)
    return torch.from_numpy( np.where( np.abs(min_set) > 0, min_set, np.where( np.abs(max_set) > 0, max_set, quantized_arr ) ) )

def quantize_array_to_levels(array, quant_vals):
    quant_threshold = quant_thresh(quant_vals)
    thresh = ( len(quant_vals) - 1 ) / 2
    quant_levels = len(quant_vals)
    max_set = np.where( array.numpy() >= max(quant_vals), (quant_levels - 1) - thresh, 0 )
    min_set = np.where( array.numpy() <= min(quant_vals), -thresh, 0 )
    quantized_arr = np.array([ np.where( (quant_vals[i] - array.numpy()) < quant_threshold - 10**-8, np.where( (quant_vals[i] - array.numpy()) >= -quant_threshold, ( i - thresh ), 0 ), 0) for i in range(len(quant_vals)) ]).sum(axis = 0)
    return torch.from_numpy( np.where( np.abs(min_set) > 0, min_set, np.where( np.abs(max_set) > 0, max_set, quantized_arr ) ) )