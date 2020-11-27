#include "params.h"

ap_int<QUANT_SCHEME> quant_dequant_val(
		ap_int<ACCUMILATOR_WIDTH> val_in,
		ap_fixed<16,0,AP_RND> scale);

void quant_convolution(
		q_data in_ch[][REQ_FEATURE_MAP_IN_HW][REQ_FEATURE_MAP_IN_HW],
		q_data out_ch[][REQ_FEATURE_MAP_OUT_HW][REQ_FEATURE_MAP_OUT_HW],
		q_data weight_kernel[][REQ_KERNEL_DEPTH][KERNEL_SHAPE][KERNEL_SHAPE],
		acc_data bias_kernel[],
		int in_ch_shape[],
		int kernel_shape[],
		int pad_size,
		int stride_size,
		ap_fixed<16,0,AP_RND> quant_dequant_scale)
{
#pragma HLS inline off
	int ker_pad = (int)(kernel_shape[0] / 2);
	int out_ch_shape[3] = { (int)(in_ch_shape[0] + 2*pad_size - kernel_shape[0] + 1), (int)(in_ch_shape[1] + 2*pad_size - kernel_shape[1] + 1), kernel_shape[2] };
#pragma HLS array_partition variable=out_ch_shape complete dim=0
	acc_data value;

	for( int num_ker = 0; num_ker < MAX_NUM_OF_KERNELS; num_ker++ ){
		if( num_ker < kernel_shape[2] ){
			for( int i1 = 0; i1 < REQ_FEATURE_MAP_OUT_HW; i1++ ){
				if( i1 < out_ch_shape[0] ){
					for( int i2 = 0; i2 < REQ_FEATURE_MAP_OUT_HW; i2++ ){
						if( i2 < out_ch_shape[1] ){
							value = 0;
							for( int i3 = 0; i3 < REQ_KERNEL_DEPTH; i3++ ){
								if( i3 < in_ch_shape[2] ){
									for( int k1 = -(KERNEL_SHAPE - 1)/2; k1 <= (KERNEL_SHAPE - 1)/2; k1++ ){
#pragma HLS unroll factor=5
										if( k1 <= ker_pad ){
											for( int k2 = -(KERNEL_SHAPE - 1)/2; k2 <= (KERNEL_SHAPE - 1)/2; k2++ ){
#pragma HLS pipeline
#pragma HLS unroll factor=5
												if( k2 <= ker_pad ){
													value += in_ch[ i3 ][ i2 + ker_pad + k1 ][ i1 + ker_pad + k2 ] * weight_kernel[ num_ker ][ i3 ][ k1 + ker_pad ][ k2 + ker_pad ];
												}
											}
										}
									}
								}
								else{
									break;
								}
							}
							value = value + bias_kernel[num_ker];
							if( value > 0 ){
								out_ch[num_ker][i2][i1] = quant_dequant_val(value, quant_dequant_scale);
							}
							else{
								out_ch[num_ker][i2][i1] = 0;
							}
						}
						else{
							break;
						}
					}
				}
				else{
					break;
				}
			}
		}
		else{
			break;
		}
	}
}
