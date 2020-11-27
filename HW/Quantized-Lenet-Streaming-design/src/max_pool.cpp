#include "params.h"

void quant_max_pool(
		q_data in_ch[][REQ_FEATURE_MAP_OUT_HW][REQ_FEATURE_MAP_OUT_HW],
		q_data out_ch[][REQ_FEATURE_MAP_IN_HW][REQ_FEATURE_MAP_IN_HW],
		int in_ch_shape[],
		int kernel_size,
		int stride_size)
{
#pragma HLS inline off
	int out_ch_shape[3] = { (int)(in_ch_shape[0] / stride_size), (int)(in_ch_shape[1] / stride_size), in_ch_shape[2] };
#pragma HLS array_partition variable=out_ch_shape complete dim=0
	q_data max_val;
	q_data max_prev;

	for( int i1 = 0; i1 < CONV2_IN_HW; i1++ ){
#pragma HLS unroll factor=5
		if( i1 < out_ch_shape[0] ){
			for( int i2 = 0; i2 < CONV2_IN_HW; i2++ ){
#pragma HLS unroll factor=5
				if( i2 < out_ch_shape[1] ){
					for( int i3 = 0; i3 < REQ_KERNEL_DEPTH; i3++ ){
						if( i3 < out_ch_shape[2] ){
							for( int k1 = 0; k1 < MP_WINDOW_SHAPE; k1++ ){
#pragma HLS pipeline
#pragma HLS unroll factor=2
								if( k1 < kernel_size ){
									max_prev = max_val;
									if( in_ch[ i3 ][ i2 * stride_size + k1 ][ i1 * stride_size ] > in_ch[ i3 ][ i2 * stride_size + k1 ][ i1 * stride_size + 1 ] ){
										max_val = in_ch[ i3 ][ i2 * stride_size + k1 ][ i1 * stride_size ];
									}
									else {
										max_val = in_ch[ i3 ][ i2 * stride_size + k1 ][ i1 * stride_size + 1 ];
									}
								}
							if( max_prev > max_val ){
								out_ch[ i3 ][ i2 ][ i1 ] = max_prev;
							}
							else {
								out_ch[ i3 ][ i2 ][ i1 ] = max_val;
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
		else{
			break;
		}
	}
}
