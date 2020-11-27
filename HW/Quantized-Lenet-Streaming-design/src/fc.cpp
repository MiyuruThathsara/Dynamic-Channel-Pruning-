#include "params.h"

ap_int<QUANT_SCHEME> quant_dequant_val(
		ap_int<ACCUMILATOR_WIDTH> val_in,
		ap_fixed<16,0,AP_RND> scale);

ap_fixed<QUANT_SCHEME + 1, QUANT_SCHEME, AP_RND> dequant_val(
		ap_int<ACCUMILATOR_WIDTH> val_in,
		ap_fixed<16,0,AP_RND> scale);

void quant_fc_type1(
		q_data in_neur[][REQ_FEATURE_MAP_OUT_HW][REQ_FEATURE_MAP_OUT_HW],
		q_data out_neur[],
		q_data weight[][ FC1_NUM_NEURONES ],
		acc_data bias[],
		ap_fixed<16,0,AP_RND> quant_dequant_scale,
		int in_size = FC1_NUM_NEURONES,
		int out_size = FC2_NUM_NEURONES)
{
#pragma HLS inline off
	acc_data value;

	fc_type1_label1:for( int i1 = 0; i1 < FC2_NUM_NEURONES; i1++ ){
		if( i1 < out_size ){
			value = 0;
			fc_type1_label0:for( int i2 = 0; i2 < FC1_NUM_NEURONES; i2++ ){
#pragma HLS pipeline
#pragma HLS unroll factor=5
				if( i2 < in_size ){
					value += weight[ i1 ][ i2 ] * in_neur[ i2 ][ 0 ][ 0 ];
				}
			}
			value = value + bias[ i1 ];
			if( value > 0 ){
				out_neur[ i1 ] = quant_dequant_val(value, quant_dequant_scale);
			}
			else {
				out_neur[ i1 ] = 0;
			}
		}
		else{
			break;
		}
	}
}

void quant_fc_type2(
		q_data in_neur[],
		out_data out_neur[],
		q_data weight[][ FC2_NUM_NEURONES ],
		acc_data bias[],
		ap_fixed<16,0,AP_RND> quant_dequant_scale,
		int in_size = FC2_NUM_NEURONES,
		int out_size = NUM_OUT_NEURONES)
{
#pragma HLS inline off
	acc_data value;

	fc_type2_label1:for( int i1 = 0; i1 < NUM_OUT_NEURONES; i1++ ){
		if( i1 < out_size ){
			value = 0;
			fc_type2_label0:for( int i2 = 0; i2 < FC2_NUM_NEURONES; i2++ ){
#pragma HLS pipeline
#pragma HLS unroll factor=7
				if( i2 < in_size ){
					value += weight[ i1 ][ i2 ] * in_neur[ i2 ];
				}
			}
			value = value + bias[ i1 ];
			out_neur[ i1 ] = dequant_val(value, quant_dequant_scale);
		}
		else{
			break;
		}
	}
}
