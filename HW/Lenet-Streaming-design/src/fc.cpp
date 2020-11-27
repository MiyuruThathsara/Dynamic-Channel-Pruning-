#include "params.h"

void fc_type1(
		float in_neur[][REQ_FEATURE_MAP_OUT_HW][REQ_FEATURE_MAP_OUT_HW],
		float out_neur[],
		float weight[][ FC1_NUM_NEURONES ],
		float bias[],
		int in_size = FC1_NUM_NEURONES,
		int out_size = FC2_NUM_NEURONES)
{
#pragma HLS inline off
	float value;

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
				out_neur[ i1 ] = value;
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

void fc_type2(
		float in_neur[],
		float out_neur[],
		float weight[][ FC2_NUM_NEURONES ],
		float bias[],
		int in_size = FC2_NUM_NEURONES,
		int out_size = NUM_OUT_NEURONES)
{
#pragma HLS inline off
	float value;

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
			out_neur[ i1 ] = value;
		}
		else{
			break;
		}
	}
}
