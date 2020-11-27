#include "params.h"

ap_int<QUANT_SCHEME> q_scheme_multiply(
		ap_int<QUANT_SCHEME> val_a,
		ap_int<QUANT_SCHEME> val_b){
	ap_int<QUANT_SCHEME * 2> inter_val = val_a * val_b;
	ap_int<QUANT_SCHEME> mult_out;
	if(inter_val.range(QUANT_SCHEME - 1, QUANT_SCHEME - 1) > 0){
		mult_out = inter_val.range(2 * QUANT_SCHEME - 1, QUANT_SCHEME) + (ap_int<QUANT_SCHEME>)1;
	}
	else{
		mult_out = inter_val.range(2 * QUANT_SCHEME - 1, QUANT_SCHEME);
	}
	return mult_out;
}

ap_int<QUANT_SCHEME> var_calc(
		ap_int<QUANT_SCHEME> in_ch[][MAX_ACT_WIDTH],
		ap_int<QUANT_SCHEME> scaled_mean){

//	quantizing scale factors
//	scale_in -> root( ( ( 2 ^ ( q-scheme - 1 ) ) - 1 ) / range of activations )
//	mean_in -> root( ( ( 2 ^ ( q-scheme - 1 ) ) - 1 ) / range of activations )

	ap_int<ACCUMILATOR_BITS> sd_out = 0;

	for(int i1 = 0; i1 < MAX_ACT_HEIGHT; i1++){
		for(int i2 = 0; i2 < MAX_ACT_WIDTH; i2++){
			sd_out += ( in_ch[i1][i2] - scaled_mean ) * ( in_ch[i1][i2] - scaled_mean );
		}
	}

//	quantizing scale factor of sd_val -> ( ( ( 2 ^ ( q-scheme - 1 ) ) - 1 ) / range of activations ) ^ 2
//	when de-quantizing no.of bits being shifted also should be considered
	ap_int<QUANT_SCHEME> sd_val = sd_out.range(ACCUMILATOR_BITS - 1, ADDED_ACCUMILATOR_BITS);
	return sd_val;
}

ap_int<QUANT_SCHEME> abs_calc(
		ap_int<QUANT_SCHEME> in_ch[][MAX_ACT_WIDTH],
		ap_int<QUANT_SCHEME> scaled_mean){

//	quantizing scale factors
//	scale_in -> root( ( ( 2 ^ ( q-scheme - 1 ) ) - 1 ) / range of activations )
//	mean_in -> root( ( ( 2 ^ ( q-scheme - 1 ) ) - 1 ) / range of activations )

	ap_int<ACCUMILATOR_BITS> abs_out = 0;
	ap_int<QUANT_SCHEME> inter_val;

	for(int i1 = 0; i1 < MAX_ACT_HEIGHT; i1++){
		for(int i2 = 0; i2 < MAX_ACT_WIDTH; i2++){
			inter_val = in_ch[i1][i2] - scaled_mean;
			if( inter_val.range(QUANT_SCHEME - 1, QUANT_SCHEME - 1) > 0 ){
				abs_out += ~inter_val + (ap_int<QUANT_SCHEME>) 1;
			}
			else{
				abs_out += inter_val;
			}
		}
	}

//	quantizing scale factor of abs_val -> ( ( 2 ^ ( q-scheme - 1 ) ) - 1 ) / range of activations
//	when de-quantizing no.of bits being shifted also should be considered
	ap_int<QUANT_SCHEME> abs_val = abs_out.range(ACCUMILATOR_BITS - 1, ADDED_ACCUMILATOR_BITS);
	return abs_val;
}

void var_ch_calc(
		ap_int<QUANT_SCHEME> in_ch[][MAX_ACT_HEIGHT][MAX_ACT_WIDTH],
		ap_int<QUANT_SCHEME> scaled_mean_in[],
		int ker_shape[],
		ap_int<QUANT_SCHEME> var_ch[]){

	for(int j = 0; j < MAX_CHANNEL_SIZE; j++){
		if(j < ker_shape[2]){
			var_ch[j] = var_calc(in_ch[j], scaled_mean_in[j]);
		}
	}
}

void abs_ch_calc(
		ap_int<QUANT_SCHEME> in_ch[][MAX_ACT_HEIGHT][MAX_ACT_WIDTH],
		ap_int<QUANT_SCHEME> scaled_mean_in[],
		int ker_shape[],
		ap_int<QUANT_SCHEME> abs_ch[]){

	for(int j = 0; j < MAX_CHANNEL_SIZE; j++){
		if(j < ker_shape[2]){
			abs_ch[j] = abs_calc(in_ch[j], scaled_mean_in[j]);
		}
	}
}

void select_channels(
		ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND> threshold,
		ap_int<QUANT_SCHEME> in_ch_array[],
		ap_uint<1> select_ch[],
		int ker_shape[],
		ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND> in_ch_scale){

	ap_int<QUANT_SCHEME> max_val;
	ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND> in_ch_vals[MAX_CHANNEL_SIZE] = {0};
	ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND> inter_fixed_val;
	for(int j = 0; j < MAX_CHANNEL_SIZE; j++){
		if(j < ker_shape[2]){
			if(j == 0){
				max_val = in_ch_array[j];
			}
			else{
				if(max_val < in_ch_array[j]){
					max_val = in_ch_array[j];
				}
			}

			inter_fixed_val = in_ch_scale * (ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND>)in_ch_array[j];

			if( inter_fixed_val.range( QUANT_SCHEME * 2 - 1, QUANT_SCHEME * 2 - 1 ) > 0){
				in_ch_vals[j] = QUANT_SCHEME_MAX;
			}
			else{
				in_ch_vals[j] = (ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND>)inter_fixed_val;
			}
		}
	}

	ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND> thresh_val = in_ch_scale * (ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND>)max_val * threshold;

	for(int k = 0; k < MAX_CHANNEL_SIZE; k++){
		if(k < ker_shape[2]){
			if(in_ch_vals[k] < thresh_val){
				select_ch[k] = 0;
			}
			else{
				select_ch[k] = 1;
			}
		}
	}
}
