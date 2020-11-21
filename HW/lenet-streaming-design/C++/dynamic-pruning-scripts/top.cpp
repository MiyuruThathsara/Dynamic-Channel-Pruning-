#include "params.h"

void quant_dequant_act(
	    ap_int<QUANT_SCHEME> in_ch[][MAX_ACT_HEIGHT][MAX_ACT_WIDTH],
	    ap_int<QUANT_SCHEME> out_ch[][MAX_ACT_HEIGHT][MAX_ACT_WIDTH],
		ap_fixed<16,8,AP_RND> range_scale);

void var_ch_calc(
		ap_int<QUANT_SCHEME> in_ch[][MAX_ACT_HEIGHT][MAX_ACT_WIDTH],
		ap_int<QUANT_SCHEME> scaled_mean_in[],
		int ker_shape[],
		ap_int<QUANT_SCHEME> var_ch[]);

void abs_ch_calc(
		ap_int<QUANT_SCHEME> in_ch[][MAX_ACT_HEIGHT][MAX_ACT_WIDTH],
		ap_int<QUANT_SCHEME> scaled_mean_in[],
		int ker_shape[],
		ap_int<QUANT_SCHEME> abs_ch[]);

void select_channels(
		ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND> threshold,
		ap_int<QUANT_SCHEME> in_ch_array[],
		ap_uint<1> select_ch[],
		int ker_shape[],
		ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND> in_ch_scale);

void top(
		ap_int<QUANT_SCHEME> quantized_feature_map_in[3][MAX_ACT_HEIGHT][MAX_ACT_WIDTH],
		ap_int<QUANT_SCHEME> quantized_feature_map_out[3][MAX_ACT_HEIGHT][MAX_ACT_WIDTH],
		ap_int<QUANT_SCHEME> quantized_mean[MAX_CHANNEL_SIZE],
		int ker_shape[KERNEL_SHAPE_SIZE],
		ap_int<QUANT_SCHEME> abs_ch[MAX_CHANNEL_SIZE],
		ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND> threshold,
		ap_uint<1> selected_ch[MAX_CHANNEL_SIZE],
		ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND> de_quant_scale_1){

	ap_fixed<16,8,AP_RND> range_scale = 1.1;

	abs_ch_calc(quantized_feature_map_in, quantized_mean, ker_shape, abs_ch);
	select_channels(threshold, abs_ch, selected_ch, ker_shape, de_quant_scale_1);
	quant_dequant_act(quantized_feature_map_in, quantized_feature_map_out, range_scale);
}
