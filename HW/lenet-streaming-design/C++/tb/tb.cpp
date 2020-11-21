#include "params.h"

ap_int<QUANT_SCHEME> quant_val(
		ap_int<QUANT_SCHEME> val_in,
		ap_fixed<16,8,AP_RND> scale);

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
		ap_uint<1> select_ch[MAX_CHANNEL_SIZE],
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
		ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND> de_quant_scale_1);

int main(){
////////////////////////////////////////////////////////////////////////////////////
//	Testing Quantization
///////////////////////////////////////////////////////////////////////////////////

//	ap_int<QUANT_SCHEME> val_in = 5;
//	float scale = 1.14;
//	ap_int<QUANT_SCHEME> out;
//
//	out = quant_val(val_in, scale);
//	cout << "Final quantized value = " << out << endl;

////////////////////////////////////////////////////////////////////////////////////
//	Testing Channel Selection
////////////////////////////////////////////////////////////////////////////////////

//	int ker_shape[3] = {5, 5, 6};
//	ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND> threshold = 0.7;
//	ap_int<QUANT_SCHEME> channel_arr[6] = { 112, 23, 56, 45, 43, 69 };
//	ap_uint<1> sel_chs[MAX_CHANNEL_SIZE] = 0;
//
//	select_channels(threshold, channel_arr, sel_chs, ker_shape);
//	for(int i = 0; i < MAX_CHANNEL_SIZE; i++){
//		cout << (int)sel_chs[i] << endl;
//	}

////////////////////////////////////////////////////////////////////////////////////
//	Testing Quantization and Channel selection together for a given activation map
///////////////////////////////////////////////////////////////////////////////////

//	ap_int<QUANT_SCHEME> quantized_feature_map[3][MAX_ACT_HEIGHT][MAX_ACT_WIDTH] = {{{23,45,65},{12,99,112},{69,32,21}},{{56,32,75},{120,56,87},{10,100,46}},{{21,25,36},{50,66,31},{110,86,34}}};
//	ap_int<QUANT_SCHEME> mean_arr[3] = {53,65,51};

	float feature_map[3][MAX_ACT_HEIGHT][MAX_ACT_WIDTH] = {{{0.23,0.05,0.45},{0.687,0.124,0.86},{0.36,0.548,0.632}},{{0.04,0.0158,0.754},{0.147,0.86,0.5},{0.15,0.48,0.96}},{{0.642,0.364,0.25},{0.75,0.621,0.358},{0.84,0.14,0.35}}};
	ap_int<QUANT_SCHEME> quantized_feature_map_in[3][MAX_ACT_HEIGHT][MAX_ACT_WIDTH];
	ap_int<QUANT_SCHEME> quantized_feature_map_out[3][MAX_ACT_HEIGHT][MAX_ACT_WIDTH];
	float float_mean[3];
	float float_mean_scale = 0.2;
	ap_int<QUANT_SCHEME> quantized_mean[3];
	float acc_val;
	for(int i1 = 0; i1 < 3; i1 ++){
		acc_val = 0;
		for(int i2 = 0; i2 < MAX_ACT_HEIGHT; i2++){
			for(int i3 = 0; i3 < MAX_ACT_WIDTH; i3++){
				quantized_feature_map_in[i1][i2][i3] = (ap_int<QUANT_SCHEME>)( feature_map[i1][i2][i3] * (pow(2, QUANT_SCHEME - 1) - 1) / QUANT_RANGE_LAYER_1 );
				acc_val += feature_map[i1][i2][i3];
			}
		}
		float_mean[i1] = acc_val / (MAX_ACT_HEIGHT * MAX_ACT_WIDTH);
 		quantized_mean[i1] = (ap_int<QUANT_SCHEME>)( float_mean[i1] * float_mean_scale * ( pow(2, QUANT_SCHEME - 1) - 1 ) / QUANT_RANGE_LAYER_1 );
	}

	int ker_shape[KERNEL_SHAPE_SIZE] = {5,5,3};
	ap_int<QUANT_SCHEME> abs_ch[MAX_CHANNEL_SIZE] = {0};
	ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND> threshold = THRESHOLD;
	float float_threshold = THRESHOLD;
	ap_uint<1> selected_ch[MAX_CHANNEL_SIZE] = {0};

//	Feature quantity parameter here is ABS. So for that, de-quantization factor for ABS list is as follows
	ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND> de_quant_scale_1 = (ap_fixed<QUANT_SCHEME * 2, QUANT_SCHEME, AP_RND>) ( QUANT_RANGE_LAYER_1 * pow(2, ADDED_ACCUMILATOR_BITS) / ( pow(2, QUANT_SCHEME - 1) - 1 ) );

	top(quantized_feature_map_in, quantized_feature_map_out, quantized_mean, ker_shape, abs_ch, threshold, selected_ch, de_quant_scale_1);

	for(int i = 0; i < ker_shape[2]; i++){
			cout << "quantized act sub mean: " << abs_ch[i] << endl;
		}

	for(int m1 = 0; m1 < 3; m1++){
		cout << "/////////////////////////////////" << endl;
		for(int m2 = 0; m2 < MAX_ACT_HEIGHT; m2++){
			for(int m3 = 0; m3 < MAX_ACT_WIDTH; m3++){
				cout << quantized_feature_map_out[m1][m2][m3] << ", ";
			}
			cout << endl;
		}
	}
	cout << "/////////////////////////////////" << endl;

//	Obtained Answer after selecting channels
	for(int i = 0; i < ker_shape[2]; i++){
		cout << "Selected channels: " << selected_ch[i] << endl;
	}

	float act_sub_mean;
	float acc_act_sub_mean[ker_shape[2]];
	for(int j1 = 0; j1 < ker_shape[2]; j1++){
		act_sub_mean = 0;
		for(int j2 = 0; j2 < MAX_ACT_HEIGHT; j2++){
			for(int j3 = 0; j3 < MAX_ACT_WIDTH; j3++){
				act_sub_mean += abs(feature_map[j1][j2][j3] - float_mean_scale * float_mean[j1]);
			}
		}
		acc_act_sub_mean[j1] = act_sub_mean;
		cout << "float act sub mean: " << act_sub_mean << endl;
	}

	float max_acc_act_sub_mean;
	for(int k = 0; k < ker_shape[2]; k++){
		if(k==0){
			max_acc_act_sub_mean = acc_act_sub_mean[k];
		}
		else{
			if(max_acc_act_sub_mean < acc_act_sub_mean[k]){
				max_acc_act_sub_mean = acc_act_sub_mean[k];
			}
		}
	}

	ap_uint<1> float_selected_ch[MAX_CHANNEL_SIZE] = {0};
	for(int l = 0; l < ker_shape[2]; l++){
		if(acc_act_sub_mean[l] < float_threshold * max_acc_act_sub_mean){
			float_selected_ch[l] = 0;
		}
		else{
			float_selected_ch[l] = 1;
		}
//	Expecting Answer after selecting channels
		cout << "Expected selected channels: " << float_selected_ch[l] << endl;
	}
}
