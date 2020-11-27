#include "params.h"

void convolution(
		float in_ch[][REQ_FEATURE_MAP_IN_HW][REQ_FEATURE_MAP_IN_HW],
		float out_ch[][REQ_FEATURE_MAP_OUT_HW][REQ_FEATURE_MAP_OUT_HW],
		float weight_kernel[][REQ_KERNEL_DEPTH][KERNEL_SHAPE][KERNEL_SHAPE],
		float bias_kernel[],
		int in_ch_shape[],
		int kernel_shape[],
		int pad_size,
		int stride_size);

void max_pool(
		float in_ch[][REQ_FEATURE_MAP_OUT_HW][REQ_FEATURE_MAP_OUT_HW],
		float out_ch[][REQ_FEATURE_MAP_IN_HW][REQ_FEATURE_MAP_IN_HW],
		int in_ch_shape[],
		int kernel_size,
		int stride_size);

void fc_type1(
		float in_neur[][REQ_FEATURE_MAP_OUT_HW][REQ_FEATURE_MAP_OUT_HW],
		float out_neur[],
		float weight[][ FC1_NUM_NEURONES ],
		float bias[],
		int in_size = FC1_NUM_NEURONES,
		int out_size = FC2_NUM_NEURONES);

void fc_type2(
		float in_neur[],
		float out_neur[],
		float weight[][ FC2_NUM_NEURONES ],
		float bias[],
		int in_size = FC2_NUM_NEURONES,
		int out_size = NUM_OUT_NEURONES);

void conv1_weights(float weights_out[][REQ_KERNEL_DEPTH][KERNEL_SHAPE][KERNEL_SHAPE]);
void conv2_weights(float weights_out[][REQ_KERNEL_DEPTH][KERNEL_SHAPE][KERNEL_SHAPE]);
void conv3_weights(float weights_out[][REQ_KERNEL_DEPTH][KERNEL_SHAPE][KERNEL_SHAPE]);
void fc1_weights(float weights_out[][FC1_NUM_NEURONES]);
void fc2_weights(float weights_out[][FC2_NUM_NEURONES]);

void conv1_bias(float biases_out[MAX_NUM_OF_KERNELS]);
void conv2_bias(float biases_out[MAX_NUM_OF_KERNELS]);
void conv3_bias(float biases_out[MAX_NUM_OF_KERNELS]);
void fc1_bias(float biases_out[MAX_NUM_OF_NEURONES]);
void fc2_bias(float biases_out[MAX_NUM_OF_NEURONES]);

void top(
		stream<ch_data> &in_stream,
		stream<float> &out_stream)
{
#pragma HLS interface axis register both port=in_stream
#pragma HLS interface axis register both port=outstream

	float layer_even_in[MAX_NUM_OF_KERNELS][REQ_FEATURE_MAP_OUT_HW][REQ_FEATURE_MAP_OUT_HW];
	float layer_odd_in[MAX_NUM_OF_KERNELS][REQ_FEATURE_MAP_IN_HW][REQ_FEATURE_MAP_IN_HW];
	float out_neur[NUM_OUT_NEURONES];

	float fc1_out[ FC2_NUM_NEURONES ];

	float conv1_parameters[MAX_NUM_OF_KERNELS][REQ_KERNEL_DEPTH][KERNEL_SHAPE][KERNEL_SHAPE];
	float conv2_parameters[MAX_NUM_OF_KERNELS][REQ_KERNEL_DEPTH][KERNEL_SHAPE][KERNEL_SHAPE];
	float conv3_parameters[MAX_NUM_OF_KERNELS][REQ_KERNEL_DEPTH][KERNEL_SHAPE][KERNEL_SHAPE];
	float fc1_parameters[FC2_NUM_NEURONES][FC1_NUM_NEURONES];
	float fc2_parameters[NUM_OUT_NEURONES][FC2_NUM_NEURONES];

	float conv1_biases[MAX_NUM_OF_KERNELS];
	float conv2_biases[MAX_NUM_OF_KERNELS];
	float conv3_biases[MAX_NUM_OF_KERNELS];
	float fc1_biases[MAX_NUM_OF_NEURONES];
	float fc2_biases[NUM_OUT_NEURONES];

	int conv1_in_shape[ 3 ] = { CONV1_IN_HW, CONV1_IN_HW, CONV1_IN_DEPTH };
	int conv2_in_shape[ 3 ] = { CONV2_IN_HW, CONV2_IN_HW, CONV2_IN_DEPTH };
	int conv3_in_shape[ 3 ] = { CONV3_IN_HW, CONV3_IN_HW, CONV3_IN_DEPTH };

	int max_pool1_in_shape[ 3 ] = { MAX_POOL1_IN_HW, MAX_POOL1_IN_HW, MAX_POOL1_IN_DEPTH };
	int max_pool2_in_shape[ 3 ] = { MAX_POOL2_IN_HW, MAX_POOL2_IN_HW, MAX_POOL2_IN_DEPTH };

	int conv1_ker_shape[ 3 ] = { KERNEL_SHAPE, KERNEL_SHAPE, CONV1_NUM_KERNELS };
	int conv2_ker_shape[ 3 ] = { KERNEL_SHAPE, KERNEL_SHAPE, CONV2_NUM_KERNELS };
	int conv3_ker_shape[ 3 ] = { KERNEL_SHAPE, KERNEL_SHAPE, CONV3_NUM_KERNELS };

#pragma HLS array_map variable=conv1_in_shape instance=params horizontal
#pragma HLS array_map variable=conv2_in_shape instance=params horizontal
#pragma HLS array_map variable=conv3_in_shape instance=params horizontal
#pragma HLS array_map variable=max_pool1_in_shape instance=params horizontal
#pragma HLS array_map variable=max_pool2_in_shape instance=params horizontal
#pragma HLS array_map variable=conv1_ker_shape instance=params horizontal
#pragma HLS array_map variable=conv2_ker_shape instance=params horizontal
#pragma HLS array_map variable=conv3_ker_shape instance=params horizontal

	int stride_size = 1;
	int pad_size = 0;

	conv1_weights(conv1_parameters);
	conv2_weights(conv2_parameters);
	conv3_weights(conv3_parameters);
	fc1_weights(fc1_parameters);
	fc2_weights(fc2_parameters);

	conv1_bias(conv1_biases);
	conv2_bias(conv2_biases);
	conv3_bias(conv3_biases);
	fc1_bias(fc1_biases);
	fc2_bias(fc2_biases);

	ch_data pixel_data;
	float out_neurone_val;

	for(unsigned int i1 = 0; i1 < IMAGE_IN_DEPTH; i1++){
		for(unsigned int i2 = 0; i2 < REQ_FEATURE_MAP_IN_HW; i2++){
			for(unsigned int i3 = 0; i3 < REQ_FEATURE_MAP_IN_HW; i3++){
				while(in_stream.empty());
				pixel_data = in_stream.read();
				layer_odd_in[i1][i2][i3] = pixel_data.data;
			}
		}
	}


	convolution( layer_odd_in, layer_even_in, conv1_parameters, conv1_biases, conv1_in_shape, conv1_ker_shape, pad_size, stride_size );
	max_pool( layer_even_in, layer_odd_in, max_pool1_in_shape, MP_WINDOW_SHAPE, MP_WINDOW_SHAPE );
	convolution( layer_odd_in, layer_even_in, conv2_parameters, conv2_biases, conv2_in_shape, conv2_ker_shape, pad_size, stride_size );
	max_pool( layer_even_in, layer_odd_in, max_pool2_in_shape, MP_WINDOW_SHAPE, MP_WINDOW_SHAPE );
	convolution( layer_odd_in, layer_even_in, conv3_parameters, conv3_biases, conv3_in_shape, conv3_ker_shape, pad_size, stride_size );
	fc_type1( layer_even_in, fc1_out, fc1_parameters, fc1_biases );
	fc_type2( fc1_out, out_neur, fc2_parameters, fc2_biases );

	for(unsigned int j = 0; j < NUM_OUT_NEURONES; j++){
		out_neurone_val = out_neur[j];
		out_stream.write(out_neurone_val);
	}

}
