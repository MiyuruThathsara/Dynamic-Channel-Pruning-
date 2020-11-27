#include <stdio.h>
#include <string>
#include "params.h"

using namespace std;

void top(
		stream<ch_data> &in_stream,
		stream<out_data> &out_stream);

void feature_map(q_data feat_map[][REQ_FEATURE_MAP_IN_HW][REQ_FEATURE_MAP_IN_HW]);

int main(){
	out_data output[ NUM_OUT_NEURONES ] = {0};
	q_data image[MAX_NUM_OF_KERNELS][CONV1_IN_HW][CONV1_IN_HW] = {0};

	stream<ch_data> in_stream("In_Stream");
	stream<out_data> out_stream("Out_Stream");

	ch_data pixel_data;

	feature_map(image);

	for(int i1 = 0; i1 < IMAGE_IN_DEPTH; i1++){
		for(int i2 = 0; i2 < REQ_FEATURE_MAP_IN_HW; i2++){
			for(int i3 = 0; i3 < REQ_FEATURE_MAP_IN_HW; i3++){
				pixel_data.data = image[i1][i2][i3];
				in_stream.write(pixel_data);
			}
		}
	}

	top(in_stream, out_stream);

	for( int i = 0; i < NUM_OUT_NEURONES; i++ ){
		while(out_stream.empty());
		output[i] = out_stream.read();
		cout << output[i] << endl;
	}
	return 0;
}
