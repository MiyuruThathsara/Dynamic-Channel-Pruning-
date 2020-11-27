#include "feature_map.h"
#include <stdio.h>
#include <string>
#include "params.h"

using namespace std;

void top(
		stream<ch_data> &in_stream,
		stream<float> &out_stream);

int main(){
	float output[ 10 ] = {0};
	float image[120][32][32] = {0};

	stream<ch_data> in_stream("In_Stream");
	stream<float> out_stream("Out_Stream");

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
