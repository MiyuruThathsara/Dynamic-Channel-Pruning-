#include "hls_stream.h"
#include "hls_video.h"
#include <iostream>
#include "ap_int.h"

/////////////////////////////////////////////////////////////////////////////////////
// LOG2 based on template meta-programming
template<int N,int OFFSET,int L2>
class HLS_LOG2_INT {
public:
    static const int value = HLS_LOG2_INT<(N>>1),((N&1)|OFFSET),(L2+1)>::value;
};

template<int OFFSET,int L2>
class HLS_LOG2_INT<1,OFFSET,L2> {
public:
    static const int value = L2+OFFSET;
};

template<int OFFSET,int L2>
class HLS_LOG2_INT<0,OFFSET,L2> {
public:
    static const int value = -9999999;
};

#define HLS_LOG2(N) (HLS_LOG2_INT<(N),0,0>::value)
////////////////////////////////////////////////////////////////////////////////////

#define QUANT_SCHEME              8
#define QUANT_SCHEME_MAX        pow(2, QUANT_SCHEME - 1) - 1

#define KERNEL_SHAPE 		 	  5
#define REQ_KERNEL_DEPTH 		 16
#define MAX_NUM_OF_KERNELS 		120
#define MAX_NUM_OF_NEURONES      84

#define IMAGE_IN_DEPTH 			  1
#define CONV1_NUM_KERNELS		  6
#define CONV2_NUM_KERNELS        16
#define CONV3_NUM_KERNELS       120

#define REQ_FEATURE_MAP_IN_HW 	 32
#define REQ_FEATURE_MAP_OUT_HW   28
#define REQ_FEATURE_MAP_DEPTH   MAX_NUM_OF_KERNELS
#define FC1_NUM_NEURONES        120
#define FC2_NUM_NEURONES         84
#define NUM_OUT_NEURONES         10

#define CONV1_IN_HW				REQ_FEATURE_MAP_IN_HW
#define CONV2_IN_HW				( CONV1_IN_HW - ( KERNEL_SHAPE - 1 ) ) / 2
#define CONV3_IN_HW				( CONV2_IN_HW - ( KERNEL_SHAPE - 1 ) ) / 2
#define CONV1_IN_DEPTH          IMAGE_IN_DEPTH
#define CONV2_IN_DEPTH			CONV1_NUM_KERNELS
#define CONV3_IN_DEPTH			CONV2_NUM_KERNELS

#define MAX_POOL1_IN_HW			CONV1_IN_HW - ( KERNEL_SHAPE - 1 )
#define MAX_POOL2_IN_HW			CONV2_IN_HW - ( KERNEL_SHAPE - 1 )
#define MAX_POOL1_IN_DEPTH		CONV1_NUM_KERNELS
#define MAX_POOL2_IN_DEPTH		CONV2_NUM_KERNELS

#define CONV1_QUANT_DEQUANT		  0.0026249346550221288
#define CONV2_QUANT_DEQUANT       0.0021132679956538058
#define CONV3_QUANT_DEQUANT       0.0025229738951628405
#define FC1_QUANT_DEQUANT         0.004471839235828596
#define FC2_QUANT_DEQUANT         0.0019453315489905772

#define MP_WINDOW_SHAPE           2

#define ACCUMILATOR_WIDTH        25

using namespace hls;
using namespace std;

typedef ap_int<QUANT_SCHEME> q_data;
typedef ap_int<ACCUMILATOR_WIDTH> acc_data;
typedef ap_fixed<QUANT_SCHEME + 1, QUANT_SCHEME, AP_RND> out_data;

typedef struct{
	q_data data;
} ch_data;
