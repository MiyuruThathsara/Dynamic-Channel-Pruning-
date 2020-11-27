#include "hls_stream.h"
#include "hls_video.h"
#include <iostream>

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

#define MP_WINDOW_SHAPE           2

using namespace hls;
using namespace std;

typedef struct{
	float data;
} ch_data;
