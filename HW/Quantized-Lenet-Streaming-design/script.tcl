############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project Quantized_Lenet_Accelerator
set_top top
add_files biases.cpp
add_files convolution.cpp
add_files fc.cpp
add_files max_pool.cpp
add_files params.h
add_files quant_dequant.cpp
add_files top.cpp
add_files weights.cpp
add_files -tb feature_map.cpp
add_files -tb tb.cpp
open_solution "solution1"
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 10 -name default
#source "./Quantized_Lenet_Accelerator/solution1/directives.tcl"
csim_design -ldflags {-Wl,--stack,10485760} -clean
csynth_design
cosim_design -ldflags {-Wl,--stack,10485760}
export_design -format ip_catalog
