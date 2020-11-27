############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project Lenet_Accelerator
set_top top
add_files weights.cpp
add_files top.cpp
add_files max_pool.cpp
add_files fc.cpp
add_files convolution.cpp
add_files biases.cpp
add_files -tb tb.cpp -cflags "-Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb feature_map.h -cflags "-Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "Ultrasale_solution_with_dataflow"
set_part {xczu9eg-ffvb1156-2-i}
create_clock -period 10 -name default
csim_design -ldflags {-Wl,--stack,10485760} -clean
csynth_design
cosim_design -ldflags {-Wl,--stack,10485760} -trace_level all -compiled_library_dir "D:/Vivado/Vivado_Projects/Compiled_Libraries" -tool xsim
export_design -format ip_catalog
