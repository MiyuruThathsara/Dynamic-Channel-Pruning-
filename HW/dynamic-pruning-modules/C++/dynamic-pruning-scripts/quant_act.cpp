#include "params.h"

ap_int<QUANT_SCHEME> quant_val(
		ap_int<QUANT_SCHEME> val_in,
//		Here scale multiplication of both quantization and de-quantization scale factors
		ap_fixed<16,8,AP_RND> scale){
	ap_fixed<QUANT_SCHEME + 1, QUANT_SCHEME, AP_RND> inter_val;
	ap_int<QUANT_SCHEME> val_out;

	/////////////////////////////////////////////////////////////////////////////////////////
	inter_val = (ap_fixed<QUANT_SCHEME + 1, QUANT_SCHEME, AP_RND>)( (ap_fixed<16,8,AP_RND>)(val_in) * scale );
	/////////////////////////////////////////////////////////////////////////////////////////

	if(val_in > 0 && inter_val.range(QUANT_SCHEME,QUANT_SCHEME) > 0){
		val_out = QUANT_SCHEME_MAX;
	}
	else if(val_in < 0 && inter_val.range(QUANT_SCHEME,QUANT_SCHEME) == 0){
		val_out = -QUANT_SCHEME_MAX;
	}
	else {
		if(inter_val.range(0,0) > 0){
			val_out = inter_val.range(QUANT_SCHEME,1) + (ap_int<QUANT_SCHEME>)1;
		}
		else{
			val_out = inter_val.range(QUANT_SCHEME,1);
		}
	}
	return val_out;
}

void quant_dequant_act(
                      ap_int<QUANT_SCHEME> in_ch[][MAX_ACT_HEIGHT][MAX_ACT_WIDTH],
                      ap_int<QUANT_SCHEME> out_ch[][MAX_ACT_HEIGHT][MAX_ACT_WIDTH],
					  ap_fixed<16,8,AP_RND> range_scale){

    for(int i1=0; i1<=3; i1++){
        for(int i2=0; i2<=MAX_ACT_HEIGHT; i2++){
            for(int i3=0; i3<=MAX_ACT_WIDTH; i3++){
                out_ch[i1][i2][i3] = quant_val(in_ch[i1][i2][i3], range_scale);
            }
        }
    }
}
