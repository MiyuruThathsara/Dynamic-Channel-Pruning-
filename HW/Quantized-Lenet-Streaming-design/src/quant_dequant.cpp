#include "params.h"

ap_int<QUANT_SCHEME> quant_dequant_val(
		ap_int<ACCUMILATOR_WIDTH> val_in,
//		Here scale is multiplication of both quantization and de-quantization scale factors
		ap_fixed<16,0,AP_RND> scale){
	ap_fixed<QUANT_SCHEME + 1, QUANT_SCHEME, AP_RND> inter_val;
	ap_int<QUANT_SCHEME> val_out;

	/////////////////////////////////////////////////////////////////////////////////////////
	inter_val = (ap_fixed<QUANT_SCHEME + 1, QUANT_SCHEME, AP_RND>)( (ap_fixed<ACCUMILATOR_WIDTH,ACCUMILATOR_WIDTH,AP_RND>)(val_in) * scale );
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

ap_fixed<QUANT_SCHEME + 1, QUANT_SCHEME, AP_RND> dequant_val(
		ap_int<ACCUMILATOR_WIDTH> val_in,
		ap_fixed<16,0,AP_RND> scale){

	ap_fixed<QUANT_SCHEME + 1, QUANT_SCHEME, AP_RND> val_out;

	val_out = (ap_fixed<QUANT_SCHEME + 1, QUANT_SCHEME, AP_RND>)( (ap_fixed<ACCUMILATOR_WIDTH,ACCUMILATOR_WIDTH,AP_RND>)(val_in) * scale );

	return val_out;
}
