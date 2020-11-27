import math

def quant_dequant_scale_calc(act_max_list, weight_max_list, q_scheme):
    act_max_len = len(act_max_list)
    weight_max_len = len(weight_max_list)
    bias_quant_factors = []
    if(act_max_len != weight_max_len):
        raise Exception("Activation max list and Weight max list lengths should be equal!!!!")
    else:
        for i in range(act_max_len):
            if( i == act_max_len - 1 ):
                quant_dequant_scale = act_max_list[i] * weight_max_list[i] / ( 2 ** (q_scheme - 1) - 1 ) ** 2
            else:
                quant_dequant_scale = act_max_list[i] * weight_max_list[i] / ( ( 2 ** (q_scheme - 1) - 1 ) * act_max_list[i + 1] )
            bias_quant = ( ( 2 ** (q_scheme - 1) - 1 ) ** 2 ) / act_max_list[i] * weight_max_list[i]
            bias_quant_factors.append(bias_quant)
            print("conv/fc_layer" + str(i + 1) + " = " + str(quant_dequant_scale))
        print("Bias Quantization Factors : {}".format(bias_quant_factors))

act_max_list = [1.0, 2.0500648021698, 6.805856227874756, 22.459524154663086, 31.003461837768555]
weight_max_list = [0.6834233403205872, 0.8909913301467896, 1.0573877096176147, 0.7839701771736145, 1.012024164199829]
# num_add_list = [25, 150, 400, 120, 84]
q_scheme = 8

quant_dequant_scale_calc(act_max_list, weight_max_list, q_scheme)

# Results
# conv/fc_layer1 = 0.0026249346550221288
# conv/fc_layer2 = 0.0021132679956538058
# conv/fc_layer3 = 0.0025229738951628405
# conv/fc_layer4 = 0.004471839235828596
# conv/fc_layer5 = 0.0019453315489905772
# Bias Quantization Factors : [11022.93505603075, 7009.924344209723, 2505.8722660893027, 562.9974571392655, 526.4875848313935]