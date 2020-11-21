# Data Architecture
# data = (['layer1', (fh,fw,ph,pw,i,o,s) ])

# num_of_layers = 4
# Custom Net data
# data = (['layer1', (32,32,3,3,3,6)], ['layer2', (16,16,3,3,6,16)], ['layer3', (8,8,5,5,16,32)], ['layer4', (8,8,3,3,32,32)])
# percent = [ 17.48, 22.86, 16.11, 10.22 ]

# VGG16 data
# data = (['layer1', (32,32,3,3,3,64)], ['layer2', (32,32,3,3,64,64)], ['layer3', (16,16,3,3,64,128)], ['layer4', (16,16,3,3,128,128)], ['layer5', (8,8,3,3,128,256)], ['layer6', (8,8,3,3,256,256)], ['layer7', (8,8,3,3,256,256)], ['layer8', (4,4,3,3,256,512)], ['layer9', (4,4,3,3,512,512)], ['layer10', (4,4,3,3,512,512)], ['layer11', (2,2,3,3,512,512)], ['layer12', (2,2,3,3,512,512)], ['layer13', (2,2,3,3,512,512)])
# percent = [7.480625000000001, 16.17141357421875, 22.439083251953125, 21.645259399414062, 23.105072631835938, 29.772737884521483, 26.96969955444336, 26.28944389343262, 25.218508644104006, 30.17079772949219, 36.16966938018799, 30.990462417602537, 15.067539062499998]

# ResNet18 data
Imagenet_data = (['conv_layer', (224,224,7,7,3,64,2)], ['layer1_block_0_conv1', (112,112,3,3,64,64,1)], ['layer1_block_0_conv_2', (112,112,3,3,64,64,1)], ['layer1_block_1_conv_1', (112,112,3,3,64,64,1)], ['layer1_block_1_conv_2', (112,112,3,3,64,64,1)], ['layer2_block_0_conv_1', (112,112,3,3,64,128,2)], ['int_layer1', (112,112,1,1,64,128,2)], ['layer2_block_0_conv_2', (56,56,3,3,128,128,1)], ['layer2_block_1_conv_1', (56,56,3,3,128,128,1)], ['layer2_block_1_conv_2', (56,56,3,3,128,128,1)], ['layer3_block_0_conv_1', (56,56,3,3,128,256,2)], ['int_layer2', (56,56,1,1,128,256,2)], ['layer3_block_0_conv_2', (28,28,3,3,256,256,1)], ['layer3_block_1_conv_1', (28,28,3,3,256,256,1)], ['layer3_block_1_conv_2', (28,28,3,3,256,256,1)], ['layer4_block_0_conv_1', (28,28,3,3,256,512,2)], ['int_layer3', (28,28,1,1,256,512,2)], ['layer4_block_0_conv_2', (14,14,3,3,512,512,1)], ['layer4_block_1_conv_1', (14,14,3,3,512,512,1)], ['layer4_block_1_conv_2', (14,14,3,3,512,512,1)])
# data = (['conv_layer', (32,32,3,3,3,64)], ['layer1', (32,32,3,3,64,64)], ['layer2', (32,32,3,3,64,64)], ['layer3', (32,32,3,3,64,64)], ['layer4', (32,32,3,3,64,64)], ['layer5', (32,32,3,3,64,128)], ['int_layer1', (32,32,1,1,64,128)], ['layer6', (16,16,3,3,128,128)], ['layer7', (16,16,3,3,128,128)], ['layer8', (16,16,3,3,128,128)], ['layer9', (16,16,3,3,128,256)], ['int_layer2', (16,16,1,1,128,256)], ['layer10', (8,8,3,3,256,256)], ['layer11', (8,8,3,3,256,256)], ['layer12', (8,8,3,3,256,256)], ['layer13', (8,8,3,3,256,512)], ['int_layer3', (8,8,1,1,256,512)], ['layer14', (4,4,3,3,512,512)], ['layer15', (4,4,3,3,512,512)], ['layer16', (4,4,3,3,512,512)])
# percent = [ 3.01, 4.00, 2.22, 0.98, 0.49, 0.16, 0.0, 0.05, 0.03, 0.02, 0.02, 0.0, 0.04, 0.04, 0.00, 0.05, 0.0, 0.29, 0.58, 0.34 ]

# print("Data array length: ",len(data))
# print("Percent array length: ", len(percent))

# tot_flop = 0
# tot_drop_flop = 0

# for i in range(num_of_layers):
#     flop = 2 * data[i][1][0] * data[i][1][1] * data[i][1][2] * data[i][1][3] * data[i][1][4] * data[i][1][5]
#     drop_flop = percent[i] * flop / 100
#     tot_flop += flop
#     tot_drop_flop += drop_flop

# flop_drop_percent = tot_drop_flop / tot_flop * 100

# print("Total FLOP Drop Percentage: ", flop_drop_percent)

# Number of Activation Maps Calculation
activations = []
tot_acts = 0
for i in Imagenet_data:
    if(i[0] != 'int_layer1' and i[0] !='int_layer2' and i[0] != 'int_layer3'):
        num_acts = i[1][0] * i[1][1] * i[1][4]
        tot_acts += num_acts
        activations.append([i[0],num_acts])
print('Layer by Layer Activations: {}'.format(activations))
print('Total Activations in ResNet18: {}'.format(tot_acts))