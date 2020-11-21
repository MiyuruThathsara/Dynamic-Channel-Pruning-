import torch
import torch.nn as nn
import torch.nn.functional as F

#################################################################################
# VGG16 Network
#################################################################################

class vgg16(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.max_pool_1 = nn.MaxPool2d( kernel_size=2, stride=2 )
        self.conv2_1 = nn.Conv2d( 64, 128, kernel_size=3, padding=1 )
        self.conv2_2 = nn.Conv2d( 128, 128, kernel_size=3, padding=1 )
        self.max_pool_2 = nn.MaxPool2d( kernel_size=2, stride=2 )
        self.conv3_1 = nn.Conv2d( 128, 256, kernel_size=3, padding=1 )
        self.conv3_2 = nn.Conv2d( 256, 256, kernel_size=3, padding=1 )
        self.conv3_3 = nn.Conv2d( 256, 256, kernel_size=3, padding=1 )
        self.max_pool_3 = nn.MaxPool2d( kernel_size=2, stride=2 )
        self.conv4_1 = nn.Conv2d( 256, 512, kernel_size=3, padding=1 )
        self.conv4_2 = nn.Conv2d( 512, 512, kernel_size=3, padding=1 )
        self.conv4_3 = nn.Conv2d( 512, 512, kernel_size=3, padding=1 )
        self.max_pool_4 = nn.MaxPool2d( kernel_size=2, stride=2 )
        self.conv5_1 = nn.Conv2d( 512, 512, kernel_size=3, padding=1 )
        self.conv5_2 = nn.Conv2d( 512, 512, kernel_size=3, padding=1 )
        self.conv5_3 = nn.Conv2d( 512, 512, kernel_size=3, padding=1 )
        self.max_pool_5 = nn.MaxPool2d( kernel_size=2, stride=2 )
        #########################################
        self.avg_pool = nn.AdaptiveAvgPool2d((7,7))
        #########################################
        self.fc1 = nn.Linear( 512 * 7 * 7, 4096 )
        self.fc2 = nn.Linear( 4096, 4096 )
        self.fc3 = nn.Linear( 4096, num_classes )
        ##########################################
        self.mean_scale = 1.2
        self.threshold = 0.235
        self.mean = []
        self.threshold_mask = []

        if(init_weights):
            self.initialize_weights()

    def forward(self, x):
        x = self.max_pool_1(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))))
        x = self.max_pool_2(F.relu(self.conv2_2(F.relu(self.conv2_1(x)))))
        x = self.max_pool_3(F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(x)))))))
        x = self.max_pool_4(F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(x)))))))
        x = self.max_pool_5(F.relu(self.conv5_3(F.relu(self.conv5_2(F.relu(self.conv5_1(x)))))))
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5)
        x = self.fc3(x)
        return x

    def masking_forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.activation_vol_masking(x,1)
        x = self.max_pool_1(F.relu(self.conv1_2(x)))
        x = self.activation_vol_masking(x,2)
        x = F.relu(self.conv2_1(x))
        x = self.activation_vol_masking(x,3)
        x = self.max_pool_2(F.relu(self.conv2_2(x)))
        x = self.activation_vol_masking(x,4)
        x = F.relu(self.conv3_1(x))
        x = self.activation_vol_masking(x,5)
        x = F.relu(self.conv3_2(x))
        x = self.activation_vol_masking(x,6)
        x = self.max_pool_3(F.relu(self.conv3_3(x)))
        x = self.activation_vol_masking(x,7)
        x = F.relu(self.conv4_1(x))
        x = self.activation_vol_masking(x,8)
        x = F.relu(self.conv4_2(x))
        x = self.activation_vol_masking(x,9)
        x = self.max_pool_4(F.relu(self.conv4_3(x)))
        x = self.activation_vol_masking(x,10)
        x = F.relu(self.conv5_1(x))
        x = self.activation_vol_masking(x,11)
        x = F.relu(self.conv5_2(x))
        x = self.activation_vol_masking(x,12)
        x = self.max_pool_5(F.relu(self.conv5_3(x)))
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5)
        x = self.fc3(x)
        # count = [ count1, count2, count3, count4, count5, count6, count7, count8, count9, count10, count11, count12 ]
        # map_count = [ map_count1, map_count2, map_count3, map_count4, map_count5, map_count6, map_count7, map_count8, map_count9, map_count10, map_count11, map_count12 ]
        # return x, count, map_count
        return x

    def abs_threshold_forward(self, x):
        x = F.relu(self.conv1_1(x))
        x, count1, map_count1= self.abs_threshold(x)
        x = self.max_pool_1(F.relu(self.conv1_2(x)))
        x, count2, map_count2 = self.abs_threshold(x)
        x = F.relu(self.conv2_1(x))
        x, count3, map_count3 = self.abs_threshold(x)
        x = self.max_pool_2(F.relu(self.conv2_2(x)))
        x, count4, map_count4 = self.abs_threshold(x)
        x = F.relu(self.conv3_1(x))
        x, count5, map_count5 = self.abs_threshold(x)
        x = F.relu(self.conv3_2(x))
        x, count6, map_count6 = self.abs_threshold(x)
        x = self.max_pool_3(F.relu(self.conv3_3(x)))
        x, count7, map_count7 = self.abs_threshold(x)
        x = F.relu(self.conv4_1(x))
        x, count8, map_count8 = self.abs_threshold(x)
        x = F.relu(self.conv4_2(x))
        x, count9, map_count9 = self.abs_threshold(x)
        x = self.max_pool_4(F.relu(self.conv4_3(x)))
        x, count10, map_count10 = self.abs_threshold(x)
        x = F.relu(self.conv5_1(x))
        x, count11, map_count11 = self.abs_threshold(x)
        x = F.relu(self.conv5_2(x))
        x, count12, map_count12 = self.abs_threshold(x)
        x = self.max_pool_5(F.relu(self.conv5_3(x)))
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5)
        x = self.fc3(x)
        ###########################################################################################
        count = [ count1, count2, count3, count4, count5, count6, count7, count8, count9, count10, count11, count12 ]
        map_count = [ map_count1, map_count2, map_count3, map_count4, map_count5, map_count6, map_count7, map_count8, map_count9, map_count10, map_count11, map_count12 ]
        return x, count, map_count

    def var_threshold_forward(self, x):
        x = F.relu(self.conv1_1(x))
        x, count1, map_count1= self.var_threshold(x)
        x = self.max_pool_1(F.relu(self.conv1_2(x)))
        x, count2, map_count2 = self.var_threshold(x)
        x = F.relu(self.conv2_1(x))
        x, count3, map_count3 = self.var_threshold(x)
        x = self.max_pool_2(F.relu(self.conv2_2(x)))
        x, count4, map_count4 = self.var_threshold(x)
        x = F.relu(self.conv3_1(x))
        x, count5, map_count5 = self.var_threshold(x)
        x = F.relu(self.conv3_2(x))
        x, count6, map_count6 = self.var_threshold(x)
        x = self.max_pool_3(F.relu(self.conv3_3(x)))
        x, count7, map_count7 = self.var_threshold(x)
        x = F.relu(self.conv4_1(x))
        x, count8, map_count8 = self.var_threshold(x)
        x = F.relu(self.conv4_2(x))
        x, count9, map_count9 = self.var_threshold(x)
        x = self.max_pool_4(F.relu(self.conv4_3(x)))
        x, count10, map_count10 = self.var_threshold(x)
        x = F.relu(self.conv5_1(x))
        x, count11, map_count11 = self.var_threshold(x)
        x = F.relu(self.conv5_2(x))
        x, count12, map_count12 = self.var_threshold(x)
        x = self.max_pool_5(F.relu(self.conv5_3(x)))
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5)
        x = self.fc3(x)
        ###########################################################################################
        count = [ count1, count2, count3, count4, count5, count6, count7, count8, count9, count10, count11, count12 ]
        map_count = [ map_count1, map_count2, map_count3, map_count4, map_count5, map_count6, map_count7, map_count8, map_count9, map_count10, map_count11, map_count12 ]
        return x, count, map_count

    def abs_threshold_fixed_forward_single_iter(self, x):
        x = F.relu(self.conv1_1(x))
        count1, count1_list, map_count1, map_count1_list = self.abs_threshold_with_fixed_mean_single_iter(x,1)
        x = self.max_pool_1(F.relu(self.conv1_2(x)))
        count2, count2_list, map_count2, map_count2_list = self.abs_threshold_with_fixed_mean_single_iter(x,2)
        x = F.relu(self.conv2_1(x))
        count3, count3_list, map_count3, map_count3_list = self.abs_threshold_with_fixed_mean_single_iter(x,3)
        x = self.max_pool_2(F.relu(self.conv2_2(x)))
        count4, count4_list, map_count4, map_count4_list = self.abs_threshold_with_fixed_mean_single_iter(x,4)
        x = F.relu(self.conv3_1(x))
        count5, count5_list, map_count5, map_count5_list = self.abs_threshold_with_fixed_mean_single_iter(x,5)
        x = F.relu(self.conv3_2(x))
        count6, count6_list, map_count6, map_count6_list = self.abs_threshold_with_fixed_mean_single_iter(x,6)
        x = self.max_pool_3(F.relu(self.conv3_3(x)))
        count7, count7_list, map_count7, map_count7_list = self.abs_threshold_with_fixed_mean_single_iter(x,7)
        x = F.relu(self.conv4_1(x))
        count8, count8_list, map_count8, map_count8_list = self.abs_threshold_with_fixed_mean_single_iter(x,8)
        x = F.relu(self.conv4_2(x))
        count9, count9_list, map_count9, map_count9_list = self.abs_threshold_with_fixed_mean_single_iter(x,9)
        x = self.max_pool_4(F.relu(self.conv4_3(x)))
        count10, count10_list, map_count10, map_count10_list = self.abs_threshold_with_fixed_mean_single_iter(x,10)
        x = F.relu(self.conv5_1(x))
        count11, count11_list, map_count11, map_count11_list = self.abs_threshold_with_fixed_mean_single_iter(x,11)
        x = F.relu(self.conv5_2(x))
        count12, count12_list, map_count12, map_count12_list = self.abs_threshold_with_fixed_mean_single_iter(x,12)
        x = self.max_pool_5(F.relu(self.conv5_3(x)))
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5)
        x = self.fc3(x)
        ###########################################################################################
        count = [ count1, count2, count3, count4, count5, count6, count7, count8, count9, count10, count11, count12 ]
        count_list = [ count1_list, count2_list, count3_list, count4_list, count5_list, count6_list, count7_list, count8_list, count9_list, count10_list, count11_list, count12_list ]
        map_count = [ map_count1, map_count2, map_count3, map_count4, map_count5, map_count6, map_count7, map_count8, map_count9, map_count10, map_count11, map_count12 ]
        map_count_list = [ map_count1_list, map_count2_list, map_count3_list, map_count4_list, map_count5_list, map_count6_list, map_count7_list, map_count8_list, map_count9_list, map_count10_list, map_count11_list, map_count12_list ]
        return x,count, count_list, map_count, map_count_list

    def var_threshold_fixed_forward_single_iter(self, x):
        x = F.relu(self.conv1_1(x))
        count1, count1_list, map_count1, map_count1_list = self.var_threshold_with_fixed_mean_single_iter(x,1)
        x = self.max_pool_1(F.relu(self.conv1_2(x)))
        count2, count2_list, map_count2, map_count2_list = self.var_threshold_with_fixed_mean_single_iter(x,2)
        x = F.relu(self.conv2_1(x))
        count3, count3_list, map_count3, map_count3_list = self.var_threshold_with_fixed_mean_single_iter(x,3)
        x = self.max_pool_2(F.relu(self.conv2_2(x)))
        count4, count4_list, map_count4, map_count4_list = self.var_threshold_with_fixed_mean_single_iter(x,4)
        x = F.relu(self.conv3_1(x))
        count5, count5_list, map_count5, map_count5_list = self.var_threshold_with_fixed_mean_single_iter(x,5)
        x = F.relu(self.conv3_2(x))
        count6, count6_list, map_count6, map_count6_list = self.var_threshold_with_fixed_mean_single_iter(x,6)
        x = self.max_pool_3(F.relu(self.conv3_3(x)))
        count7, count7_list, map_count7, map_count7_list = self.var_threshold_with_fixed_mean_single_iter(x,7)
        x = F.relu(self.conv4_1(x))
        count8, count8_list, map_count8, map_count8_list = self.var_threshold_with_fixed_mean_single_iter(x,8)
        x = F.relu(self.conv4_2(x))
        count9, count9_list, map_count9, map_count9_list = self.var_threshold_with_fixed_mean_single_iter(x,9)
        x = self.max_pool_4(F.relu(self.conv4_3(x)))
        count10, count10_list, map_count10, map_count10_list = self.var_threshold_with_fixed_mean_single_iter(x,10)
        x = F.relu(self.conv5_1(x))
        count11, count11_list, map_count11, map_count11_list = self.var_threshold_with_fixed_mean_single_iter(x,11)
        x = F.relu(self.conv5_2(x))
        count12, count12_list, map_count12, map_count12_list = self.var_threshold_with_fixed_mean_single_iter(x,12)
        x = self.max_pool_5(F.relu(self.conv5_3(x)))
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5)
        x = self.fc3(x)
        ###########################################################################################
        count = [ count1, count2, count3, count4, count5, count6, count7, count8, count9, count10, count11, count12 ]
        count_list = [ count1_list, count2_list, count3_list, count4_list, count5_list, count6_list, count7_list, count8_list, count9_list, count10_list, count11_list, count12_list ]
        map_count = [ map_count1, map_count2, map_count3, map_count4, map_count5, map_count6, map_count7, map_count8, map_count9, map_count10, map_count11, map_count12 ]
        map_count_list = [ map_count1_list, map_count2_list, map_count3_list, map_count4_list, map_count5_list, map_count6_list, map_count7_list, map_count8_list, map_count9_list, map_count10_list, map_count11_list, map_count12_list ]
        return x,count, count_list, map_count, map_count_list

    def abs_threshold_fixed_forward(self, x):
        x = F.relu(self.conv1_1(x))
        count1, count1_list, map_count1, map_count1_list, mask_vec1 = self.abs_threshold_with_fixed_mean(x,1)
        x = self.max_pool_1(F.relu(self.conv1_2(x)))
        count2, count2_list, map_count2, map_count2_list, mask_vec2 = self.abs_threshold_with_fixed_mean(x,2)
        x = F.relu(self.conv2_1(x))
        count3, count3_list, map_count3, map_count3_list, mask_vec3 = self.abs_threshold_with_fixed_mean(x,3)
        x = self.max_pool_2(F.relu(self.conv2_2(x)))
        count4, count4_list, map_count4, map_count4_list, mask_vec4 = self.abs_threshold_with_fixed_mean(x,4)
        x = F.relu(self.conv3_1(x))
        count5, count5_list, map_count5, map_count5_list, mask_vec5 = self.abs_threshold_with_fixed_mean(x,5)
        x = F.relu(self.conv3_2(x))
        count6, count6_list, map_count6, map_count6_list, mask_vec6 = self.abs_threshold_with_fixed_mean(x,6)
        x = self.max_pool_3(F.relu(self.conv3_3(x)))
        count7, count7_list, map_count7, map_count7_list, mask_vec7 = self.abs_threshold_with_fixed_mean(x,7)
        x = F.relu(self.conv4_1(x))
        count8, count8_list, map_count8, map_count8_list, mask_vec8 = self.abs_threshold_with_fixed_mean(x,8)
        x = F.relu(self.conv4_2(x))
        count9, count9_list, map_count9, map_count9_list, mask_vec9 = self.abs_threshold_with_fixed_mean(x,9)
        x = self.max_pool_4(F.relu(self.conv4_3(x)))
        count10, count10_list, map_count10, map_count10_list, mask_vec10 = self.abs_threshold_with_fixed_mean(x,10)
        x = F.relu(self.conv5_1(x))
        count11, count11_list, map_count11, map_count11_list, mask_vec11 = self.abs_threshold_with_fixed_mean(x,11)
        x = F.relu(self.conv5_2(x))
        count12, count12_list, map_count12, map_count12_list, mask_vec12 = self.abs_threshold_with_fixed_mean(x,12)
        x = self.max_pool_5(F.relu(self.conv5_3(x)))
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5)
        x = self.fc3(x)
        ###########################################################################################
        count = [ count1, count2, count3, count4, count5, count6, count7, count8, count9, count10, count11, count12 ]
        count_list = [ count1_list, count2_list, count3_list, count4_list, count5_list, count6_list, count7_list, count8_list, count9_list, count10_list, count11_list, count12_list ]
        map_count = [ map_count1, map_count2, map_count3, map_count4, map_count5, map_count6, map_count7, map_count8, map_count9, map_count10, map_count11, map_count12 ]
        map_count_list = [ map_count1_list, map_count2_list, map_count3_list, map_count4_list, map_count5_list, map_count6_list, map_count7_list, map_count8_list, map_count9_list, map_count10_list, map_count11_list, map_count12_list ]
        ###########################################################################################
        self.threshold_mask = [ mask_vec1, mask_vec2, mask_vec3, mask_vec4, mask_vec5, mask_vec6, mask_vec7, mask_vec8, mask_vec9, mask_vec10, mask_vec11, mask_vec12 ]
        return count, count_list, map_count, map_count_list
    
    def var_threshold_fixed_forward(self, x):
        x = F.relu(self.conv1_1(x))
        count1, count1_list, map_count1, map_count1_list, mask_vec1 = self.var_threshold_with_fixed_mean(x,1)
        x = self.max_pool_1(F.relu(self.conv1_2(x)))
        count2, count2_list, map_count2, map_count2_list, mask_vec2 = self.var_threshold_with_fixed_mean(x,2)
        x = F.relu(self.conv2_1(x))
        count3, count3_list, map_count3, map_count3_list, mask_vec3 = self.var_threshold_with_fixed_mean(x,3)
        x = self.max_pool_2(F.relu(self.conv2_2(x)))
        count4, count4_list, map_count4, map_count4_list, mask_vec4 = self.var_threshold_with_fixed_mean(x,4)
        x = F.relu(self.conv3_1(x))
        count5, count5_list, map_count5, map_count5_list, mask_vec5 = self.var_threshold_with_fixed_mean(x,5)
        x = F.relu(self.conv3_2(x))
        count6, count6_list, map_count6, map_count6_list, mask_vec6 = self.var_threshold_with_fixed_mean(x,6)
        x = self.max_pool_3(F.relu(self.conv3_3(x)))
        count7, count7_list, map_count7, map_count7_list, mask_vec7 = self.var_threshold_with_fixed_mean(x,7)
        x = F.relu(self.conv4_1(x))
        count8, count8_list, map_count8, map_count8_list, mask_vec8 = self.var_threshold_with_fixed_mean(x,8)
        x = F.relu(self.conv4_2(x))
        count9, count9_list, map_count9, map_count9_list, mask_vec9 = self.var_threshold_with_fixed_mean(x,9)
        x = self.max_pool_4(F.relu(self.conv4_3(x)))
        count10, count10_list, map_count10, map_count10_list, mask_vec10 = self.var_threshold_with_fixed_mean(x,10)
        x = F.relu(self.conv5_1(x))
        count11, count11_list, map_count11, map_count11_list, mask_vec11 = self.var_threshold_with_fixed_mean(x,11)
        x = F.relu(self.conv5_2(x))
        count12, count12_list, map_count12, map_count12_list, mask_vec12 = self.var_threshold_with_fixed_mean(x,12)
        x = self.max_pool_5(F.relu(self.conv5_3(x)))
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5)
        x = self.fc3(x)
        ###########################################################################################
        count = [ count1, count2, count3, count4, count5, count6, count7, count8, count9, count10, count11, count12 ]
        count_list = [ count1_list, count2_list, count3_list, count4_list, count5_list, count6_list, count7_list, count8_list, count9_list, count10_list, count11_list, count12_list ]
        map_count = [ map_count1, map_count2, map_count3, map_count4, map_count5, map_count6, map_count7, map_count8, map_count9, map_count10, map_count11, map_count12 ]
        map_count_list = [ map_count1_list, map_count2_list, map_count3_list, map_count4_list, map_count5_list, map_count6_list, map_count7_list, map_count8_list, map_count9_list, map_count10_list, map_count11_list, map_count12_list ]
        ###########################################################################################
        self.threshold_mask = [ mask_vec1, mask_vec2, mask_vec3, mask_vec4, mask_vec5, mask_vec6, mask_vec7, mask_vec8, mask_vec9, mask_vec10, mask_vec11, mask_vec12 ]
        return count, count_list, map_count, map_count_list

    def mean_calc_forward(self, x):
        x = F.relu(self.conv1_1(x))
        run_mean1 = self.feature_map_running_mean(x)
        x = self.max_pool_1(F.relu(self.conv1_2(x)))
        run_mean2 = self.feature_map_running_mean(x)
        x = F.relu(self.conv2_1(x))
        run_mean3 = self.feature_map_running_mean(x)
        x = self.max_pool_2(F.relu(self.conv2_2(x)))
        run_mean4 = self.feature_map_running_mean(x)
        x = F.relu(self.conv3_1(x))
        run_mean5 = self.feature_map_running_mean(x)
        x = F.relu(self.conv3_2(x))
        run_mean6 = self.feature_map_running_mean(x)
        x = self.max_pool_3(F.relu(self.conv3_3(x)))
        run_mean7 = self.feature_map_running_mean(x)
        x = F.relu(self.conv4_1(x))
        run_mean8 = self.feature_map_running_mean(x)
        x = F.relu(self.conv4_2(x))
        run_mean9 = self.feature_map_running_mean(x)
        x = self.max_pool_4(F.relu(self.conv4_3(x)))
        run_mean10 = self.feature_map_running_mean(x)
        x = F.relu(self.conv5_1(x))
        run_mean11 = self.feature_map_running_mean(x)
        x = F.relu(self.conv5_2(x))
        run_mean12 = self.feature_map_running_mean(x)
        x = self.max_pool_5(F.relu(self.conv5_3(x)))
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5)
        x = self.fc3(x)
        running_mean_list = [ run_mean1, run_mean2, run_mean3, run_mean4, run_mean5, run_mean6, run_mean7, run_mean8, run_mean9, run_mean10, run_mean11, run_mean12 ]
        return running_mean_list

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def activation_vol_masking(self, x, layer_num):
        mask = self.threshold_mask[layer_num - 1].unsqueeze(dim = 2).unsqueeze(dim = 3)
        x = x * mask
        return x

    def feature_map_running_mean(self, x):
        running_mean = torch.zeros(x.size(1)).cuda()
        for i,im in enumerate(x,1):
            running_mean += im.mean(dim=(1,2))
        return running_mean

    # Single Iteration Variance Methods
    ###################################################################################################
    def var_threshold(self, x):
        map_count = x.size()[0] * x.size()[1]
        var = x.var(dim=(2,3))
        max_var = var.max(dim=1).values
        x[ var <= max_var.unsqueeze(dim=1) * 0.03 ] = 0
        count = ( var <= max_var.unsqueeze(dim=1) * 0.03 ).sum().item()
        return x, count, map_count

    def var_threshold_with_fixed_mean_single_iter(self, x, layer_num):
        map_count = x.size()[0] * x.size()[1]
        map_count_list = [ x.size()[1] ] * x.size()[0]
        x_mean = self.mean[ layer_num - 1 ].unsqueeze(dim = 1).unsqueeze(dim = 2)
        var = ( ( x - self.mean_scale * x_mean ) ** 2 ).sum(dim=(2,3))
        max_var = var.max(dim=1).values
        count = ( var <= max_var.unsqueeze(dim=1) * self.threshold ).sum().item()
        count_list = ( var <= max_var.unsqueeze(dim=1) * self.threshold ).sum(dim=1).tolist()
        x[ var <= max_var.unsqueeze(dim=1) * self.threshold ] = 0
        return count, count_list, map_count, map_count_list
    ###################################################################################################
    # Two Iteration Variance Methods
    ###################################################################################################
    def var_threshold_with_fixed_mean(self, x, layer_num):
        map_count = x.size()[0] * x.size()[1]
        map_count_list = [ x.size()[1] ] * x.size()[0]
        x_mean = self.mean[ layer_num - 1 ].unsqueeze(dim = 1).unsqueeze(dim = 2)
        var = ( ( x - self.mean_scale * x_mean ) ** 2 ).sum(dim=(2,3))
        max_var = var.max(dim=1).values
        count = ( var <= max_var.unsqueeze(dim=1) * self.threshold ).sum().item()
        count_list = ( var <= max_var.unsqueeze(dim=1) * self.threshold ).sum(dim=1).tolist()
        var[ var <= max_var.unsqueeze(dim=1) * self.threshold ] = 0
        var[ var > 0 ] = 1
        mask_vec = var.type(torch.int)
        return count, count_list, map_count, map_count_list, mask_vec

    def var_threshold_with_dynamic_mean(self, x, layer_num):
        map_count = x.size()[0] * x.size()[1]
        var = x.var(dim=(2,3))
        max_var = var.max(dim=1).values
        count = ( var <= max_var.unsqueeze(dim=1) * 0.03 ).sum().item()
        var[ var <= max_var.unsqueeze(dim=1) * self.threshold ] = 0
        var[ var > 0 ] = 1
        mask_vec = var.type(torch.int)
        return count, map_count, mask_vec
    ###################################################################################################
    # Single Iteration ABS Methods
    ###################################################################################################
    def abs_threshold(self, x):
        map_count = x.size()[0] * x.size()[1]
        x_abs = (x - x.mean(dim=(2,3), keepdim=True)).abs().sum(dim=(2,3))
        max_abs = x_abs.max(dim=1).values
        x[ x_abs <= max_abs.unsqueeze(dim=1) * 0.28 ] = 0
        count = ( x_abs <= max_abs.unsqueeze(dim=1) * 0.28 ).sum().item()
        return x, count, map_count

    def abs_threshold_with_fixed_mean_single_iter(self, x, layer_num):
        map_count = x.size()[0] * x.size()[1]
        map_count_list = [ x.size()[1] ] * x.size()[0]
        x_mean = self.mean[ layer_num - 1 ].unsqueeze(dim = 1).unsqueeze(dim = 2)
        x_abs = ( x - self.mean_scale * x_mean ).abs().sum(dim=(2,3))
        max_abs = x_abs.max(dim=1).values
        count = ( x_abs <= max_abs.unsqueeze(dim=1) * self.threshold ).sum().item()
        count_list = ( x_abs <= max_abs.unsqueeze(dim=1) * self.threshold ).sum(dim=1).tolist()
        x[ x_abs <= max_abs.unsqueeze(dim=1) * self.threshold ] = 0
        return count, count_list, map_count, map_count_list
    ###################################################################################################
    # Two Iteration ABS Methods
    ###################################################################################################
    def abs_threshold_with_fixed_mean(self, x, layer_num):
        map_count = x.size()[0] * x.size()[1]
        map_count_list = [ x.size()[1] ] * x.size()[0]
        x_mean = self.mean[ layer_num - 1 ].unsqueeze(dim = 1).unsqueeze(dim = 2)
        x_abs = ( x - self.mean_scale * x_mean ).abs().sum(dim=(2,3))
        max_abs = x_abs.max(dim=1).values
        count = ( x_abs <= max_abs.unsqueeze(dim=1) * self.threshold ).sum().item()
        count_list = ( x_abs <= max_abs.unsqueeze(dim=1) * self.threshold ).sum(dim=1).tolist()
        x_abs[ x_abs <= max_abs.unsqueeze(dim=1) * self.threshold ] = 0
        x_abs[ x_abs > 0 ] = 1
        mask_vec = x_abs.type(torch.int)
        return count, count_list, map_count, map_count_list, mask_vec

    def abs_threshold_with_dynamic_mean(self, x, layer_num):
        map_count = x.size()[0] * x.size()[1]
        x_abs = (x - x.mean(dim=(2,3), keepdim=True)).abs().sum(dim=(2,3))
        max_abs = x_abs.max(dim=1).values
        count = ( x_abs <= max_abs.unsqueeze(dim=1) * 0.28 ).sum().item()
        x_abs[ x_abs <= max_abs.unsqueeze(dim=1) * self.threshold ] = 0
        x_abs[ x_abs > 0 ] = 1
        mask_vec = x_abs.type(torch.int)
        return count, map_count, mask_vec
    ####################################################################################################