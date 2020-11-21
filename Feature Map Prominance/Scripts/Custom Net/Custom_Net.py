import torch 
import torch.nn as nn
import torch.nn.functional as F

class Custom_Net(nn.Module):

    def __init__(self):
        super(Custom_Net,self).__init__()
        self.conv1 = nn.Conv2d( 3, 6, kernel_size = 3, stride = 1, padding = 1 )
        self.conv2 = nn.Conv2d( 6, 16, kernel_size = 3, stride = 1, padding = 1 )
        self.conv3 = nn.Conv2d( 16, 32, kernel_size = 5, stride = 1, padding = 2 )
        self.conv4 = nn.Conv2d( 32, 32, kernel_size = 3, stride = 1, padding = 1 )
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.threshold = 0.24906243051590288
        self.mean_scale = 0.40000000000000013
        self.mean = []
        self.threshold_mask = []
        self.num_kernels = [ 6, 16, 32, 32 ]
        self.size_kernels = [ 3, 3, 5, 3 ]
    
    def masking_forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = self.activation_vol_masking(x,1)
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = self.activation_vol_masking(x,2)
        x = F.relu(self.conv3(x))
        x = self.activation_vol_masking(x,3)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.feature_map_flat(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # tot_count = count1 + count2 + count3
        # tot_map_count = map_count1 + map_count2 + map_count3
        return x

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.feature_map_flat(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # tot_count = count1 + count2 + count3
        # tot_map_count = map_count1 + map_count2 + map_count3
        return x

    def dynamic_mean_forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x, count1, map_count1 = self.var_threshold(x)
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x, count2, map_count2 = self.var_threshold(x)
        x = F.relu(self.conv3(x))
        x, count3, map_count3 = self.var_threshold(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.feature_map_flat(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # tot_count = count1 + count2 + count3
        # tot_map_count = map_count1 + map_count2 + map_count3
        return x, count1, map_count1, count2, map_count2, count3, map_count3

    def dynamic_flop_calc(self, x): # Run this for one image only
        num_conv0, num_decision0 = self.num_flop_calc(x, 0, method = 'std')
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        num_conv1, num_decision1 = self.num_flop_calc(x, 1, method = 'std')
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        num_conv2, num_decision2 = self.num_flop_calc(x, 2, method = 'std')
        x = F.relu(self.conv3(x))
        num_conv3, num_decision3 = self.num_flop_calc(x, 3, method = 'std')
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.feature_map_flat(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return num_conv0, num_decision0, num_conv1, num_decision1, num_conv2, num_decision2, num_conv3, num_decision3

    def static_flop_calc(self, x):
        num_conv0, num_decision0 = self.num_flop_calc(x, 0, type_calc='static', method = 'std')
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        num_conv1, num_decision1 = self.num_flop_calc(x, 1, type_calc='static', method = 'std')
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        num_conv2, num_decision2 = self.num_flop_calc(x, 2, type_calc='static', method = 'std')
        x = F.relu(self.conv3(x))
        num_conv3, num_decision3 = self.num_flop_calc(x, 3, type_calc='static', method = 'std')
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.feature_map_flat(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return num_conv0, num_decision0, num_conv1, num_decision1, num_conv2, num_decision2, num_conv3, num_decision3

    def activation_vol_masking(self, x, layer_num):
        mask = self.threshold_mask[layer_num - 1].unsqueeze(dim = 2).unsqueeze(dim = 3)
        x = x * mask
        return x

    def abs_threshold_forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        count1, map_count1, mask_vec1 = self.abs_threshold_with_fixed_mean(x,1)
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        count2, map_count2, mask_vec2 = self.abs_threshold_with_fixed_mean(x,2)
        x = F.relu(self.conv3(x))
        count3, map_count3, mask_vec3 = self.abs_threshold_with_fixed_mean(x,3)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.feature_map_flat(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # tot_count = count1 + count2 + count3
        # tot_map_count = map_count1 + map_count2 + map_count3
        self.threshold_mask = [ mask_vec1, mask_vec2, mask_vec3 ]
        return count1, map_count1, count2, map_count2, count3, map_count3
        # return x

    def abs_threshold_dynamic_forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        count1, map_count1, mask_vec1 = self.abs_threshold_with_dynamic_mean(x,1)
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        count2, map_count2, mask_vec2 = self.abs_threshold_with_dynamic_mean(x,2)
        x = F.relu(self.conv3(x))
        count3, map_count3, mask_vec3 = self.abs_threshold_with_dynamic_mean(x,3)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.feature_map_flat(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # tot_count = count1 + count2 + count3
        # tot_map_count = map_count1 + map_count2 + map_count3
        self.threshold_mask = [ mask_vec1, mask_vec2, mask_vec3 ]
        return count1, map_count1, count2, map_count2, count3, map_count3
        # return x

    def var_threshold_forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        count1, map_count1, mask_vec1 = self.var_threshold_with_fixed_mean(x,1)
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        count2, map_count2, mask_vec2 = self.var_threshold_with_fixed_mean(x,2)
        x = F.relu(self.conv3(x))
        count3, map_count3, mask_vec3 = self.var_threshold_with_fixed_mean(x,3)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.feature_map_flat(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # tot_count = count1 + count2 + count3
        # tot_map_count = map_count1 + map_count2 + map_count3
        self.threshold_mask = [ mask_vec1, mask_vec2, mask_vec3 ]
        return count1, map_count1, count2, map_count2, count3, map_count3
        # return x

    def var_threshold_dynamic_forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        count1, map_count1, mask_vec1 = self.var_threshold_with_dynamic_mean(x,1)
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        count2, map_count2, mask_vec2 = self.var_threshold_with_dynamic_mean(x,2)
        x = F.relu(self.conv3(x))
        count3, map_count3, mask_vec3 = self.var_threshold_with_dynamic_mean(x,3)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.feature_map_flat(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # tot_count = count1 + count2 + count3
        # tot_map_count = map_count1 + map_count2 + map_count3
        self.threshold_mask = [ mask_vec1, mask_vec2, mask_vec3 ]
        return count1, map_count1, count2, map_count2, count3, map_count3
        # return x

    def mean_calc_forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        run_mean1 = self.feature_map_running_mean(x)
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        run_mean2 = self.feature_map_running_mean(x)
        x = F.relu(self.conv3(x))
        run_mean3 = self.feature_map_running_mean(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.feature_map_flat(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        running_mean_list = [ run_mean1, run_mean2, run_mean3 ]
        return running_mean_list

    def feature_map_flat(self,x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

    def feature_map_running_mean(self, x):
        return x.mean(dim=(2,3)).sum(dim=0)

    def var_threshold(self, x):
        map_count = x.size()[0] * x.size()[1]
        var = x.var(dim=(2,3))
        max_var = var.max(dim=1).values
        x[ var <= max_var.unsqueeze(dim=1) * 0.03 ] = 0
        count = ( var <= max_var.unsqueeze(dim=1) * 0.03 ).sum().item()
        return x, count, map_count

    def var_threshold_with_fixed_mean(self, x, layer_num):
        map_count = x.size()[0] * x.size()[1]
        # map_count_list = [ x.size()[1] ] * x.size()[0]
        x_mean = self.mean[ layer_num - 1 ].unsqueeze(dim = 1).unsqueeze(dim = 2)
        var = ( ( x - self.mean_scale * x_mean ) ** 2 ).sum(dim=(2,3))
        max_var = var.max(dim=1).values
        count = ( var <= max_var.unsqueeze(dim=1) * self.threshold ).sum().item()
        # count_list = ( var <= max_var.unsqueeze(dim=1) * self.threshold ).sum(dim=1).tolist()
        var[ var <= max_var.unsqueeze(dim=1) * self.threshold ] = 0
        var[ var > 0 ] = 1
        mask_vec = var.type(torch.int)
        return count, map_count, mask_vec

    def var_threshold_with_dynamic_mean(self, x, layer_num):
        map_count = x.size()[0] * x.size()[1]
        var = x.var(dim=(2,3))
        max_var = var.max(dim=1).values
        count = ( var <= max_var.unsqueeze(dim=1) * 0.03 ).sum().item()
        var[ var <= max_var.unsqueeze(dim=1) * self.threshold ] = 0
        var[ var > 0 ] = 1
        mask_vec = var.type(torch.int)
        return count, map_count, mask_vec

    def abs_threshold(self, x):
        map_count = x.size()[0] * x.size()[1]
        x_abs = (x - x.mean(dim=(2,3), keepdim=True)).abs().sum(dim=(2,3))
        max_abs = x_abs.max(dim=1).values
        x[ x_abs <= max_abs.unsqueeze(dim=1) * 0.28 ] = 0
        count = ( x_abs <= max_abs.unsqueeze(dim=1) * 0.28 ).sum().item()
        return x, count, map_count

    def abs_threshold_with_fixed_mean(self, x, layer_num):
        map_count = x.size()[0] * x.size()[1]
        # map_count_list = [ x.size()[1] ] * x.size()[0]
        x_mean = self.mean[ layer_num - 1 ].unsqueeze(dim = 1).unsqueeze(dim = 2)
        x_abs = ( x - self.mean_scale * x_mean ).abs().sum(dim=(2,3))
        max_abs = x_abs.max(dim=1).values
        count = ( x_abs <= max_abs.unsqueeze(dim=1) * self.threshold ).sum().item()
        # count_list = ( x_abs <= max_abs.unsqueeze(dim=1) * self.threshold ).sum(dim=1).tolist()
        x_abs[ x_abs <= max_abs.unsqueeze(dim=1) * self.threshold ] = 0
        x_abs[ x_abs > 0 ] = 1
        mask_vec = x_abs.type(torch.int)
        return count, map_count, mask_vec

    def abs_threshold_with_dynamic_mean(self, x, layer_num):
        map_count = x.size()[0] * x.size()[1]
        x_abs = (x - x.mean(dim=(2,3), keepdim=True)).abs().sum(dim=(2,3))
        max_abs = x_abs.max(dim=1).values
        count = ( x_abs <= max_abs.unsqueeze(dim=1) * 0.28 ).sum().item()
        x_abs[ x_abs <= max_abs.unsqueeze(dim=1) * self.threshold ] = 0
        x_abs[ x_abs > 0 ] = 1
        mask_vec = x_abs.type(torch.int)
        return count, map_count, mask_vec

    def num_flop_calc(self, x, layer_num, type_calc='dynamic', method='abs'):
        kernel_size = self.size_kernels[layer_num]
        kernel_num = self.num_kernels[layer_num]
        num_of_convolutions = 2 * x.size()[0] * x.size()[1] * kernel_num * kernel_size * kernel_size * x.size()[2] * x.size()[3]
        decision_num_of_calcs = 0
        if(layer_num != 0):
            if(type_calc == 'dynamic'):
                if(method =='abs'):
                    decision_num_of_calcs = ( ( x.size()[2] * x.size()[3] - 1 ) + ( 2 * x.size()[2] * x.size()[3] - 1 ) ) * x.size()[0] * x.size()[1]
                elif(method == 'std'):
                    decision_num_of_calcs = ( ( x.size()[2] * x.size()[3] - 1 ) + ( 3 * x.size()[2] * x.size()[3] - 1 ) ) * x.size()[0] * x.size()[1]
            elif(type_calc == 'static'):
                if(method == 'abs'):
                    decision_num_of_calcs = ( 2 * x.size()[2] * x.size()[3] - 1 ) * x.size()[0] * x.size()[1]
                elif(method == 'std'):
                    decision_num_of_calcs = ( 3 * x.size()[2] * x.size()[3] - 1 ) * x.size()[0] * x.size()[1]
        return num_of_convolutions, decision_num_of_calcs
