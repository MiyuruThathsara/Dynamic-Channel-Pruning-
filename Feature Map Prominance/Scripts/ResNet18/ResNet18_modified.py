import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.mean_scale = 1
        self.threshold = 0.9
        self.mean = []
        self.threshold_mask = []

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def masking_forward(self, x):
        x = self.activation_vol_masking(x,1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.activation_vol_masking(out,2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def abs_threshold_fixed_forward(self, x):
        count1, count1_list, map_count1, map_count1_list, mask_vec1 = self.abs_threshold_with_fixed_mean(x,1)
        out = F.relu(self.bn1(self.conv1(x)))
        count2, count2_list, map_count2, map_count2_list, mask_vec2 = self.abs_threshold_with_fixed_mean(out,2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        count = [ count1, count2 ]
        count_list = [ count1_list, count2_list ]
        map_count = [ map_count1, map_count2 ]
        map_count_list = [ map_count1_list, map_count2_list ]
        self.threshold_mask = [ mask_vec1, mask_vec2 ]
        return out, count, count_list, map_count, map_count_list

    def var_threshold_fixed_forward(self, x):
        count1, count1_list, map_count1, map_count1_list, mask_vec1 = self.var_threshold_with_fixed_mean(x,1)
        out = F.relu(self.bn1(self.conv1(x)))
        count2, count2_list, map_count2, map_count2_list, mask_vec2 = self.var_threshold_with_fixed_mean(out,2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        count = [ count1, count2 ]
        count_list = [ count1_list, count2_list ]
        map_count = [ map_count1, map_count2 ]
        map_count_list = [ map_count1_list, map_count2_list ]
        self.threshold_mask = [ mask_vec1, mask_vec2 ]
        return out, count, count_list, map_count, map_count_list
    def mean_calc_forward(self, x):
        run_mean1 = self.feature_map_running_mean(x)
        out = F.relu(self.bn1(self.conv1(x)))
        run_mean2 = self.feature_map_running_mean(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        running_mean_list = [ run_mean1, run_mean2 ]
        return out, running_mean_list
    
    def activation_vol_masking(self, x, layer_num):
        mask = self.threshold_mask[layer_num - 1].unsqueeze(dim = 2).unsqueeze(dim = 3)
        x = x * mask
        return x
    
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

    def abs_threshold(self, x):
        map_count = x.size()[0] * x.size()[1]
        x_abs = (x - x.mean(dim=(2,3), keepdim=True)).abs().sum(dim=(2,3))
        max_abs = x_abs.max(dim=1).values
        x[ x_abs <= max_abs.unsqueeze(dim=1) * 0.28 ] = 0
        count = ( x_abs <= max_abs.unsqueeze(dim=1) * 0.28 ).sum().item()
        return x, count, map_count

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential_Modified(layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def masking_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1.masking_forward(out)
        out = self.layer2.masking_forward(out)
        out = self.layer3.masking_forward(out)
        out = self.layer4.masking_forward(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def abs_threshold_fixed_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, count1, count1_list, map_count1, map_count1_list = self.layer1.abs_threshold_fixed_forward(out)
        out, count2, count2_list, map_count2, map_count2_list = self.layer2.abs_threshold_fixed_forward(out)
        out, count3, count3_list, map_count3, map_count3_list = self.layer3.abs_threshold_fixed_forward(out)
        out, count4, count4_list, map_count4, map_count4_list = self.layer4.abs_threshold_fixed_forward(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        count = [ count1, count2, count3, count4 ]
        count_list = [ *count1_list, *count2_list, *count3_list, *count4_list ]
        map_count = [ map_count1, map_count2, map_count3, map_count4 ]
        map_count_list = [ *map_count1_list, *map_count2_list, *map_count3_list, *map_count4_list ]
        return count, count_list, map_count, map_count_list

    def var_threshold_fixed_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, count1, count1_list, map_count1, map_count1_list = self.layer1.var_threshold_fixed_forward(out)
        out, count2, count2_list, map_count2, map_count2_list = self.layer2.var_threshold_fixed_forward(out)
        out, count3, count3_list, map_count3, map_count3_list = self.layer3.var_threshold_fixed_forward(out)
        out, count4, count4_list, map_count4, map_count4_list = self.layer4.var_threshold_fixed_forward(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        count = [ count1, count2, count3, count4 ]
        count_list = [ *count1_list, *count2_list, *count3_list, *count4_list ]
        map_count = [ map_count1, map_count2, map_count3, map_count4 ]
        map_count_list = [ *map_count1_list, *map_count2_list, *map_count3_list, *map_count4_list ]
        return count, count_list, map_count, map_count_list
        
    def mean_calc_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out, run_mean1 = self.layer1.mean_calc_forward(out)
        out, run_mean2 = self.layer2.mean_calc_forward(out)
        out, run_mean3 = self.layer3.mean_calc_forward(out)
        out, run_mean4 = self.layer4.mean_calc_forward(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        running_mean_list = [ run_mean1, run_mean2, run_mean3, run_mean4 ]
        return running_mean_list

    def static_mean_feed(self, mean_list):
        self.layer1.static_mean_feed(mean_list[0])
        self.layer2.static_mean_feed(mean_list[1])
        self.layer3.static_mean_feed(mean_list[2])
        self.layer4.static_mean_feed(mean_list[3])

    def threshold_feed(self, threshold):
        self.layer1.threshold_feed(threshold)
        self.layer2.threshold_feed(threshold)
        self.layer3.threshold_feed(threshold)
        self.layer4.threshold_feed(threshold)

    def mean_scale_feed(self, mean_scale):
        self.layer1.mean_scale_feed(mean_scale)
        self.layer2.mean_scale_feed(mean_scale)
        self.layer3.mean_scale_feed(mean_scale)
        self.layer4.mean_scale_feed(mean_scale)

    def threshold_access(self):
        return self.layer1.threshold_access()

    def mean_scale_access(self):
        return self.layer1.mean_scale_access()

class Sequential_Modified(nn.Sequential):
    def __init__(self, x):
        super().__init__(*x)
        self.sequential_blocks = x
    
    def mean_calc_forward(self, x):
        running_mean = []
        for i in range(len(self.sequential_blocks)):
            x, run_mean = self.sequential_blocks[i].mean_calc_forward(x)
            running_mean.append(run_mean)
        return x, running_mean

    def static_mean_feed(self, mean_list):
        for i in range(len(self.sequential_blocks)):
            self.sequential_blocks[i].mean = mean_list[i]
    
    def threshold_feed(self, threshold):
        for i in range(len(self.sequential_blocks)):
            self.sequential_blocks[i].threshold = threshold

    def mean_scale_feed(self, mean_scale):
        for i in range(len(self.sequential_blocks)):
            self.sequential_blocks[i].mean_scale = mean_scale 
    
    def threshold_access(self):
        return self.sequential_blocks[0].threshold

    def mean_scale_access(self):
        return self.sequential_blocks[0].mean_scale

    def masking_forward(self, x):
        for i in range(len(self.sequential_blocks)):
            x = self.sequential_blocks[i].masking_forward(x)
        return x

    def abs_threshold_fixed_forward(self, x):
        seq_count = []
        seq_count_list = []
        seq_map_count = []
        seq_map_count_list = []
        for i in range(len(self.sequential_blocks)):
            x, count, count_list, map_count, map_count_list = self.sequential_blocks[i].abs_threshold_fixed_forward(x)
            seq_count.append(count)
            seq_map_count.append(map_count)
            for length in range(len(count_list)):
                seq_count_list.append(count_list[length])
                seq_map_count_list.append(map_count_list[length])
        return x, seq_count, seq_count_list, seq_map_count, seq_map_count_list

    def var_threshold_fixed_forward(self, x):
        seq_count = []
        seq_count_list = []
        seq_map_count = []
        seq_map_count_list = []
        for i in range(len(self.sequential_blocks)):
            x, count, count_list, map_count, map_count_list = self.sequential_blocks[i].var_threshold_fixed_forward(x)
            seq_count.append(count)
            seq_map_count.append(map_count)
            for length in range(len(count_list)):
                seq_count_list.append(count_list[length])
                seq_map_count_list.append(map_count_list[length])
        return x, seq_count, seq_count_list, seq_map_count, seq_map_count_list

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])