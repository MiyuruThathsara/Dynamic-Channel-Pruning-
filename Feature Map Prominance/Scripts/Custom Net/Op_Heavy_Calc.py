import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
from Custom_Net import Custom_Net

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ',device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
testset = torchvision.datasets.CIFAR10(root = '../../../../data', train = False, download = False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 0)

PATH = './custom_net_100_epochs.pth'
net = Custom_Net()
model = torch.load(PATH)
net.load_state_dict(model.state_dict())
net.eval()
net.to(device)

dataiter = iter(testloader)
data = dataiter.next()
images, labels = data[0].to(device), data[1].to(device)

num_conv0, num_decision0, num_conv1, num_decision1, num_conv2, num_decision2, num_conv3, num_decision3 = net.static_flop_calc(images)
print('Decision Calculation Operation Percentage: {} '.format(( num_decision0 + num_decision1 + num_decision2 + num_decision3 )/( num_conv0 + num_conv1 + num_conv2 + num_conv3 ) * 100)) 

# Dynamic Apporoach
# ABS Method
# Decision Calculation Operation Percentage: 0.38182814281641964%
# STD Method
# Decision Calculation Operation Percentage: 0.5101063640250856%


# Static Approach
# ABS Method
# Decision Calculation Operation Percentage: 0.2550531820125428%
# STD Method
# Decision Calculation Operation Percentage: 0.3833314032212087%