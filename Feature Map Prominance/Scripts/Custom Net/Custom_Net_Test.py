import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from Custom_Net import Custom_Net

def forward_hook(self, input, output):
    print('*******************************************************')
    global out_features
    out_features = output.data
    # print(out_features.size())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
testset = torchvision.datasets.CIFAR10(root = '../../../../data', train = False, download = False, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = False, num_workers = 0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

PATH = './custom_net_100_epochs.pth'
net = Custom_Net()
model = torch.load(PATH)
net.load_state_dict(model.state_dict())
net.eval()
net.to(device)

# dataiter = iter(testloader)
# data = dataiter.next()
# images, labels = data[0].to(device), data[1].to(device)

# outputs = net(images)

# for k in range(4):
    # layer = list(net._modules.items())[k]
    # layer[1].register_forward_hook(forward_hook)

    # outputs = net(images)
    # plt.figure()

    # for j,im in enumerate(out_features,1):
    #     im_size = im.size()[1] * im.size()[2]
    #     mean = [ sum(sum(im[i]))/im_size for i in range(im.size()[0]) ]
    #     std = [ sum(sum((im[i] - mean[i])**2)) for i in range(len(mean)) ]
        # plt.plot(range(len(mean)), std)
        # plt.title('Standard Deviation Curve')
        # plt.xlabel('Index of the Feature Map')
        # plt.ylabel('Std value')
        # plt.savefig('Conv%1d.png' % (k + 1))
    
# correct = 0
# total = 0

# with torch.no_grad():
#     for i,data in enumerate(testloader,1):
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         # print('Accuracy = ', (predicted == labels).sum().item()/labels.size(0))
#         if(i%200 == 0):
#             print('Mini-Batch Count: %4d' % i)
#     print('Finished Testing!!!!!')


# print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))

# correct = 0.0
# total = 0.0
# tot_flop_drop_layer1 = 0
# tot_flop_drop_layer2 = 0
# tot_flop_drop_layer3 = 0
# tot_flop_drop_layer4 = 0

# with torch.no_grad():
#     for i, data in enumerate(testloader, 1):
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs, count1, map_count1, count2, map_count2, count3, map_count3 = net.dynamic_mean_forward(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         tot_flop_drop_layer1 += 1 - ( ( map_count1 - count1 ) / map_count1 )
#         tot_flop_drop_layer2 += 1 - ( ( map_count1 - count1 ) * ( map_count2 - count2 ) ) / ( map_count1 * map_count2 )
#         tot_flop_drop_layer3 += 1 - ( ( map_count2 - count2 ) * ( map_count3 - count3 ) ) / ( map_count2 * map_count3 )
#         tot_flop_drop_layer4 += 1 - ( ( map_count3 - count3 ) / map_count3 )
#         correct += (predicted == labels).sum().item()
#         if(i%200 == 0):
#             print('Mini-Batch Count: %4d' % i)
#     print('Finished Testing!!!!!!')
#     # print('Total Removed Feature Map Percentage as an Average:= ',tot_map_drop / i * 100)
#     print('Layer1 Total FLOP drop Percentage : %.2f %%' % ( tot_flop_drop_layer1 / i * 100 ) )
#     print('Layer2 Total FLOP drop Percentage : %.2f %%' % ( tot_flop_drop_layer2 / i * 100 ) )
#     print('Layer3 Total FLOP drop Percentage : %.2f %%' % ( tot_flop_drop_layer3 / i * 100 ) )
#     print('Layer4 Total FLOP drop Percentage : %.2f %%' % ( tot_flop_drop_layer4 / i * 100 ) )

# print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))