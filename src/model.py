
from torch.nn import Conv2d, ReLU, MaxPool2d, BatchNorm2d, AdaptiveAvgPool2d, Linear, Softmax
import torch.nn as nn
import torch
import numpy as np

class ContractingPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContractingPath, self).__init__()
        self.contracting = nn.Sequential(Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= 3, padding= 1),
                                            ReLU(),
                                            Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size= 3, padding= 1),
                                            ReLU(),
                                            MaxPool2d(kernel_size= 2, stride= 2))
    def forward(self, input):
        return self.contracting(input)

class UNet(nn.Module):

    def __init__(self, img_size = 572, num_classes= 102):
        super(UNet, self).__init__()

        # repeated application of two 3x3 convolutions (unpadded convolutions) with 64 out channels, 
        # each followed by a rectified linear unit (ReLU) 
        # and a 2x2 max pooling operation with stride 2 for downsampling
        # resulting features should be a 64x284x284 tensor 
        self.contracting_1 = ContractingPath(3, 64)
        self.contracting_2 = ContractingPath(64, 128)
        # self.mlp = nn.Sequential(nn.Linear(hidden_d, num_classes), nn.Softmax(dim = -1))
        
    def forward(self, input):

        out = self.contracting_1(input)

        return out

if __name__ == "__main__":
    import numpy as np
    import torch
    from torchsummary import summary

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(5, 3, 572, 572).astype('float32')
    X = torch.tensor(X)

    model = UNet()
    
    summary(model, (1, 3, 572, 572))
    print (model.forward(X).shape)