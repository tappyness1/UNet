
from torch.nn import Conv2d, ReLU, MaxPool2d, ConvTranspose2d, BatchNorm2d, AdaptiveAvgPool2d, Linear, Softmax
import torch.nn as nn
import torch
import numpy as np

class ContractingPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContractingPath, self).__init__()
        self.contracting = nn.Sequential(Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= 3, padding= 0),
                                         BatchNorm2d(out_channels),
                                         ReLU(),
                                         Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size= 3, padding= 0),
                                         BatchNorm2d(out_channels),
                                         ReLU())
                                        
    def forward(self, input):
        return self.contracting(input)
    
class ExpandingPath(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExpandingPath, self).__init__()
        self.upsampling = ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 2, padding = 0, stride = 2)

        self.relu = ReLU()
        
        self.convolution = ContractingPath(in_channels, out_channels)

    def forward(self, input, to_concat_x):

        out = self.upsampling(input)
        out = self.relu(out)

        # need to concatenate the feature maps from the contracting path
        out = torch.cat((out, to_concat_x), dim = 1)
        out = self.convolution(out)

        return out
    
def clone_and_crop(out, cropped_size):
    feat_map = torch.clone(out)
    return feat_map.narrow(2, 0, cropped_size).narrow(3, 0, cropped_size)

class UNet(nn.Module):

    def __init__(self, img_size = 572, num_classes= 20):
        super(UNet, self).__init__()

        # repeated application of two 3x3 convolutions (unpadded convolutions) with 64 out channels, 
        # each followed by a rectified linear unit (ReLU) 
        # and a 2x2 max pooling operation with stride 2 for downsampling
        # resulting features should be a 64x284x284 tensor 
        self.contracting_1 = ContractingPath(3, 64)
        self.maxpool_contracting = MaxPool2d(kernel_size= 2, stride= 2)
        self.contracting_2 = ContractingPath(64, 128)
        self.contracting_3 = ContractingPath(128, 256)
        self.contracting_4 = ContractingPath(256, 512)
        self.contracting_5 = ContractingPath(512, 1024)

        # expanding part of the path
        self.expanding_1 = ExpandingPath(1024, 512)
        self.expanding_2 = ExpandingPath(512, 256)
        self.expanding_3 = ExpandingPath(256, 128)
        self.expanding_4 = ExpandingPath(128, 64)

        # final 1x1 convolution to map each 64-component feature vector to the desired number of classes
        self.class_mapping = Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, padding=0)
        
    def forward(self, input):

        out = self.contracting_1(input)
        feat_map_1_cropped = clone_and_crop(out, 392)
        out = self.maxpool_contracting(out)

        out = self.contracting_2(out)
        feat_map_2_cropped = clone_and_crop(out, 200)
        out = self.maxpool_contracting(out)

        out = self.contracting_3(out)
        feat_map_3_cropped = clone_and_crop(out, 104)
        out = self.maxpool_contracting(out)

        out = self.contracting_4(out)
        feat_map_4_cropped = clone_and_crop(out, 56)
        out = self.maxpool_contracting(out)

        # produces 1024x28x28 feature map
        out = self.contracting_5(out)

        # expansion path 1 - produces 512x52x52 feature map
        out = self.expanding_1(out, feat_map_4_cropped)
        
        # rest of the expansion paths
        out = self.expanding_2(out, feat_map_3_cropped)
        out = self.expanding_3(out, feat_map_2_cropped)
        out = self.expanding_4(out, feat_map_1_cropped)

        # final 1x1 convolution to map each 64-component feature vector to the desired number of classes
        out = self.class_mapping(out)

        return out

if __name__ == "__main__":
    import numpy as np
    import torch
    from torchsummary import summary

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(5, 3, 572, 572).astype('float32')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(X).to(device)

    model = UNet()
    model = model.to(device)
    
    summary(model, (3, 572, 572))
    print ()
    print (model.forward(X).shape)