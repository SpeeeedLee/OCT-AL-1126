import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# Defining a nested function that takes a module as input and initializes its weights
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        # Initializing the weights based on the chosen initialization method
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            # Checking if the module has a bias attribute and if it is not None
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # Checking if the module is an instance of InstanceNorm2d class
        elif classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    # Applying the initialization function to all modules in the network
    net.apply(init_func)

    
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=1),
            nn.InstanceNorm2d(ch_out),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up(x)
        return x


class Optim_U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, filter_size=32, USE_DS=False, USE_DFS=False):
        super(Optim_U_Net, self).__init__()
        """
        Binary Segmentation U-Net for Cell Nuclei Detection
        
        Args:
            img_ch: input channel number. Grayscale:1 / RGB:3
            output_ch: output segmentation channel (1 for binary segmentation)
            filter_size: number of filters in first convolution layer
            USE_DS: deprecated, kept for compatibility
            USE_DFS: deprecated, kept for compatibility
        """
        if USE_DS or USE_DFS:
            print("Warning: USE_DS and USE_DFS are deprecated in binary segmentation mode.")
        
        # Down-sampling modules
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=filter_size)
        self.Conv2 = conv_block(ch_in=filter_size, ch_out=filter_size*2)
        self.Conv3 = conv_block(ch_in=filter_size*2, ch_out=filter_size*4)
        self.Conv4 = conv_block(ch_in=filter_size*4, ch_out=filter_size*8)
        self.Conv5 = conv_block(ch_in=filter_size*8, ch_out=filter_size*16)
        self.Conv6 = conv_block(ch_in=filter_size*16, ch_out=filter_size*32)

        # Up-sampling modules
        self.Up6 = up_conv(ch_in=filter_size*32, ch_out=filter_size*16)
        self.Up_conv6 = conv_block(ch_in=filter_size*32, ch_out=filter_size*16)
        self.Up5 = up_conv(ch_in=filter_size*16, ch_out=filter_size*8)
        self.Up_conv5 = conv_block(ch_in=filter_size*16, ch_out=filter_size*8)
        self.Up4 = up_conv(ch_in=filter_size*8, ch_out=filter_size*4)
        self.Up_conv4 = conv_block(ch_in=filter_size*8, ch_out=filter_size*4)
        self.Up3 = up_conv(ch_in=filter_size*4, ch_out=filter_size*2)
        self.Up_conv3 = conv_block(ch_in=filter_size*4, ch_out=filter_size*2)
        self.Up2 = up_conv(ch_in=filter_size*2, ch_out=filter_size)
        self.Up_conv2 = conv_block(ch_in=filter_size*2, ch_out=filter_size)
        
        # Final output layer (1 channel for binary segmentation)
        self.Conv_1x1 = nn.Conv2d(filter_size, output_ch, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        """
        Forward pass with automatic sigmoid activation
        
        INPUT:
            x: input image - Size: [batch_size, img_ch, height, width]

        OUTPUT:
            output: cell nuclei probability map - Size: [batch_size, 1, height, width]
                    Values in range [0, 1] after sigmoid
        """
        # Encoder (Down-sampling)
        x1 = self.Conv1(x)
        
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        x6 = self.Maxpool(x5)
        x6 = self.Conv6(x6)

        # Decoder (Up-sampling with skip connections)
        d6 = self.Up6(x6)
        d6 = torch.cat((x5, d6), dim=1)
        d6 = self.Up_conv6(d6)
        
        d5 = self.Up5(d6)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        # Final prediction with sigmoid activation
        logits = self.Conv_1x1(d2)
        output = torch.sigmoid(logits)
        
        return output