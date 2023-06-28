#############################################################
#                    IMPORT LIBRAIRIES                      #
#############################################################

#Pytorch librairies
import torch
from torch import nn

#Usefull librairies
import matplotlib.pyplot as plt

#############################################################
#                  SEPARABLE CONVOLUTIONS                   #
#############################################################

class SepConvBlock(nn.Module):
    """
    Class for the separable convolution block
    """

    def __init__(self, inputChannels, outputChannels, kernel) -> None:
        """
        Initialization

        @input inputChannels:       Number of channels as input
        @input outputChannels:      Number of willing output channels
        @input kernel:              Size of the convolutional kernels
        """

        super(SepConvBlock, self).__init__()

        #Dephtwise convolution : each channel is convolved separately and independently (groups argument)
        self.depthwiseConv = nn.Conv2d(inputChannels, inputChannels, kernel, 1,1, groups=inputChannels, padding_mode="reflect")
        #Pointwise convolution : along the channel dimension        
        self.pointwise = nn.Conv2d(inputChannels, outputChannels, 1)
        
        # #Batch-Normalization
        # self.bn = nn.BatchNorm2d(outputChannels)


    def forward(self, x):
        """
        Forward pass of the block

        @input x :      Data to be forwarded
        """
        ret = self.depthwiseConv(x)
        ret = self.pointwise(ret)
        # ret = self.bn(ret)

        return ret        

#############################################################
#                    DECODER BRANCHES                       #
#############################################################

class DecBranch(nn.Module):
    """
    Class for a single branch of the decoder of SARDNINet
    """

    def __init__(self, inputChannel) -> None:
        """
        Initialization

        @input inputChannel :       Number of inpt channels
        """
        super(DecBranch, self).__init__()
        
        #List of the layers of the branch
        self.layers = []

        #Number of kernels for each layer
        decKernel = [inputChannel, 64,16,1]

        #Creation of the layers
        for k in range(2):
            # self.layers.append(nn.ConvTranspose2d(decKernel[k], decKernel[k+1], 3, 2, 1, output_padding=1))       #Worked worst than Upsample + Conv2D
            self.layers.append(nn.Upsample(scale_factor=2, mode = "nearest"))                               
            self.layers.append(nn.Conv2d(decKernel[k], decKernel[k+1], 5, 1, 2, padding_mode="reflect"))   
            self.layers.append(nn.Conv2d(decKernel[k+1], decKernel[k+1], 5, 1, 2, padding_mode="reflect"))
            self.layers.append(nn.LeakyReLU())
        
        #Final layer block (no ReLU)
        # self.layers.append(nn.ConvTranspose2d(decKernel[-2], decKernel[-1], 3, 2, 1, output_padding=1))
        self.layers.append(nn.Upsample(scale_factor=2, mode = "nearest"))
        self.layers.append(nn.Conv2d(decKernel[-2], decKernel[-1], 5, 1, 2, padding_mode="reflect"))
        self.layers.append(nn.Conv2d(decKernel[-1], decKernel[-1], 5, 1, 2, padding_mode="reflect"))
        
        #Create a sequential network to easy the forward pass
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        Forward pass
        
        @input x :      Data to be forwarded
        """
        ret = self.layers(x)
        return ret

#############################################################
#                     DISCRIMINATOR                         #
#############################################################

class Discriminator(nn.Module):
    """
    Class for the discriminator network used for the adversarial training
    """
    def __init__(self) -> None:
        """
        Initialization
        """
        super(Discriminator, self).__init__()

        #List of the layers
        self.layers = []
        #Number of kernels per layer
        kernels = [4,128,396,512]
        #Size of the kernels per layer
        sizes = [3,3,3]

        #Creation of the layers
        for k in range(3):
            self.layers.append(nn.Conv2d(kernels[k], kernels[k+1], sizes[k], 1, sizes[k]//2))
            self.layers.append(nn.MaxPool2d(2))
            self.layers.append(nn.ReLU())
        
        #Final Dense layer
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(kernels[-1]*25*25, 2))
        
        #Switch to Sequential network for easy purposes
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        """
        Forward pass

        @input x :          Data to be forwarded
        """

        ret = self.layers(x)
        
        #Final softmax to  get probability like output
        ret = nn.Softmax(dim = 1)(ret)
        
        return ret

#############################################################
#                       SARDINet                            #
#############################################################

class SARDINet(nn.Module):
    """
    Class for SARDINet architecture
    """

    def __init__(self):
        """
        Initialization
        """

        super(SARDINet, self).__init__()
        
        #Input conditionning
        inpCondChannels = [4,96,256]
        self.inputCond = []
        for k in range(2):
            # self.inputCond.append(nn.Conv2d(inpCondChannels[k], inpCondChannels[k],3,1,1))
            self.inputCond.append(nn.Conv2d(inpCondChannels[k],inpCondChannels[k+1],3,2-k,1, padding_mode="reflect"))
            # self.inputCond.append(nn.BatchNorm2d(inpCondChannels[k+1]))
            self.inputCond.append(nn.ReLU())
        self.inputCond = nn.Sequential(*self.inputCond)

        encChannels = [256, 396, 512]
        #Top encoder branch
        self.encoderB1 = []
        self.encoderB1.append(SepConvBlock(encChannels[0], encChannels[1], 3))
        self.encoderB1.append(nn.MaxPool2d(2))
        self.encoderB1.append(nn.Conv2d(encChannels[1], encChannels[2], 1))#, stride = 2))
        self.encoderB1.append(nn.MaxPool2d(2))
        self.encoderB1 = nn.Sequential(*self.encoderB1)

        #Middle encoder branch
        self.encoderB2 = []
        self.encoderB2.append(nn.Conv2d(encChannels[0], encChannels[1], 1))#, stride=2))
        self.encoderB2.append(nn.ReLU())
        self.encoderB2.append(nn.MaxPool2d(2))
        self.encoderB2.append(SepConvBlock(encChannels[1], encChannels[1], 3))
        self.encoderB2.append(nn.ReLU())
        self.encoderB2.append(SepConvBlock(encChannels[1], encChannels[2], 3))
        self.encoderB2.append(nn.MaxPool2d(2))
        self.encoderB2 = nn.Sequential(*self.encoderB2)

        #Bottom encoder branch
        self.encoderB3 = []
        self.encoderB3.append(SepConvBlock(encChannels[0], encChannels[1], 3))
        self.encoderB3.append(nn.ReLU())
        self.encoderB3.append(SepConvBlock(encChannels[1], encChannels[1], 3))
        self.encoderB3.append(nn.MaxPool2d(2))
        self.encoderB3.append(SepConvBlock(encChannels[1], encChannels[2], 3))
        self.encoderB3.append(nn.MaxPool2d(2))
        self.encoderB3 = nn.Sequential(*self.encoderB3)

        #Decoder branches
        self.decoderB1 = DecBranch(encChannels[-1])
        self.decoderB2 = DecBranch(encChannels[-1])
        self.decoderB3 = DecBranch(encChannels[-1])
        self.decoderB4 = DecBranch(encChannels[-1])

    def forward(self, x):
        """
        Forward pass

        @input x :      Data to be forwarded
        """
        ret = self.inputCond(x)

        ret1 = self.encoderB1(ret)
        ret2 = self.encoderB2(ret)
        ret3 = self.encoderB3(ret)

        fus = ret1 + ret2 + ret3

        d1 = self.decoderB1(fus)
        d2 = self.decoderB2(fus)
        d3 = self.decoderB3(fus)
        d4 = self.decoderB4(fus)

        finalRet = torch.cat((d1,d2,d3,d4), dim = 1)
        
        return finalRet
