#############################################################
#                       IMPORT LIBRAIRIES                   #
#############################################################

#Pytorch librairies
import torch
from torch import nn 
from torch.utils.data import DataLoader, Dataset

#Usefull librairies
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
from math import floor

#############################################################
#                       CLASS CREATION                      #
#############################################################

class ImDataset(Dataset):
    """
    Class used as dataset for the training of SARDINet

    Load and crop images randomly at each epoch
    """
    
    def __init__(self, imRows, imColumns, smplIm, percentages, id, pathInput = None, pathOutput = None) -> None:
        """
        Initialization of the class

        @input imFolder :           Folder where to find all images (prefer to use pathInput and pathOutput)
        @input imRows :             Size of the crop (rows)
        @input imColumns :          Size of the crop (columns)
        @input smplIm :             Number of crops per image
        @input percentages :        Percentages of training vs evaluation
        @input id :                 Whether this is a Training (0) or Evaluation (1) dataset
        @input pathInput :          Path to the input images (default : None)
        @input pathOutput :         Path to the ground truth images (default : None)
        """
        
        super(ImDataset, self).__init__()

        #Look for the images to be used
        
        self.SarImFiles = sorted([pathInput + f for f in os.listdir(pathInput) if f.endswith(".tif")])
        self.OptImFiles = sorted([pathOutput + f for f in os.listdir(pathOutput) if f.endswith(".tif")])
    

        if len(self.OptImFiles) != len(self.SarImFiles) :
            print("ERROR : NOT SAME NUMBER OF SAR AND OPT IMAGES")

        #Initialization
        self.rows = imRows
        self.columns = imColumns
        self.smpPerIm = smplIm

        #Selection of the images that are part of the dataset
        i = int(np.sum(percentages[:id])*len(self.SarImFiles))
        j = int(np.sum(percentages[:id+1]) * len(self.SarImFiles))
        self.SarImFiles = self.SarImFiles[i:j]
        self.OptImFiles = self.OptImFiles[i:j]

        #Size of the dataset
        self.len = len(self.SarImFiles)

    def __len__(self):
        """
        Get the size of the dataset

        @return :       Size of the dataset
        """
        #Size calculated as number of images times number of samples per image
        return self.smpPerIm * self.len

    def __getitem__(self, index) :
        """
        Function used to access a sample of the dataset
        
        @input index:           Index of the sample to display

        @return:                SAR croped image and its optical ground truth
        """

        #Index of the file
        i = index // self.smpPerIm

        #Read optical image
        im = Image.open(self.OptImFiles[i])
        im = np.array(im)
        
        #Read SAR image
        imSar = Image.open(self.SarImFiles[i])
        imSar = np.array(imSar)
        
        #Size of the image
        s = im.shape

        #select random coordinates
        i,j = random.random(), random.random()
        i *= s[0] - 1 - self.rows
        j *= s[1] - 1 - self.columns
        i, j  = int(floor(i)), int(floor(j))

        #Crop the images and normalize between [0-1]
        newIm = torch.Tensor(im[i:i+self.rows, j:j+self.columns, :]/255.0)
        newImSar = torch.Tensor(imSar[i:i+self.rows, j:j+self.columns, :]/255.0)

        #Transpose images so that channels are first and then spatial dimensions
        newIm = torch.transpose(newIm, 0,2)
        newImSar = torch.transpose(newImSar, 0, 2)
        # self.show(newIm)

        return newImSar, newIm

    
    def show(self, im, tit = ["", "" ,"", ""]):
        """
        Function to show a given image
        
        @input im:          Image to display
        @input tit :        Title for each subplot
        """
        plt.figure()

        #Display each channel independently
        for k in range(len(tit)):
            plt.subplot(2,2,k+1)
            plt.imshow(im[:,:,k])
            plt.title(tit[k])


    def showFull(self, index):
        """
        Function to show a sample
        """

        #Index of the file
        i = index // self.smpPerIm

        #Read Optical image
        im = Image.open(self.OptImFiles[i])
        im = np.array(im)

        #Display Optical image
        tit = ["Red", "Green", "Blue", "NIR"]
        self.show(im, tit)

        #Read SAR image
        im = Image.open(self.SarImFiles[i])
        im = np.array(im)

        #Display SAR image
        tit = ["HH", "HV", "VH", "VV"]
        self.show(im, tit)


#########
# TESTS #
#########

# path = "/media/bralet/Elements/DataBis/SpaceNet6/3DShapes/"
# d = ImDataset(path, 200,200,5)
# print(len(d))
# d.showFull(3)
# i,l = d[3]
# print(np.max(i))
# print(np.max(l))

