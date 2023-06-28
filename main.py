#############################################################
#                       IMPORT LIBRAIRIES                   #
#############################################################

#Pytorch librairies
import torch
from torch import nn 
from torch.utils.data import DataLoader

#Usefull librairies
import numpy as np
from datetime import datetime
import random

#My own functions
from DataLoad import *
from TransNet import *
from utils import *
from Routine import TrainRoutine

#############################################################
#                     INITIALIZATION                        #
#############################################################

random.seed(27)
torch.manual_seed(27)
np.random.seed(27)

#Create directory for checkpoints
currDate = datetime.now()
saveDir ="/media/bralet/Elements/SARDINet/" + str(currDate).replace(' ', '_').replace(':',"-") + "/" 
checkDir(saveDir)

#Initialization
pathInput = "/media/bralet/Elements/DataSpaceNet6/AOI_Subset_SAR/"
pathGT = "/media/bralet/Elements/DataSpaceNet6/AOI_Subset_RGBNIR/"

device = "cuda" if (torch.cuda.is_available()) else "cpu"      #Whether to use gpu or cpu

paths = {
    "pathInput" : pathInput,
    "pathGT" : pathGT,
    "saveDir" : saveDir
}

#############################################################
#                     HYPERPARAMETERS                       #
#############################################################

epochs = 30                #Number of epochs
percentages = [0.7,0.2,0.1]     #Percentage of training and evaluation
batch_size = 32             #Batch size
im_size = 200               #Size of the crops
lr_main = 1e-4              #learning rate for SARDINet
lr_discr = 1e-5             #learning rate for the discriminator
lambdaVal = 0.0005          #Weight for balancing the losses (to be used with discriminator)

hyperparameters = {

    "epochs" : epochs,
    "percentages" : percentages,
    "batch_size" : batch_size,
    "im_size" : im_size,
    "lr_main" : lr_main,
    "lr_discr" : lr_discr,
    "lambdaVal" : lambdaVal
}

#############################################################
#               DATASETS AND DATALOADERS                    #
#############################################################

d_train = ImDataset(im_size,im_size,5, percentages, 0, pathInput = pathInput, pathOutput = pathGT)        #Training dataset
d_evalu = ImDataset(im_size,im_size,5,percentages, 1, pathInput = pathInput, pathOutput = pathGT)         #Evaluation dataset
d_tests = ImDataset(im_size,im_size,5,percentages, 2, pathInput = pathInput, pathOutput = pathGT)         #Evaluation dataset
dataloader = DataLoader(d_train, 1, shuffle = True)                                                             #Training dataloader
dataloader_eval = DataLoader(d_evalu, 1, shuffle = False)                                                       #Evaluation dataloader
dataloader_test = DataLoader(d_tests, 1, shuffle = False)

dataloaders = {
    "dataloader": dataloader,
    "dataloader_eval": dataloader_eval,
    "dataloader_test": dataloader_test
}

#############################################################
#                         NETWORK                           #
#############################################################

adversarial = True
modelPath = ""

modelparams = {
    "adversarial" : adversarial,
    "modelPath": modelPath
}

#############################################################
#                     OPTIMIZATION                          #
#############################################################

#Loss and optimizer
loss_fn = nn.MSELoss(reduction = "none")

#Discriminator loss and optimizer
loss_discr = nn.BCELoss()

optimization = {
    "loss_fn" : loss_fn,
    "loss_discr" : loss_discr
}

#############################################################
#                   NETWORK TRAINING                        #
#############################################################

routine = TrainRoutine(paths, hyperparameters, dataloaders, modelparams, optimization, device)
routine.trainNet()

#############################################################
#                   NETWORK TRAINING                        #
#############################################################

routine.inferNet()