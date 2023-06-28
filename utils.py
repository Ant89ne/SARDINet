#############################################################
#                    IMPORT LIBRAIRIES                      #
#############################################################

#Pytorch librairies
import torch
from torch import nn 
import torchgeometry as geo

#Usefull librairies
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from math import sqrt
from PIL import Image
import copy
import cv2
from tqdm import tqdm

#############################################################
#                    METRICS FUNCTIONS                      #
#############################################################

def getMetrics(pred, l):
    """
    Function to calculate SSIM and MAE metrics
    @input pred :       Prediction to be evaluated
    @input l :          Ground truth values

    @return :           MAE and SSIM values
    """

    #MAE
    l1 = nn.L1Loss(reduction = "mean")              
    mae = l1(pred, l).item()

    #SSIM
    ssim = torch.mean(geo.losses.SSIM(3)(pred, l)).item()       
    
    # l2 = nn.MSELoss(reduction = "mean")
    #sqrt(l2(pred, l).item())

    return mae, ssim


def getMoreMetrics(pred, l):
    """
    Function to calculate 4 metrics
    @input pred :       Prediction to be evaluated
    @input l :          Ground truth values

    @return :           MAE, RMSE, SSIM-3 and SSIM-11 values
    """

    l1 = nn.L1Loss(reduction = "mean")
    l2 = nn.MSELoss(reduction = "mean")
    ssim3 = geo.losses.SSIM(3)
    ssim11 = geo.losses.SSIM(11)

    mae = l1(pred, l).item()
    rmse = l2(pred,l).item()
    ssim3Value = torch.mean(ssim3(pred, l)).item()#sqrt(l2(pred, l).item())
    ssim11Value = torch.mean(ssim11(pred, l)).item()#sqrt(l2(pred, l).item())

    return mae, sqrt(rmse), ssim3Value, ssim11Value

def check_changes(dataloader, model, discr, device):
    """
    Function to calculate the accuracy of the discriminator
    @input dataloader :         Dataloader to be used
    @input model :              Network to be used
    @input discr :              Discriminator to be tested
    @input device :             Device to be used
    """

    with torch.no_grad():
        allFake = torch.zeros((1,2)).to(device)
        allTrue = torch.zeros((1,2)).to(device)
        ddataloader = copy.deepcopy(dataloader)
        for k in range(10):
            #Get a batch of images
            d,l = next(iter(ddataloader))

            #Copy images and ground truth on selected device
            d = d.to(device)
            l = l.to(device)

            #Generation of the network
            pred = model(d)

            #Prediction of the discriminator
            dd_pred_fake = discr(pred)      #Predictions for fake images
            dd_pred = discr(l)              #Predictions for true images

            #Save results
            allFake += dd_pred_fake
            allTrue += dd_pred
        
        print("Real : " + str(allTrue/10.0) + " | Fake : " + str(allFake/10.0))
        


#############################################################
#                    TRAINING LOOPS                         #
#############################################################

def training_loop(dataloader, model, loss_fn, optimizer, batch_size = 16, device = "cpu"):
    """
    Training loop for classical network
    @input dalaoader :          Dataloader to be used
    @input model :              Neural network to be trained
    @input loss_fn :            Loss to be minimized
    @input optimizer :          Optimizer to be used
    @input batch_size :         Number of images per batch (default : 16)
    @input device :             Device to be used (default : cpu)
    
    @return :                   MSE, MAE and SSIM5 averaged for training data
    """
    
    #Switch model to training mode
    model.train()

    #Initialization
    TLoss = 0
    mae, ssim = 0,0

    #Zeroing gradient (avoid accumulation)
    optimizer.zero_grad()

    #Number of iterations
    a = len(dataloader)//batch_size

    #Routine
    for i, (d, l) in enumerate(tqdm(dataloader)) :
        #Copy data and label to the selected device
        d = d.to(device)
        l = l.to(device)


        #Prediction
        pred = model(d)

        #Loss calculation
        if i > a*batch_size :
            fact = 1.0 / (len(dataloader) - batch_size*a)
        else : 
            fact = 1.0 / batch_size
        loss = torch.mean(loss_fn(pred,l) )
        
        #Save Loss
        TLoss += loss.item()

        #Loss backpropagation
        loss *= fact
        loss.backward()
        
        #Metrics calculation
        maec, ssimc = getMetrics(pred, l)
        mae += maec
        ssim += ssimc

        #Optimization when batches are ready
        if not (i+1)%batch_size :
            optimizer.step()

            #Zeroing the gradient
            optimizer.zero_grad()

    #Averaging
    mae /= len(dataloader)
    ssim /= len(dataloader)
    TLoss /= len(dataloader)
    print("Training : Loss : " + str(TLoss)+ " | MAE : " + str(mae) + " | SSIM : " + str(ssim))
    
    return [TLoss, mae, ssim]


def adversarial_training__loop(dataloader, models, loss_fns, optimizers, batch_size = 16, device = "cpu", lambdaVal = 0.1):
    """
    Adversarial training of the network

    @input dataloader :         Dataloader to be used
    @input models :             List with the generator and the discriminator networks
    @input loss_fns :           List of the losses (generator and disccriminator)
    @input optimizers :         List of the optimizers (generator and discriminator)
    @input batch_size :         Number of images per batch (default : 16)
    @input device :             Device to be used for the training (default : cpu)
    @input lambdaVal :          Weight to be used to balance the losses (default : 0.1)

    @return :                   MSE, MAE, SSIM-5 and CE of the discriminator
    """

    #Switch networks to training mode
    model, discr = models[0].train(), models[1].train()

    #Extract losses and optimizers
    optimizer, optim_discr = optimizers[0], optimizers[1]
    loss_fn, loss_discr = loss_fns[0], loss_fns[1]
    
    #Initializations
    TLoss = 0
    DLoss = 0
    DDLoss = 0
    mae, ssim = 0,0

    #Zeroing gradients (avoid accumulation)
    optimizer.zero_grad()
    optim_discr.zero_grad()
    
    #Number of iterations
    a = len(dataloader)//batch_size

    #Copy of the dataloader for training the discriminator
    ddataloader = copy.deepcopy(dataloader)

    #Routine
    for i, (d, l) in enumerate(tqdm(dataloader)) :
        #Send data to selected device
        d = d.to(device)
        l = l.to(device)

        #Model prediction
        pred = model(d)

        #Loss calculation
        if i > a*batch_size :
            fact = 1.0 / (len(dataloader) - batch_size*a)
        else : 
            fact = 1.0 / batch_size

        #Discriminator applied on generated images
        dd_pred_fake = discr(pred.detach())
        #Discriminator applied on true images
        dd_pred = discr(l)


        #Loss on true images
        loss_dd = loss_discr(dd_pred, torch.Tensor([[0,1]]).to(device)) 
        loss_dd = loss_dd / 2.0 * lambdaVal
        DDLoss += loss_dd.item()
        loss_dd *= fact
        loss_dd.backward()
        
        #Loss on fake images
        loss_dd = loss_discr(dd_pred_fake, torch.Tensor([[1,0]]).to(device))
        loss_dd /= 2.0
        loss_dd *= lambdaVal
        DDLoss += loss_dd.item()
        loss_dd *= fact
        loss_dd.backward()
        
        # Optimization when batches are ready
        if not (i+1)%batch_size :
            #Optimize the generator
            optim_discr.step()
            
            #Zeroing generator gradients
            optimizer.zero_grad()

            #Generator routine
            for i in range(batch_size):
                #Get data and copy on selected device
                d,l = next(iter(ddataloader))
                d = d.to(device)
                l = l.to(device)
                
                #Prediction
                pred = model(d)
                d_pred = discr(pred)

                #Loss on image reconstruction
                loss = torch.mean(loss_fn(pred,l))
                #Discriminator loss
                loss_d = loss_discr(d_pred,torch.Tensor([[0,1]]).to(device))
                #Total loss
                loss = loss + loss_d*lambdaVal
                
                #Save Losses
                TLoss += loss.item()
                DLoss += loss_d.item()*lambdaVal

                loss *= fact

                #Backpropagation
                loss.backward()
                
                #Calculate metrics
                maec, ssimc = getMetrics(pred, l)
                mae += maec
                ssim += ssimc

            #Optimizer step
            optimizer.step()
            #Zeroing dicriminator gradients
            optim_discr.zero_grad()
    
    #Average metrics
    mae /= len(dataloader)
    ssim /= len(dataloader)
    TLoss /= len(dataloader)
    DLoss /= len(dataloader)
    DDLoss /= len(dataloader)

    check_changes(dataloader, model, discr, device)
    print("Training : Loss : " + str(TLoss)+ " | DLoss : " + str(DLoss) + " | DDLoss : " + str(DDLoss) + " | MAE : " + str(mae) + " | SSIM : " + str(ssim))
    return [TLoss, mae, ssim, DLoss]

#############################################################
#                   EVALUATION LOOPS                        #
#############################################################

def eval_loop(dataloader, model, loss_fn, device):
    """
    Main loop routine for the evaluation of the network

    @input dataloader :         Dataloader to be used
    @input model :              Model to be evaluated
    @input loss_fn :            Loss used for training
    @input device :             Device to be used

    @return :                   All metrics : RMSE, MAE, SSIM3 and SSIM11
    """

    #Switch to evaluation mode
    model.eval()
    ELoss = 0
    mae, rmse, ssim3, ssim11 = 0,0, 0, 0

    #Routine
    for i, (d, l) in enumerate(tqdm(dataloader)) :
        #Copy images and ground truth on selected device
        d = d.to(device)
        l = l.to(device)

        #Prediction
        pred = model(d)

        #Loss calculation
        loss = torch.mean(loss_fn(pred,l))#* l)#loss_fn(pred, l)
        
        #Get metrics MAE and RMSE
        maec, rmsec, ssim3c, ssim11c= getMoreMetrics(pred, l)
        mae += maec
        rmse += rmsec     
        ssim3 += ssim3c
        ssim11 += ssim11c
        
        #Save Loss
        ELoss += loss.item()

    #Average metrics
    ELoss /= len(dataloader)
    mae /= len(dataloader)
    rmse /= len(dataloader)
    ssim3 /= len(dataloader)
    ssim11 /= len(dataloader)

    print("Evaluation : Loss : " + str(ELoss) + " | MAE : " + str(mae) + " | RMSE : " + str(rmse) + " | SSIM_3 : " + str(ssim3) + " | SSIM_11 : " + str(ssim11) )
    
    return [ELoss, mae, ssim3]

#############################################################
#                    VISUALIZATION                          #
#############################################################

def visLoss(TrainLosses, EvalLosses):
    """
    Function used to visualize the final loss evolution

    @input TrainLosses :        Training loss and metrics
    @input EvalLosses :         Evaluation loss and metrics
    """
    plt.figure(figsize=(20,10))
    
    #Loss visualization
    plt.subplot(221)
    plt.plot([k for k in range(len(TrainLosses))], TrainLosses[:,0], 'r-')
    plt.plot([k for k in range(len(EvalLosses))], EvalLosses[:,0], 'b-')
    if len(TrainLosses[0,:]) == 4 :
        plt.plot([k for k in range(len(TrainLosses))], TrainLosses[:,-1], 'g-')
    plt.title("Loss over epochs")
    plt.legend(["Training Loss", "Evaluation Loss"])
    
    #Loss zoomed
    plt.subplot(223)
    plt.plot([k for k in range(len(TrainLosses))], TrainLosses[:,0], 'r-')
    plt.plot([k for k in range(len(EvalLosses))], EvalLosses[:,0], 'b-')
    if len(TrainLosses[0,:]) == 4 :
        plt.plot([k for k in range(len(TrainLosses))], TrainLosses[:,-1], 'g-')
    plt.title("Loss over epochs - zoom")
    plt.axis([0,len(TrainLosses), 0, np.maximum(np.sort(EvalLosses[:,0])[-5], np.sort(TrainLosses[:,0])[-5])])
    plt.legend(["Training Loss", "Evaluation Loss"])

    #MAE visualization
    plt.subplot(222)
    plt.plot([k for k in range(len(TrainLosses))], TrainLosses[:,1], 'r-')
    plt.plot([k for k in range(len(EvalLosses))], EvalLosses[:,1], 'b-')
    plt.title("MAE over epochs")
    plt.legend(["Training MAE", "Evaluation MAE"])
    plt.axis([0,len(TrainLosses), 0, np.maximum(np.sort(EvalLosses[:,1])[-5], np.sort(TrainLosses[:,1])[-5])])
    
    #SSIM visualization
    plt.subplot(224)
    plt.plot([k for k in range(len(TrainLosses))], TrainLosses[:,2], 'r-')
    plt.plot([k for k in range(len(EvalLosses))], EvalLosses[:,2], 'b-')
    plt.title("RMSE over epochs")
    plt.legend(["Training SSIM", "Evaluation SSIM"])
    plt.axis([0,len(TrainLosses), 0, np.maximum(np.sort(EvalLosses[:,2])[-5], np.sort(TrainLosses[:,2])[-5])])

def visIm(model, dataset, epoch, dir, nbIm = 4, saveSep = False):
    """
    Function used to visualize images after training
    
    @input model :          Network to be used
    @input dataset :        Dataset to be used
    @input epoch :          Number of training epochs already done
    @input dir :            Directory where to save the images
    @input nbIm :           Number of images to visualize (defualt : 4)
    @input saveSep :        Separate the saving of GT and prediction (default : False)
    """
    
    #Create directory if not yet existing
    checkDir(dir)

    if nbIm == -1 :
        nbIm = len(dataset)

    with torch.no_grad():
        random.seed(25)
        for k in range(nbIm):
            #Get an image
            d,l = dataset[k]

            #Copy network on cpu
            model.cpu()

            #Reshape image
            s = d.shape
            d = torch.reshape(d, (1, s[0], s[1], s[2]))
            
            #Prediction
            pred = model(d)

            #Transpose for visualization purposes
            pred = torch.transpose(pred,1,3)
            l = torch.transpose(l, 0,2)
            d = torch.transpose(d, 1,3)

            #Save images separately
            if saveSep :
                cv2.imwrite(dir + "soloIm" + str(k) + "_" + str(epoch) + ".png", np.array(pred[0,:,:,0:3]*255).astype(int))
                cv2.imwrite(dir + "soloGT" + str(k) + "_" + str(epoch) + ".png", np.array(l[:,:,0:3]*255).astype(int))

            #Optical prediction
            plt.figure(figsize=(20,10))
            plt.clf()
            plt.subplot(221)
            plt.imshow(pred[0,:,:,:3])
            plt.title("RGB Prediction")

            #Optical ground truth
            plt.subplot(222)
            plt.imshow(l[:,:,:3])
            plt.title("RGB Ground Truth")

            #Near Infra-Red prediction
            plt.subplot(223)
            plt.imshow(pred[0,:,:,3], cmap="gray")
            plt.title("NIR Prediction")

            #Near Infra-Red ground truth
            plt.subplot(224)
            plt.imshow(l[:,:,3], cmap="gray")
            plt.title("NIR Ground Truth")

            #Save images
            plt.savefig(dir + "im" + str(k) + "_" + str(epoch))
            plt.close()

            #Input visualization
            plt.figure(figsize=(20,10))
            plt.clf()
            plt.subplot(221)
            plt.imshow(d[0,:,:,0], cmap="gray")
            plt.title("HH")

            plt.subplot(222)
            plt.imshow(d[0,:,:,1], cmap="gray")
            plt.title("HV")

            plt.subplot(223)
            plt.imshow(d[0,:,:,2], cmap="gray")
            plt.title("VH")

            plt.subplot(224)
            plt.imshow(d[0,:,:,3], cmap="gray")
            plt.title("VV")

            plt.savefig(dir + "input" + str(k))
            plt.close()
    random.seed()

def reconstructFull(dataset, model, nbIms = 1):
    """
    Visualize results on the full image
    
    @input dataset :        Dataset to be used
    @input model :          Network to be used
    @input nbIms :          Number of images to be reconstructed (default : 1)
    """

    with torch.no_grad():
        random.seed(25)
        for i in range(nbIms) :
            #Random index
            rval = random.randint(0, len(dataset.SarImFiles))
            
            #Load SAR image
            fullIm = dataset.SarImFiles[rval]
            imSar = Image.open(fullIm)
            imSar = np.array(imSar)/255.0
            
            #Load Optical ground truth
            imOpt = Image.open(dataset.OptImFiles[rval])
            imOpt = np.array(imOpt)/255.0

            #Shape of the image
            s = imSar.shape

            #Increment for each successive crop (overlapping)
            miR = dataset.rows//2
            miC = dataset.columns//2
            incR = miR
            incC = miC 

            imFin = np.zeros(s)
            imFinBis = np.zeros(s)
            imOptBis = np.zeros(s)

            nbVals = np.ones(s)


            for k in range(miR,s[0]+miR,incR):
                for l in range(miC,s[1]+miC,incC):
                    #Index of the crop
                    k = min(k, s[0] - miR)
                    l = min(l, s[1] - miC)

                    #Crop image
                    imToUse = torch.Tensor(imSar[k-miR:k+miR, l-miC:l+miC, :]).to("cpu")
                    imToUse = torch.reshape(imToUse, (1, dataset.rows, dataset.columns, s[2]))
                    imToUse = torch.transpose(imToUse, 1,3)
                    
                    #Network prediction
                    pred = model(imToUse)
                    pred = torch.transpose(pred, 1,3)
                    pred = pred.numpy()

                    #Save prediction
                    imFin[k-miR:k+miR, l-miC:l+miC, :] += pred[0,:,:,:]
                    nbVals[k-miR:k+miR, l-miC:l+miC, :] += 1

            #Average due to overlaps
            imFin /= nbVals

            #Restructring for visualization purposes
            imFinBis[:,:,0], imFinBis[:,:,1], imFinBis[:,:,2], imFinBis[:,:,3] = imFin[:,:,2], imFin[:,:,1], imFin[:,:,0], imFin[:,:,3]
            imOptBis[:,:,0], imOptBis[:,:,1], imOptBis[:,:,2], imOptBis[:,:,3] = imOpt[:,:,2], imOpt[:,:,1], imOpt[:,:,0], imOpt[:,:,3]
            s = imFinBis.shape

            print(getMetrics(torch.Tensor(imFinBis).reshape(1,s[0], s[1], s[2]), torch.Tensor(imOptBis).reshape((1,s[0], s[1], s[2]))))

            #Visualization
            plt.figure()
            plt.clf()
            plt.subplot(211)
            plt.imshow(imFinBis[:,:,0:3])
            plt.title("Predicted Values")
            
            plt.subplot(212)
            plt.imshow(imOptBis[:,:,0:3])
            plt.title("Ground truth")

            plt.show()
            plt.close()

    
    random.seed()


#############################################################
#                         UTILS                             #
#############################################################


def checkpoint(model, TLoss, ELoss, epoch, dir):
    """
    Save the network

    @input model :          Network to be saved
    @input TLoss :          Training loss evolution to be saved
    @input ELoss :          Evaluation loss evolution to be saved
    @input dir :            Directory where to save
    """

    #Network saving
    torch.save(model, dir + "model_" + str(epoch) + ".pt")

    #Loss visualization
    visLoss(np.array(TLoss), np.array(ELoss))
    
    #Loss saving
    plt.savefig(dir + "Loss_" + str(epoch) + ".png")
    plt.close()

def checkDir(path): 
    """
    Check if the path exists, if not, creates it

    @input path :           Path to be checked
    """
    if not os.path.isdir(path): 
        os.mkdir(path)



# from DataLoad import *

# m = torch.load("/home/bralet/Code/Translation/2000epochs_1e-4_weighhtLoss/model_2000.pt", map_location="cpu")
# path = "/media/bralet/Elements/DataBis/SpaceNet6/3DShapes/"
# d_train = ImDataset(path, 200,200,5, [0.8,0.2], 0)
# reconstructFull(d_train, m, 10)


# saveDir ="/home/bralet/Code/Translation/2022-01-27_11:54:25.513517/TrainingIms/" 
# checkDir(saveDir)

# visIm(m, d_train, 0, saveDir)
